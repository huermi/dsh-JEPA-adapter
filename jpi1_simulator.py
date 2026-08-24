"""
JPI-1 实践内核模拟器 — 核心机制验证
====================================
用 numpy 实现架构的核心循环（不依赖 ML 框架），在动态网格世界运行，
产出: E1/E2 能量曲线、行动分布、目标切换、记忆增长、探索覆盖率。

组件映射（对应架构文档）:
  - 环境: 动态网格世界（物品漂移 + 光源移动）→ "身体层"
  - 编码器: 状态 → s∈R^8（可学习投影 + SIGReg 式归一化）→ "感知层"
  - 预测器: MLP (s,a) → Δŝ 残差预测 → "世界模型"
  - 两级能量: E1=预测误差(训练/惊讶); E2=累积误差改进(价值/探索)
  - 目标: 记忆采样/插值 → "Configurator 目标生成"
  - 规划: 简化 CEM (K候选×H步, 仅前向) → "规划器"
  - 记忆: surprise 门控写入 + 最近检索 → "记忆层"
  - 在线学习: AdaJEPA 式 1 步梯度, 编码器低 lr → "在线适应"
"""
import numpy as np
from dataclasses import dataclass, field
from collections import deque


def make_rng(seed: int = 42):
    return np.random.RandomState(seed)


# ─── 环境: 动态网格世界 ─────────────────────────────────────
WORLD_N = 6            # 6×6 网格
N_ITEMS = 3            # 3 个物品 (各自有属性, 缓慢漂移)
N_ACTIONS = 5          # 上/下/左/右/WAIT
ACTION_DELTA = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]


def env_state(t: int, static: bool = False) -> np.ndarray:
    """生成世界状态: [物品位置x,y,属性v ×3] = 9d
    static=True → 世界完全静止 (预测可完全学会 → 验证死寂)"""
    item_pos = []
    for i in range(N_ITEMS):
        if static:
            cx = 1.0 + i * 1.5
            cy = 2.0 + i * 0.8
            v = 0.5 + i * 0.2
        else:
            cx = 2.5 + 1.8 * np.sin(t * 0.01 + i * 2.1)
            cy = 2.5 + 1.8 * np.cos(t * 0.013 + i * 1.7)
            v = np.sin(t * 0.02 + i)
        item_pos += [cx, cy, v]
    return np.array(item_pos, dtype=np.float32)


def item_positions(t: int, static: bool = False) -> list:
    """物品的 (x, y) 位置列表 — 供价值锚计算 (锚来自 E1 之外, 硬编码)"""
    return [(1.0 + i * 1.5, 2.0 + i * 0.8) for i in range(N_ITEMS)] if static else \
           [(2.5 + 1.8 * np.sin(t * 0.01 + i * 2.1),
             2.5 + 1.8 * np.cos(t * 0.013 + i * 1.7)) for i in range(N_ITEMS)]


def value_anchor(agent_pos: np.ndarray, items: list, sigma: float = 1.2) -> float:
    """价值锚: agent 离物品越近, 锚值越高 (硬编码 intrinsic value, 与预测误差无关).
    模拟"生存/资源需求" — 靠近物品有内在价值, 世界完全可预测时仍提供行动理由."""
    a = np.asarray(agent_pos, dtype=np.float32)
    d_min = min(np.linalg.norm(a - np.array([x, y], dtype=np.float32)) for x, y in items)
    return float(np.exp(-d_min / sigma))


class Encoder:
    """感知层: 世界状态 → s∈R^8 (可学习投影 + 归一化)"""
    def __init__(self, d_in: int, d_out: int = 8, lr: float = 0.001, seed: int = 42):
        rng = make_rng(seed)
        self.W = rng.randn(d_in, d_out).astype(np.float32) * 0.1
        self.b = np.zeros(d_out, dtype=np.float32)
        self.lr = lr
        self.var_ema = np.ones(d_out, dtype=np.float32)  # SIGReg 方差跟踪

    def encode(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(np.dot(x, self.W) + self.b)
        # SIGReg 式: 方差归一化防止坍缩
        return h / (np.linalg.norm(h) + 1e-6)

    def step(self, x: np.ndarray, grad: np.ndarray, lr_scale: float = 1.0):
        g = np.clip(grad, -0.5, 0.5)
        self.W -= self.lr * lr_scale * g
        self.b -= self.lr * lr_scale * np.sum(grad, axis=0) if grad.ndim > 1 else self.lr * lr_scale * grad


class Predictor:
    """世界模型: (s, a_onehot) → Δŝ 残差预测, 两层 MLP"""
    def __init__(self, s_dim: int = 8, h_dim: int = 16,
                 lr_pred: float = 0.05, lr_enc: float = 0.001, seed: int = 42):
        rng = make_rng(seed)
        self.s_dim, self.h_dim = s_dim, h_dim
        n_a = N_ACTIONS
        self.W1 = rng.randn(s_dim + n_a, h_dim).astype(np.float32) * 0.2
        self.W2 = np.zeros((h_dim, s_dim), dtype=np.float32)
        self.lr_pred, self.lr_enc = lr_pred, lr_enc

    def predict_delta(self, s: np.ndarray, a: int) -> np.ndarray:
        x = np.concatenate([s, np.eye(N_ACTIONS)[a]])
        h = np.maximum(0, np.dot(x, self.W1))
        d = np.dot(h, self.W2)
        n = np.linalg.norm(d)
        return d if n < 5 else d * 5.0 / n

    def step(self, s: np.ndarray, a: int, target_delta: np.ndarray):
        """AdaJEPA 式单步梯度: 编码器低 lr, 预测器正常 lr"""
        x = np.concatenate([s, np.eye(N_ACTIONS)[a]])
        h = np.maximum(0, np.dot(x, self.W1))
        pred = np.dot(h, self.W2)
        err = pred - target_delta
        dW2 = np.outer(h, err)
        dH = np.dot(err, self.W2.T) * (h > 0)
        dW1 = np.outer(x, dH)
        self.W2 -= self.lr_pred * np.clip(dW2, -0.5, 0.5)
        self.W1 -= self.lr_enc * np.clip(dW1, -0.5, 0.5)
        return float(np.mean(err ** 2))


class CuriosityCritic:
    """E2 价值: 累积预测误差改进 − 基线 (Curiosity-Critic 路线)"""
    def __init__(self, alpha: float = 0.05):
        self.baseline = 0.0
        self.alpha = alpha
        self.history = deque(maxlen=200)

    def update(self, e1: float):
        self.baseline = (1 - self.alpha) * self.baseline + self.alpha * e1
        self.history.append(e1)

    def value(self, e1_new: float) -> float:
        """E2 = 该转移的可学习性 = 当前误差 − 渐近基线"""
        return max(0.0, e1_new - self.baseline)


class Memory:
    """记忆层: surprise 门控写入 + 最近检索 + 原型
    门控阈值自适应: 取 E1 历史 85% 分位数 (自组织超参数, 非手工设定)"""
    def __init__(self, cap: int = 200, quantile: float = 0.85):
        self.cap = cap
        self.quantile = quantile
        self.items = []          # (s, a, e1, e2, t)
        self.prototypes = []     # 原型中心
        self.e1_history = deque(maxlen=500)

    def update_threshold(self, e1: float):
        self.e1_history.append(e1)

    def write(self, s, a, e1, e2, t):
        self.update_threshold(e1)
        if len(self.e1_history) < 50:
            return
        thresh = float(np.percentile(self.e1_history, 85))
        if e1 > thresh and len(self.items) < self.cap:
            self.items.append((s.copy(), a, e1, e2, t))
            # 原型维护: 与已有原型距离>0.7 则新增
            if not self.prototypes:
                self.prototypes.append(s.copy())
            else:
                d = min(np.linalg.norm(s - p) for p in self.prototypes)
                if d > 0.7 and len(self.prototypes) < 20:
                    self.prototypes.append(s.copy())

    def sample_goal(self, value_weighted: bool = True) -> np.ndarray:
        """目标生成: 从记忆采样 (value_weighted=True → 按 E2 价值加权)
        记忆空则随机. 价值权重 = 记忆条目的 E2 (可学习性+锚)"""
        if not self.items:
            return make_rng().randn(8) * 0.5
        if value_weighted:
            weights = np.array([i[3] + 0.1 for i in self.items])
        else:
            weights = np.ones(len(self.items))
        weights = weights / weights.sum()
        idx = make_rng().choice(len(self.items), p=weights)
        return self.items[idx][0].copy()

    def familiarity(self, s) -> float:
        if not self.prototypes:
            return 0.0
        d = min(np.linalg.norm(s - p) for p in self.prototypes)
        return 1.0 / (1.0 + d)


# ─── 主模拟 ────────────────────────────────────────────────
def run(n_steps: int = 4000, mode: str = "two_level", seed: int = 42,
        static: bool = False, anchor_w: float = 0.0, hunger_rate: float = 0.001):
    """mode: 'two_level' = E1+E2; 'pure_e1' = 仅 E1 (对照)
    anchor_w: 价值锚权重 (>0 启用 E1 之外的价值锚 + 记忆价值加权目标)
    static: True → 静态世界 (验证纯能量死寂)
    每次 run 用独立 RNG 流, 保证对照公平"""
    rng = make_rng(seed)
    enc = Encoder(d_in=11, d_out=8, seed=seed)
    pred = Predictor(seed=seed)
    critic = CuriosityCritic()
    mem = Memory()

    # 统计
    e1_hist, e2_hist, e1_mean = [], [], []
    action_hist = np.zeros(N_ACTIONS)
    goal_switch = 0
    coverage = set()   # 离散化覆盖
    agent_pos = np.array([3, 3], dtype=np.float32)

    cur_goal = None
    goal_hold = 0
    dist_prev = None   # 上一距离 (坚持性预算: 距离是否在下降)
    anchor_hist = []
    hunger = 0.0        # 内稳态需求: 0=满足, 1=饥饿 (随时间增长, 进食下降)

    for t in range(n_steps):
        # 世界状态 + agent 位置 → 观测
        world = env_state(t, static=static)
        obs = np.concatenate([agent_pos, world]).astype(np.float32)
        s = enc.encode(obs)

        # 内稳态需求 (饥饿): 随时间增长, 靠近物品"进食"下降
        hunger = min(1.0, hunger + hunger_rate)
        items = item_positions(t, static=static)
        anc_now = value_anchor(agent_pos, items)
        # 进食: 在物品旁饥饿快速下降
        if anc_now > 0.7:
            hunger = max(0.0, hunger - 0.05)
        # 有效锚 = 饥饿 × 位置价值 (饥饿时才"想要")
        anc = anc_now * hunger
        anchor_hist.append(anc)

        # 离散覆盖 (agent 所在格子)
        coverage.add((int(agent_pos[0]), int(agent_pos[1])))

        # 目标: 坚持性预算 = 60 步 + 距离不再下降才切换
        if cur_goal is not None:
            dist_cur = float(np.linalg.norm(s - cur_goal))
            improving = (dist_prev is None or dist_cur < dist_prev * 0.98)
            dist_prev = dist_cur
        else:
            improving = False
        if cur_goal is None or (goal_hold >= 60 and not improving):
            cur_goal = mem.sample_goal(value_weighted=(anchor_w > 0))
            goal_switch += 1
            goal_hold = 0
            dist_prev = None

        # 规划: 简化 CEM — 对每个动作前向 1 步
        scores = []
        for a in range(N_ACTIONS):
            d_pred = pred.predict_delta(s, a)
            s_next = s + d_pred
            e1_est = float(np.mean(d_pred ** 2)) + 0.05
            e2_est = critic.value(e1_est) if mode == "two_level" else 0.0
            dist_goal = float(np.linalg.norm(s_next - cur_goal))
            # 前瞻锚: 评估执行该动作后 agent 的下一位置价值 (期望价值 × 饥饿)
            dx_a, dy_a = ACTION_DELTA[a]
            next_pos = np.clip(agent_pos + np.array([dx_a, dy_a], dtype=np.float32),
                               0, WORLD_N - 1)
            anc_next = value_anchor(next_pos, items) * hunger if anchor_w > 0 else 0.0
            # 评分: E2(可学习性) + 前瞻锚(价值) + 目标趋近 - 已熟悉惩罚(anti-loop)

            fam = mem.familiarity(s_next)
            score = (0.6 * e2_est + anchor_w * anc_next
                     - 0.3 * dist_goal - 0.3 * fam + rng.rand() * 0.05)
            scores.append(score)

        a_sel = int(np.argmax(scores))
        action_hist[a_sel] += 1

        # 执行动作
        dx, dy = ACTION_DELTA[a_sel]
        agent_pos = np.clip(agent_pos + np.array([dx, dy], dtype=np.float32),
                            0, WORLD_N - 1)

        # 观测真实下一状态
        world_next = env_state(t + 1, static=static)
        obs_next = np.concatenate([agent_pos, world_next]).astype(np.float32)
        s_next_real = enc.encode(obs_next)
        target_delta = s_next_real - s

        # E1: 预测误差 (真实)
        d_pred_real = pred.predict_delta(s, a_sel)
        e1 = float(np.mean((d_pred_real - target_delta) ** 2))

        # E2: 可学习性 + 内稳态锚 (饥饿×价值)
        critic.update(e1)
        e2 = critic.value(e1) + anchor_w * anc

        # 在线学习 (AdaJEPA: 1 步梯度)
        pe = pred.step(s, a_sel, target_delta)
        # 编码器轻微更新 (对齐目标空间, 低 lr)
        enc.step(obs, np.outer(obs, target_delta) * 0.01 * 0.1, lr_scale=0.1)

        # 记忆写入 (surprise 门控)
        mem.write(s, a_sel, e1, e2, t)

        # 统计
        e1_hist.append(e1); e2_hist.append(e2)
        if t % 200 == 0:
            e1_mean.append(np.mean(e1_hist[-200:]))
        goal_hold += 1

    return {
        "e1_hist": e1_hist, "e2_hist": e2_hist, "e1_mean": e1_mean,
        "action_hist": action_hist, "goal_switch": goal_switch,
        "coverage": len(coverage), "mem_size": len(mem.items),
        "n_prototypes": len(mem.prototypes),
        "anchor_mean": float(np.mean(anchor_hist)),
        "wait_ratio": float(action_hist[4] / n_steps),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("JPI-1 实践内核模拟 — 2×2 对照实验")
    print("  动态世界: two_level vs pure_e1")
    print("  静态世界: two_level vs pure_e1 (验证纯能量死寂)")
    print("=" * 60)

    # 3×2 对照: (锚关闭/锚开启) × (动态/静态) — 验证锚能否防死寂
    configs = [
        # (static, mode, anchor_w, label)
        (False, "two_level", 0.0, "动态·无锚(两级)"),
        (False, "two_level", 0.6, "动态·有锚"),
        (True,  "two_level", 0.0, "静态·无锚(两级)"),
        (True,  "two_level", 0.6, "静态·有锚"),
        (True,  "pure_e1",   0.0, "静态·纯E1对照"),
        (True,  "pure_e1",   0.6, "静态·纯E1+锚"),
    ]
    for static, mode, aw, label in configs:
        r = run(4000, mode=mode, seed=7, static=static, anchor_w=aw)
        e1_init = np.mean(r['e1_hist'][:500]); e1_fin = np.mean(r['e1_hist'][-500:])
        e2_init = np.mean(r['e2_hist'][:500]); e2_fin = np.mean(r['e2_hist'][-500:])
        print(f"\n--- {label} ---")
        print(f"  E1: {e1_init:.5f} → {e1_fin:.5f} (下降 {(1-e1_fin/max(e1_init,1e-9))*100:.1f}%)")
        print(f"  E2: {e2_init:.5f} → {e2_fin:.5f} | 锚均值: {r['anchor_mean']:.3f}")
        print(f"  WAIT占比: {r['wait_ratio']*100:.1f}%")
        acts = dict(zip(['上','下','左','右','WAIT'], [int(x) for x in r['action_hist']]))
        print(f"  行动: {acts}")
        print(f"  目标切换: {r['goal_switch']} | 覆盖: {r['coverage']}/36 | "
              f"记忆: {r['mem_size']} | 原型: {r['n_prototypes']}")
