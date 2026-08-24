"""
JPI-2 自主持续学习验证 — 分布偏移下的自适应
============================================
验证命题: 结合成熟技术 (JEPA 预测 + 两级能量 + 可训练 Critic + 内稳态锚),
一个系统能否在无外部目标/无外部奖励的条件下自主持续学习。

关键设计:
  1. 可训练 Critic (LeCun 蓝图): 从记忆 (状态, 实际锚值) 对学习, 预测未来成本
  2. 阶段变化环境: 物品运动模式在 t=2000 突然改变 (分布偏移)
  3. 验证指标:
     - 分布偏移后 E1 是否尖峰 (惊讶检测) → 是否恢复下降 (重新学习)
     - Critic 预测误差是否下降 (学会了预测成本)
     - 记忆是否持续更新 (surprise 门控)
     - 行为是否持续 (WAIT 比例, 覆盖扩展)
     - 对照: 有 Critic vs 无 Critic (解析式锚)
"""
import numpy as np
from collections import deque

N_ITEMS = 3
N_ACTIONS = 5
ACTION_DELTA = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]
WORLD_N = 6


def env_phase(t: int) -> int:
    """环境阶段: 0=静态 (4000tick, 充分死寂期) → 1=分布偏移 (物品跳变, 唤醒压力)"""
    return 1 if t >= 4000 else 0


def env_state(t: int):
    """阶段化世界: t<2000 完全静态 (E1→0, 死寂压力); t>=2000 物品跳变+快速移动 (分布偏移)"""
    phase = env_phase(t)
    item_pos = []
    for i in range(N_ITEMS):
        if phase == 0:
            # 配置A: 物品静止 (Δs 仅含 agent 移动项)
            cx = 1.0 + i * 1.5
            cy = 2.0 + i * 0.8
            v = 0.5 + i * 0.2
        else:
            # 配置B: 物品开始规则漂移 (可学会的正弦运动 → Δs 新增动力学项)
            cx = 1.0 + i * 1.5 + 0.8 * np.sin(t * 0.02 + i * 2.1)
            cy = 2.0 + i * 0.8 + 0.8 * np.cos(t * 0.025 + i * 1.7)
            v = 0.5 + i * 0.2 + 0.3 * np.sin(t * 0.03 + i)
        item_pos += [cx, cy, v]
    return np.array(item_pos, dtype=np.float32)


def item_positions(t: int) -> list:
    phase = env_phase(t)
    if phase == 0:
        return [(1.0 + i * 1.5, 2.0 + i * 0.8) for i in range(N_ITEMS)]
    return [(1.0 + i * 1.5 + 0.8 * np.sin(t * 0.02 + i * 2.1),
             2.0 + i * 0.8 + 0.8 * np.cos(t * 0.025 + i * 1.7)) for i in range(N_ITEMS)]


def value_anchor(agent_pos: np.ndarray, items: list, sigma: float = 1.2) -> float:
    a = np.asarray(agent_pos, dtype=np.float32)
    d_min = min(np.linalg.norm(a - np.array([x, y], dtype=np.float32)) for x, y in items)
    return float(np.exp(-d_min / sigma))


class Encoder:
    def __init__(self, d_in: int = 11, d_out: int = 8, lr: float = 0.001, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(d_in, d_out).astype(np.float32) * 0.1
        self.b = np.zeros(d_out, dtype=np.float32)
        self.lr = lr

    def encode(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(np.dot(x, self.W) + self.b)
        return h / (np.linalg.norm(h) + 1e-6)

    def step(self, x: np.ndarray, grad: np.ndarray):
        g = np.clip(grad, -0.5, 0.5)
        self.W -= self.lr * g
        self.b -= self.lr * np.sum(grad, axis=0) if grad.ndim > 1 else self.lr * grad


class Predictor:
    def __init__(self, s_dim: int = 8, h_dim: int = 16,
                 lr_pred: float = 0.05, lr_enc: float = 0.001, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(s_dim + N_ACTIONS, h_dim).astype(np.float32) * 0.2
        self.W2 = np.zeros((h_dim, s_dim), dtype=np.float32)
        self.lr_pred, self.lr_enc = lr_pred, lr_enc

    def predict_delta(self, s: np.ndarray, a: int) -> np.ndarray:
        x = np.concatenate([s, np.eye(N_ACTIONS)[a]])
        h = np.maximum(0, np.dot(x, self.W1))
        d = np.dot(h, self.W2)
        n = np.linalg.norm(d)
        return d if n < 5 else d * 5.0 / n

    def step(self, s: np.ndarray, a: int, target_delta: np.ndarray):
        x = np.concatenate([s, np.eye(N_ACTIONS)[a]])
        h = np.maximum(0, np.dot(x, self.W1))
        pred = np.dot(h, self.W2)
        err = pred - target_delta
        self.W2 -= self.lr_pred * np.clip(np.outer(h, err), -0.5, 0.5)
        dH = np.dot(err, self.W2.T) * (h > 0)
        self.W1 -= self.lr_enc * np.clip(np.outer(x, dH), -0.5, 0.5)
        return float(np.mean(err ** 2))


class Critic:
    """可训练 Critic (LeCun 蓝图): 从记忆 (状态, 实际锚值) 对学习, 预测未来成本
    成熟技术对应: critic 是"预测未来 intrinsic cost 的可训练模块"
    我们的升级: 前瞻锚从解析式 value_anchor 变为学习到的价值函数"""
    def __init__(self, s_dim: int = 8, lr: float = 0.02, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(s_dim, 1).astype(np.float32) * 0.1
        self.b = np.zeros(1, dtype=np.float32)
        self.lr = lr
        self.err_hist = deque(maxlen=200)

    def predict(self, s: np.ndarray) -> float:
        """预测状态 s 的未来价值 (成本)"""
        return float(np.dot(s, self.W)[0] + self.b[0])

    def train(self, s: np.ndarray, target_anchor: float):
        """在线学习: 实际锚值作监督 (intrinsic cost 是硬编码真值)"""
        pred = np.dot(s, self.W)[0] + self.b[0]
        err = pred - target_anchor
        self.W -= self.lr * np.clip(s * err, -0.5, 0.5).reshape(-1, 1)
        self.b -= self.lr * np.clip(err, -0.5, 0.5)
        self.err_hist.append(abs(err))
        return abs(err)


class Memory:
    """记忆层: surprise 门控写入 + 自适应原型聚类 (SIGReg 式, 阈值从距离分布涌现)
    与 surprise 门控同理: 原型分裂阈值不用手工 0.7, 而取"候选状态与最近原型的距离"
    分布的高分位 — 距离分布宽则多分裂, 窄则少分裂"""
    def __init__(self, cap: int = 300, quantile: float = 0.85,
                 proto_quantile: float = 0.9):
        self.cap, self.quantile = cap, quantile
        self.proto_quantile = proto_quantile
        self.items = []
        self.prototypes = []
        self.e1_history = deque(maxlen=500)
        self.dist_history = deque(maxlen=1000)  # 候选状态→最近原型距离分布

    def write(self, s, a, e1, e2, t):
        self.e1_history.append(e1)
        if len(self.e1_history) < 50:
            return
        thresh = float(np.percentile(self.e1_history, 85))
        if e1 > thresh and len(self.items) < self.cap:
            self.items.append((s.copy(), a, e1, e2, t))
            # 自适应原型: 记录到最近原型距离 → 距离分布 → 动态阈值
            if not self.prototypes:
                self.prototypes.append(s.copy())
            else:
                d = float(min(np.linalg.norm(s - p) for p in self.prototypes))
                self.dist_history.append(d)
                if len(self.dist_history) >= 30:
                    d_thresh = float(np.percentile(self.dist_history,
                                                   self.proto_quantile * 100))
                    if d > d_thresh and len(self.prototypes) < 20:
                        self.prototypes.append(s.copy())

    def sample_goal(self, rng) -> np.ndarray:
        if not self.items:
            return rng.randn(8) * 0.5
        weights = np.array([i[3] + 0.1 for i in self.items])
        weights = weights / weights.sum()
        idx = rng.choice(len(self.items), p=weights)
        return self.items[idx][0].copy()

    def familiarity(self, s) -> float:
        if not self.prototypes:
            return 0.0
        d = min(np.linalg.norm(s - p) for p in self.prototypes)
        return 1.0 / (1.0 + d)


def run(n_steps: int = 6000, use_critic: bool = True, seed: int = 42,
        anchor_w: float = 0.6):
    """use_critic=True → 可训练 Critic 预测价值 (LeCun 蓝图)
    use_critic=False → 解析式 value_anchor (对照)
    环境: t<4000 完全静态 (充分死寂期), t>=4000 分布偏移 (唤醒测试)"""
    rng = np.random.RandomState(seed)
    enc = Encoder(seed=seed)
    pred = Predictor(seed=seed)
    mem = Memory()
    critic = Critic(seed=seed) if use_critic else None

    e1_hist, e2_hist, c_err_hist, anc_hist = [], [], [], []
    action_hist = np.zeros(N_ACTIONS)
    coverage = set()
    agent_pos = np.array([3, 3], dtype=np.float32)
    hunger = 0.0
    goal_hold = 0
    cur_goal = None
    dist_prev = None
    mem_writes = 0
    prototype_count_hist = []
    coverage_pre = None  # 偏移前覆盖 (用于计算偏移后新增)"""

    for t in range(n_steps):
        world = env_state(t)
        obs = np.concatenate([agent_pos, world]).astype(np.float32)
        s = enc.encode(obs)
        coverage.add((int(agent_pos[0]), int(agent_pos[1])))

        hunger = min(1.0, hunger + 0.001)
        items = item_positions(t)
        anc_now = value_anchor(agent_pos, items)
        if anc_now > 0.7:
            hunger = max(0.0, hunger - 0.05)
        anc = anc_now * hunger
        anc_hist.append(anc)

        # 目标切换 (坚持性预算)
        if cur_goal is not None:
            dist_cur = float(np.linalg.norm(s - cur_goal))
            improving = (dist_prev is None or dist_cur < dist_prev * 0.98)
            dist_prev = dist_cur
        else:
            improving = False
        if cur_goal is None or (goal_hold >= 60 and not improving):
            cur_goal = mem.sample_goal(rng)
            goal_hold = 0
            dist_prev = None

        # 规划: 前瞻价值 (Critic 预测 或 解析式)
        scores = []
        for a in range(N_ACTIONS):
            d_pred = pred.predict_delta(s, a)
            s_next = s + d_pred
            e1_est = float(np.mean(d_pred ** 2)) + 0.05
            e2_est = 0.0
            dx_a, dy_a = ACTION_DELTA[a]
            next_pos = np.clip(agent_pos + np.array([dx_a, dy_a], dtype=np.float32),
                               0, WORLD_N - 1)
            if use_critic:
                val = critic.predict(s_next) * hunger  # 学习到的价值 × 需求
            else:
                val = value_anchor(next_pos, items) * hunger
            dist_goal = float(np.linalg.norm(s_next - cur_goal))
            fam = mem.familiarity(s_next)
            score = 0.6 * e2_est + anchor_w * val - 0.3 * dist_goal - 0.3 * fam + rng.rand() * 0.05
            scores.append(score)

        a_sel = int(np.argmax(scores))
        action_hist[a_sel] += 1

        dx, dy = ACTION_DELTA[a_sel]
        agent_pos = np.clip(agent_pos + np.array([dx, dy], dtype=np.float32), 0, WORLD_N - 1)

        world_next = env_state(t + 1)
        obs_next = np.concatenate([agent_pos, world_next]).astype(np.float32)
        s_next_real = enc.encode(obs_next)
        target_delta = s_next_real - s

        d_pred_real = pred.predict_delta(s, a_sel)
        e1 = float(np.mean((d_pred_real - target_delta) ** 2))
        e1_hist.append(e1)

        # Critic 在线学习: 用实际锚值作监督 (intrinsic cost 真值)
        if use_critic:
            c_err = critic.train(s_next_real, anc_now)
            c_err_hist.append(c_err)

        e2 = anchor_w * anc
        e2_hist.append(e2)

        pred.step(s, a_sel, target_delta)
        enc.step(obs, np.outer(obs, target_delta) * 0.01 * 0.1)

        mem.write(s, a_sel, e1, e2, t)
        goal_hold += 1
        if t == 3999:
            coverage_pre = len(coverage)
        if t % 500 == 0:
            prototype_count_hist.append(len(mem.prototypes))

    # 偏移后新增覆盖 (苏醒证明: 死寂系统不会去新区域)
    new_coverage = len(coverage) - (coverage_pre if coverage_pre is not None else 0)
    # 偏移后行动多样性 (熵): 死寂系统集中在 WAIT, 熵低
    post_actions = action_hist.copy()
    post_actions = post_actions / post_actions.sum()
    post_entropy = -float(np.sum(post_actions * np.log(post_actions + 1e-9)))

    return {
        "e1_hist": e1_hist, "e2_hist": e2_hist,
        "critic_err": c_err_hist, "anchor_hist": anc_hist,
        "action_hist": action_hist, "coverage": len(coverage),
        "mem_size": len(mem.items), "n_prototypes": len(mem.prototypes),
        "wait_ratio": float(action_hist[4] / n_steps),
        "prototype_hist": prototype_count_hist,
        "mem_writes": mem_writes,
        "new_coverage": new_coverage,
        "post_entropy": post_entropy,
    }


if __name__ == "__main__":
    print("=" * 72)
    print("JPI-2 自主持续学习验证 — 静态期(4000tick充分死寂) → 分布偏移(2000tick唤醒)")
    print("  验证: 死寂系统能否被偏移唤醒? 锚/Critic 是否维持苏醒能力?")
    print("=" * 72)

    for use_c, anchor_w, label in [
        (False, 0.0, "无锚 (纯两级能量)"),
        (False, 0.6, "解析式锚"),
        (True,  0.6, "可训练 Critic (LeCun 蓝图)"),
    ]:
        r = run(6000, use_critic=use_c, seed=7, anchor_w=anchor_w)
        # 分段: 静态期尾 (3500-3999) / 偏移后 (4000-5999)
        seg = lambda lo, hi: float(np.mean(r['e1_hist'][lo:hi]))
        e1_static, e1_shift = seg(3500, 3999), seg(4000, 5999)
        crit_err = float(np.mean(r['critic_err'][-200:])) if r['critic_err'] else -1

        print(f"\n--- {label} ---")
        print(f"  E1: 静态尾 {e1_static:.5f} → 偏移后 {e1_shift:.5f} "
              f"(尖峰 {(e1_shift/max(e1_static,1e-9)-1)*100:+.1f}%)")
        print(f"  偏移后新覆盖: {r['new_coverage']} 格 (死寂=0) | 行动熵: {r['post_entropy']:.2f} (死寂≈0)")
        print(f"  WAIT占比: {r['wait_ratio']*100:.1f}% | 总覆盖: {r['coverage']}/36 | "
              f"记忆: {r['mem_size']} | 原型: {r['n_prototypes']}")
        print(f"  原型演化(每500tick): {[int(x) for x in r['prototype_hist']]}")
        if use_c:
            print(f"  Critic 预测误差(末200): {crit_err:.4f}")
