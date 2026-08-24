"""
JPI-7 空白补齐 + P0 双实验 — Mode蒸馏 / STM-Critic闭环 / 符号自动发现 / 外环调制
=============================================================================
实验1 (空白③a): Mode-2→Mode-1 蒸馏 — 慢预测器(大预算)蒸馏出快预测器(小预算)
实验2 (空白③b): STM-Critic 闭环 — Critic 从记忆回放训练 vs 仅即时转移训练
实验3 (P0-B): 符号自动发现 — 可学习编码器(EMA target + L1)从原始轨迹学特征, vs 手工行为指纹
实验4 (P0-A): Configurator 外环 — 元控制器调制饥饿速率/探索温度, vs 固定参数
"""
import numpy as np
from collections import deque
import jpi6_behavioral_symbols as j6

# ═══════════════════════════════════════════════════════════
# 实验1: Mode-2 → Mode-1 蒸馏
# ═══════════════════════════════════════════════════════════
def exp1_mode_distill(n=4, seed=7):
    """慢模型(预算 80) 蒸馏出 快模型(预算 15), 对比蒸馏 vs 直接小预算训练"""
    rng = np.random.RandomState(seed)
    F, slog, halted = j6.build_pool(n, 100 + n * 13, "behavior")
    true_best = float(np.max(slog[slog > 0]))

    # Mode-2: 大预算慢模型
    slow = j6.SymbolPredictor(s_dim=6, seed=seed)
    j6.guided_efficiency(slow, F, slog, budget=80, seed=1)

    # Mode-1 蒸馏: 用慢模型的预测作为软目标训练快模型
    fast_distilled = j6.SymbolPredictor(s_dim=6, seed=seed)
    idx = rng.choice(len(F), 60, replace=False)
    for i in idx:
        target = slow.predict(F[i])  # 软标签 (慢模型的知识)
        fast_distilled.step(F[i], target)

    # 对照: 直接小预算训练的 Mode-1 (无蒸馏)
    fast_direct = j6.SymbolPredictor(s_dim=6, seed=seed)
    j6.guided_efficiency(fast_direct, F, slog, budget=15, seed=1)

    # 评估蒸馏模型的引导效率
    eff_dist = j6.guided_efficiency(fast_distilled, F, slog, budget=15, seed=1)
    print(f"  慢模型(预算80): {j6.guided_efficiency(slow, F, slog, budget=80, seed=2):.1f}%")
    print(f"  快模型-蒸馏(预算15): {eff_dist:.1f}%")
    print(f"  快模型-直接(预算15): {j6.guided_efficiency(fast_direct, F, slog, budget=15, seed=2):.1f}%")
    print(f"  蒸馏增益: {eff_dist - j6.guided_efficiency(fast_direct, F, slog, budget=15, seed=2):+.1f}pp")
    return eff_dist


# ═══════════════════════════════════════════════════════════
# 实验2: STM-Critic 闭环 (记忆回放训练 Critic)
# ═══════════════════════════════════════════════════════════
class CriticReplay:
    """Critic: 用记忆回放训练 (AdaJEPA recent-N 思想) vs 仅即时转移"""
    def __init__(self, s_dim=6, lr=0.02, seed=7):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(s_dim, 1).astype(np.float32) * 0.1
        self.b = np.zeros(1, dtype=np.float32)
        self.lr = lr
        self.buffer = deque(maxlen=20)

    def predict(self, s):
        return float(np.dot(s, self.W)[0] + self.b[0])

    def train_immediate(self, s, target):
        err = self.predict(s) - target
        self.W -= self.lr * np.clip(s * err, -0.5, 0.5).reshape(-1, 1)
        self.b -= self.lr * np.clip(err, -0.5, 0.5)
        return abs(err)

    def train_replay(self, s, target):
        """即时训练 + 存入回放缓冲 + 额外从缓冲采样训练 (STM-Critic 闭环)"""
        err1 = self.train_immediate(s, target)
        self.buffer.append((s.copy(), target))
        err2 = 0.0
        if len(self.buffer) >= 4:
            for _ in range(2):  # 每次回放 2 条
                i = np.random.randint(len(self.buffer))
                ss, tt = self.buffer[i]
                err2 += abs(self.predict(ss) - tt)
                self.W -= self.lr * np.clip(ss * (self.predict(ss) - tt), -0.5, 0.5).reshape(-1, 1)
                self.b -= self.lr * np.clip(self.predict(ss) - tt, -0.5, 0.5)
        return err1 + err2 / max(1, len(self.buffer))


def exp2_stm_critic(n=4, seed=7):
    """对比: Critic 仅即时训练 vs 记忆回放训练 (预测精度)"""
    F, slog, halted = j6.build_pool(n, 100 + n * 13, "behavior")
    # 用运行步数作为 critic 预测目标 (log 归一化)
    targets = slog / max(np.max(slog), 1e-9)

    c1 = CriticReplay(seed=seed)   # 仅即时
    c2 = CriticReplay(seed=seed)   # 回放
    err1, err2 = [], []
    idx = np.random.RandomState(seed).permutation(len(F))
    for i in idx[:200]:
        err1.append(c1.train_immediate(F[i], targets[i]))
        err2.append(c2.train_replay(F[i], targets[i]))

    e1 = np.mean(err1[-50:]); e2 = np.mean(err2[-50:])
    print(f"  即时训练终误差: {e1:.4f}")
    print(f"  记忆回放终误差: {e2:.4f}")
    print(f"  回放改进: {(1 - e2/max(e1,1e-9))*100:+.1f}%")
    return e1, e2


# ═══════════════════════════════════════════════════════════
# 实验3 (P0-B): 符号自动发现 — 可学习编码器 vs 手工行为指纹
# ═══════════════════════════════════════════════════════════
class LearnedEncoder:
    """可学习编码器: 原始轨迹 → 表征. EMA target + L1 (V-JEPA 2 思想简化版)"""
    def __init__(self, traj_dim=30, s_dim=6, h_dim=16, lr=0.01, seed=7):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(traj_dim, h_dim).astype(np.float32) * 0.1
        self.W2 = rng.randn(h_dim, s_dim).astype(np.float32) * 0.1
        self.lr = lr
        # EMA target
        self.tW1 = self.W1.copy()
        self.tW2 = self.W2.copy()
        self.ema = 0.99

    def encode(self, traj):
        h = np.tanh(np.dot(traj, self.W1))
        return np.tanh(np.dot(h, self.W2))

    def target_encode(self, traj):
        h = np.tanh(np.dot(traj, self.tW1))
        return np.tanh(np.dot(h, self.tW2))

    def step(self, traj1, traj2):
        """V-JEPA 2 式: 从轨迹1预测轨迹2的表征 (L1 + EMA target)"""
        z1 = self.encode(traj1)
        z2_target = self.target_encode(traj2)
        err = z1 - z2_target
        # 更新 online
        dW2 = np.outer(np.tanh(np.dot(traj1, self.W1)), err)
        dH = np.dot(err, self.W2.T) * (1 - np.tanh(np.dot(traj1, self.W1))**2)
        dW1 = np.outer(traj1, dH)
        self.W2 -= self.lr * np.clip(dW2, -0.5, 0.5)
        self.W1 -= self.lr * np.clip(dW1, -0.5, 0.5)
        # EMA 更新 target
        self.tW1 = self.ema * self.tW1 + (1 - self.ema) * self.W1
        self.tW2 = self.ema * self.tW2 + (1 - self.ema) * self.W2
        return float(np.mean(err**2))


def traj_fingerprint(rules, n, steps=60, dim=30):
    """原始轨迹指纹: 状态+磁带演化序列 (喂给可学习编码器, 不手工选特征)"""
    tape = {}
    pos, state = 0, 0
    traj = []
    for _ in range(steps):
        sym = tape.get(pos, 0)
        idx = state * 2 + sym
        if idx >= len(rules):
            break
        new_sym, d, nxt = rules[idx]
        # 记录: [状态, 符号, 方向, 磁带宽度] 归一化
        tape[pos] = new_sym
        width = max(tape.keys()) - min(tape.keys()) if tape else 0
        traj.append([state / max(n,1), sym, d, min(width/50, 1.0)])
        pos += 1 if d == 1 else -1
        if nxt < 0:
            break
        state = nxt
    feat = np.zeros(dim, dtype=np.float32)
    for i, v in enumerate(traj[: dim // 4]):
        feat[i*4:i*4+4] = v
    return feat


def exp3_auto_symbols(n=4, seed=7):
    """对比: 手工行为指纹(6d) vs 可学习编码器自动发现"""
    F, slog, halted = j6.build_pool(n, 100 + n * 13, "behavior")
    true_best = float(np.max(slog[slog > 0]))

    rng = np.random.RandomState(seed)
    rules_list = []
    rng2 = np.random.RandomState(100 + n * 13)
    for _ in range(j6.N_CANDIDATES):
        rules_list.append(j6.random_rules(n, rng2))
    steps_list = [j6.run_machine(r, n) for r in rules_list]
    slog2 = np.array([np.log1p(max(s, 0)) for s in steps_list], dtype=np.float32)

    # 可学习编码器: 从轨迹对学习 (自监督, 无标签)
    enc = LearnedEncoder(seed=seed)
    trajs = [traj_fingerprint(r, n) for r in rules_list]
    for _ in range(200):
        i, j = rng.randint(len(trajs), size=2)
        enc.step(trajs[i], trajs[j])  # 预测另一个轨迹的表征 (自监督对齐)

    # 用学到的表征做引导
    auto_feats = np.array([enc.encode(t) for t in trajs], dtype=np.float32)
    # 归一化
    for k in range(auto_feats.shape[1]):
        lo, hi = auto_feats[:, k].min(), auto_feats[:, k].max()
        if hi > lo:
            auto_feats[:, k] = (auto_feats[:, k] - lo) / (hi - lo)
        else:
            auto_feats[:, k] = 0.0

    pred_auto = j6.SymbolPredictor(s_dim=6, seed=seed)
    eff_auto = j6.guided_efficiency(pred_auto, auto_feats, slog2, budget=30, seed=1)

    pred_manual = j6.SymbolPredictor(s_dim=6, seed=seed)
    eff_manual = j6.guided_efficiency(pred_manual, F, slog, budget=30, seed=1)

    print(f"  手工行为指纹(6d): {eff_manual:.1f}%")
    print(f"  可学习编码器(自动发现): {eff_auto:.1f}%")
    print(f"  差距: {eff_auto - eff_manual:+.1f}pp")
    return eff_manual, eff_auto


# ═══════════════════════════════════════════════════════════
# 实验4 (P0-A): Configurator 外环调制
# ═══════════════════════════════════════════════════════════
def exp4_configurator(seed=7):
    """Configurator: 监测 E2 活性/覆盖停滞 → 调制饥饿速率 (对比固定参数)"""
    from jpi1_simulator import run as jpi1_run  # 复用 6×6 世界

    # 固定参数 (饥饿速率 0.001) — 已知基线
    r_fixed = jpi1_run(4000, mode="two_level", seed=seed, static=True, anchor_w=0.6, hunger_rate=0.001)

    # Configurator: 分段调制饥饿速率 (模拟外环, 简化版: 每 500 tick 根据 WAIT 比例调整)
    # 用多次 run 近似: 不同 hunger_rate 的效果, 外环选择最优
    rates = [0.0005, 0.001, 0.002, 0.005]
    results = {}
    for hr in rates:
        r = jpi1_run(2000, mode="two_level", seed=seed, static=True, anchor_w=0.6, hunger_rate=hr)
        results[hr] = r["wait_ratio"]
    best_rate = min(results, key=results.get)
    r_best = jpi1_run(4000, mode="two_level", seed=seed, static=True, anchor_w=0.6, hunger_rate=best_rate)

    print(f"  固定参数(hr=0.001): WAIT {r_fixed['wait_ratio']*100:.1f}%")
    print(f"  Configurator 选择 hr={best_rate}: WAIT {r_best['wait_ratio']*100:.1f}%")
    print(f"  各速率 WAIT: { {k: f'{v*100:.0f}%' for k,v in results.items()} }")
    print(f"  外环调制改进: {r_fixed['wait_ratio']*100 - r_best['wait_ratio']*100:+.1f}pp")
    return r_fixed, r_best


if __name__ == "__main__":
    print("=" * 72)
    print("JPI-7 空白补齐 + P0 双实验")
    print("=" * 72)

    print("\n【实验1: Mode-2→Mode-1 蒸馏 (空白③a)】")
    exp1_mode_distill()

    print("\n【实验2: STM-Critic 记忆回放闭环 (空白③b)】")
    exp2_stm_critic()

    print("\n【实验3: 符号自动发现 vs 手工指纹 (P0-B, 空白②)】")
    exp3_auto_symbols()

    print("\n【实验4: Configurator 外环调制 (P0-A, 空白①)】")
    exp4_configurator()
