"""
JPI-11 Configurator 攻坚 — 分段漂移世界场景（异稳态调制）
==========================================================
场景: 6×6 世界, 静止期(物品不动) ↔ 漂移期(物品快速动) 交替
  - 静止期最优饥饿速率 = 低 (不用追食物, 省力)
  - 漂移期最优饥饿速率 = 高 (食物在动, 必须持续追踪)
  → 饥饿速率的最优值随阶段漂移 (可学习的漂移, 非结构长尾)

Configurator (异稳态, AI 的"发烧"):
  监测: E1 滑动平均 (世界可预测性)
  漂移检测: E1 突升 = 世界变了 (静止→漂移)
  调制: E1 高 → 上调饥饿速率; E1 低 → 下调 (设定点异稳态调节)

对照: 固定饥饿速率 vs Configurator 调制
指标: 平均锚值(食物获取) / WAIT / 覆盖率 / E1
"""
import numpy as np
from collections import deque

WORLD_N = 6
N_ITEMS = 3
N_ACTIONS = 5
ACTION_DELTA = [(-1,0),(1,0),(0,-1),(0,1),(0,0)]
PHASE_LEN = 1500        # 每阶段长度


def env_phase(t):
    """阶段: 0=静止, 1=漂移 (交替)"""
    return (t // PHASE_LEN) % 2


def env_state(t, static):
    """世界状态: [物品位置x,y,属性v ×3]"""
    item_pos = []
    for i in range(N_ITEMS):
        if static:
            cx = 1.0 + i * 1.5
            cy = 2.0 + i * 0.8
            v = 0.5 + i * 0.2
        else:
            cx = 1.0 + i * 1.5 + 1.2 * np.sin(t * 0.03 + i * 2.1)
            cy = 2.0 + i * 0.8 + 1.2 * np.cos(t * 0.035 + i * 1.7)
            v = 0.5 + i * 0.2 + 0.4 * np.sin(t * 0.05 + i)
        item_pos += [cx, cy, v]
    return np.array(item_pos, dtype=np.float32)


def item_positions(t, static):
    if static:
        return [(1.0 + i * 1.5, 2.0 + i * 0.8) for i in range(N_ITEMS)]
    return [(1.0 + i * 1.5 + 1.2 * np.sin(t * 0.03 + i * 2.1),
             2.0 + i * 0.8 + 1.2 * np.cos(t * 0.035 + i * 1.7)) for i in range(N_ITEMS)]


def value_anchor(agent_pos, items, sigma=1.2):
    a = np.asarray(agent_pos, dtype=np.float32)
    d_min = min(np.linalg.norm(a - np.array([x, y], dtype=np.float32)) for x, y in items)
    return float(np.exp(-d_min / sigma))


class Encoder:
    def __init__(self, d_in=11, d_out=8, lr=0.001, seed=42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(d_in, d_out).astype(np.float32) * 0.1
        self.b = np.zeros(d_out, dtype=np.float32)
        self.lr = lr

    def encode(self, x):
        h = np.tanh(np.dot(x, self.W) + self.b)
        return h / (np.linalg.norm(h) + 1e-6)

    def step(self, x, grad):
        g = np.clip(grad, -0.5, 0.5)
        self.W -= self.lr * g
        self.b -= self.lr * np.sum(grad, axis=0) if grad.ndim > 1 else self.lr * grad


class Predictor:
    def __init__(self, s_dim=8, h_dim=16, lr=0.05, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(s_dim + N_ACTIONS, h_dim).astype(np.float32) * 0.2
        self.W2 = np.zeros((h_dim, s_dim), dtype=np.float32)
        self.lr = lr

    def predict_delta(self, s, a):
        x = np.concatenate([s, np.eye(N_ACTIONS)[a]])
        h = np.maximum(0, np.dot(x, self.W1))
        d = np.dot(h, self.W2)
        n = np.linalg.norm(d)
        return d if n < 5 else d * 5 / n

    def step(self, s, a, target):
        x = np.concatenate([s, np.eye(N_ACTIONS)[a]])
        h = np.maximum(0, np.dot(x, self.W1))
        pred = np.dot(h, self.W2)
        err = pred - target
        self.W2 -= self.lr * np.clip(np.outer(h, err), -0.5, 0.5)
        dH = np.dot(err, self.W2.T) * (h > 0)
        self.W1 -= self.lr * np.clip(np.outer(x, dH), -0.5, 0.5)
        return float(np.mean(err**2))


class Memory:
    def __init__(self, cap=200, proto_q=0.9):
        self.cap = cap
        self.proto_q = proto_q
        self.items = []
        self.prototypes = []
        self.dist_hist = deque(maxlen=1000)

    def write(self, s, e1):
        if len(self.items) >= self.cap:
            return
        if not self.prototypes:
            self.prototypes.append(s.copy())
            return
        d = float(min(np.linalg.norm(s - p) for p in self.prototypes))
        self.dist_hist.append(d)
        if len(self.dist_hist) >= 30:
            thresh = float(np.percentile(self.dist_hist, self.proto_q * 100))
            if d > thresh and len(self.prototypes) < 20:
                self.prototypes.append(s.copy())
        self.items.append(s.copy())

    def familiarity(self, s):
        if not self.prototypes:
            return 0.0
        d = min(np.linalg.norm(s - p) for p in self.prototypes)
        return 1.0 / (1.0 + d)


class HomeoConfigurator:
    """异稳态调制器 (AI 的发烧):
    监测 E1 滑动平均 → 检测漂移 (E1 突升) → 调制饥饿速率设定点"""
    def __init__(self, base_rate=0.001, e1_window=150, check_every=50,
                 up_factor=4.0, down_factor=2.0, lo=0.0005, hi=0.008):
        self.base_rate = base_rate
        self.e1_window = e1_window
        self.check_every = check_every
        self.up_factor = up_factor      # 检测到漂移: 饥饿速率 x4 (追食物)
        self.down_factor = down_factor  # 稳定: 饥饿速率 /2 (休息)
        self.lo, self.hi = lo, hi
        self.e1_hist = deque(maxlen=e1_window)
        self.rate = base_rate
        self.drift_events = 0
        self.rate_hist = []

    def observe(self, e1, t):
        self.e1_hist.append(e1)
        if t % self.check_every != 0 or len(self.e1_hist) < self.e1_window:
            return
        # 漂移检测: 近期 E1 均值 vs 基线 (E1 突升 = 世界变了)
        recent = np.mean(list(self.e1_hist)[-50:])
        baseline = np.mean(list(self.e1_hist)[:-50])
        if baseline > 1e-9 and recent > baseline * 2.0:
            # 世界变难了 (静止→漂移): 上调饥饿速率
            self.rate = min(self.hi, self.rate * self.up_factor)
            self.drift_events += 1
        elif recent < baseline * 0.6 and self.rate > self.base_rate:
            # 世界变简单了 (漂移→静止): 下调饥饿速率
            self.rate = max(self.lo, self.rate / self.down_factor)
        self.rate_hist.append(self.rate)


def run(n_steps=6000, use_config=False, seed=42, fixed_rate=0.001):
    rng = np.random.RandomState(seed)
    enc = Encoder(seed=seed)
    pred = Predictor(seed=seed)
    mem = Memory()
    cfg = HomeoConfigurator(base_rate=fixed_rate) if use_config else None

    e1_hist, anc_hist = [], []
    action_hist = np.zeros(N_ACTIONS)
    agent_pos = np.array([3, 3], dtype=np.float32)
    hunger = 0.0
    rate = fixed_rate
    phase_hist = []

    for t in range(n_steps):
        static = (env_phase(t) == 0)
        world = env_state(t, static)
        obs = np.concatenate([agent_pos, world]).astype(np.float32)
        s = enc.encode(obs)

        # 饥饿 + 锚
        hunger = min(1.0, hunger + rate)
        items = item_positions(t, static)
        anc_now = value_anchor(agent_pos, items)
        if anc_now > 0.7:
            hunger = max(0.0, hunger - 0.05)
        anc = anc_now * hunger

        # 动作选择 (简单: 锚 + 熟悉度惩罚)
        scores = []
        for a in range(N_ACTIONS):
            d_pred = pred.predict_delta(s, a)
            s_next = s + d_pred
            dx_a, dy_a = ACTION_DELTA[a]
            next_pos = np.clip(agent_pos + np.array([dx_a, dy_a], dtype=np.float32),
                               0, WORLD_N - 1)
            anc_next = value_anchor(next_pos, items) * hunger
            fam = mem.familiarity(s_next)
            scores.append(0.6 * anc_next - 0.3 * fam + rng.rand() * 0.05)
        a_sel = int(np.argmax(scores))
        action_hist[a_sel] += 1

        dx, dy = ACTION_DELTA[a_sel]
        agent_pos = np.clip(agent_pos + np.array([dx, dy], dtype=np.float32), 0, WORLD_N - 1)

        # 预测误差
        world_next = env_state(t + 1, static)
        obs_next = np.concatenate([agent_pos, world_next]).astype(np.float32)
        s_next_real = enc.encode(obs_next)
        target_delta = s_next_real - s
        d_pred_real = pred.predict_delta(s, a_sel)
        e1 = float(np.mean((d_pred_real - target_delta) ** 2))
        e1_hist.append(e1)

        pred.step(s, a_sel, target_delta)
        enc.step(obs, np.outer(obs, target_delta) * 0.001)
        mem.write(s, e1)
        anc_hist.append(anc)

        # Configurator: 调制饥饿速率
        if cfg:
            cfg.observe(e1, t)
            rate = cfg.rate
            phase_hist.append((env_phase(t), rate))

    return {
        "e1_hist": e1_hist, "anc_hist": anc_hist,
        "action_hist": action_hist,
        "wait_ratio": float(action_hist[4] / n_steps),
        "drift_events": cfg.drift_events if cfg else 0,
        "phase_hist": phase_hist,
        "n_proto": len(mem.prototypes),
    }


def main():
    print("=" * 76)
    print("JPI-11 Configurator 攻坚 — 分段漂移世界 (静止↔漂移交替)")
    print("  静止期最优饥饿速率=低, 漂移期最优=高 → 设定点随阶段漂移")
    print("=" * 76)

    for seed in [1, 7, 42]:
        r_fixed = run(6000, use_config=False, seed=seed)
        r_cfg = run(6000, use_config=True, seed=seed)
        # 分阶段统计锚值 (食物获取率)
        def phase_anc(hist, phase):
            idx = [i for i in range(6000) if env_phase(i) == phase]
            return float(np.mean([hist[i] for i in idx]))
        print(f"\n--- seed {seed} ---")
        print(f"  固定: WAIT {r_fixed['wait_ratio']*100:.1f}% | 锚均值 "
              f"{np.mean(r_fixed['anc_hist']):.3f} | 静止 {phase_anc(r_fixed['anc_hist'],0):.3f} "
              f"漂移 {phase_anc(r_fixed['anc_hist'],1):.3f}")
        print(f"  Config: WAIT {r_cfg['wait_ratio']*100:.1f}% | 锚均值 "
              f"{np.mean(r_cfg['anc_hist']):.3f} | 静止 {phase_anc(r_cfg['anc_hist'],0):.3f} "
              f"漂移 {phase_anc(r_cfg['anc_hist'],1):.3f} | 漂移事件 {r_cfg['drift_events']}")

    # 汇总 3 seed
    print("\n### 汇总 (3 seed 平均) ###")
    f_anc, c_anc, f_wait, c_wait, events = [], [], [], [], []
    for seed in [1, 7, 42]:
        rf = run(6000, use_config=False, seed=seed)
        rc = run(6000, use_config=True, seed=seed)
        f_anc.append(np.mean(rf['anc_hist'])); c_anc.append(np.mean(rc['anc_hist']))
        f_wait.append(rf['wait_ratio']); c_wait.append(rc['wait_ratio'])
        events.append(rc['drift_events'])
    print(f"  锚值(食物获取): 固定 {np.mean(f_anc):.3f} vs Config {np.mean(c_anc):.3f} "
          f"→ {np.mean(c_anc)-np.mean(f_anc):+.3f}")
    print(f"  WAIT: 固定 {np.mean(f_wait)*100:.1f}% vs Config {np.mean(c_wait)*100:.1f}% "
          f"→ {np.mean(f_wait)*100-np.mean(c_wait)*100:+.1f}pp")
    print(f"  漂移检测事件: {int(np.mean(events))} 次 (期望 ~2-3 次阶段切换)")


if __name__ == "__main__":
    main()
