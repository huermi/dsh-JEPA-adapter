"""
JPI-11c Configurator v2 — 绩效梯度爬山式异稳态调节
====================================================
机制改进 (基于 JPI-11b 诊断):
  错误: 用 E1 (认知状态) 做检测信号 — 它与阶段切换不同步
        (静止期探索式移动→E1高; 漂移期规律追踪→E1低, 关系被饥饿速率倒转)
  正确: 用锚值 (绩效) 做检测信号 — 食物获取率直接反映"当前饥饿速率
        是否匹配环境动力学"

算法 (绩效梯度爬山, 无阶段知识, 无 E1):
  每 check_every tick:
    新锚 = 最近窗口锚值均值
    绩效变化 = 新锚 - 旧锚
    if 绩效改善: 保持当前调节方向
    if 绩效下降: 反转调节方向
    rate += direction * step  (clamp [lo, hi])

参数不敏感: step/窗口不需要精确调 — 爬山自适应
"""
import numpy as np
from collections import deque
import jpi11_configurator_drift as m

N_STEPS = 6000


class PerfConfigurator:
    """绩效梯度爬山式异稳态调节器 (v2)"""
    def __init__(self, base_rate=0.001, win=200, check_every=100,
                 step=0.0005, lo=0.0003, hi=0.01):
        self.rate = base_rate
        self.win = win
        self.check_every = check_every
        self.step = step
        self.lo, self.hi = lo, hi
        self.anc_hist = deque(maxlen=win)
        self.direction = 1.0
        self.reversals = 0
        self.rate_hist = []
        self.last_anchor = None

    def observe(self, anc, t):
        self.anc_hist.append(anc)
        if t % self.check_every != 0 or len(self.anc_hist) < self.win:
            return
        # 当前窗口绩效
        cur = float(np.mean(list(self.anc_hist)[-100:]))
        prev = float(np.mean(list(self.anc_hist)[:-100])) if len(self.anc_hist) > 100 else cur
        if self.last_anchor is not None:
            delta = cur - self.last_anchor
            # 绩效下降 → 反转方向
            if delta < -1e-5:
                self.direction = -self.direction
                self.reversals += 1
        self.last_anchor = cur
        # 沿当前方向走一步
        self.rate = float(np.clip(self.rate + self.direction * self.step, self.lo, self.hi))
        self.rate_hist.append(self.rate)


def run(n_steps=N_STEPS, use_config=False, seed=42, fixed_rate=0.001):
    rng = np.random.RandomState(seed)
    enc = m.Encoder(seed=seed)
    pred = m.Predictor(seed=seed)
    mem = m.Memory()
    cfg = PerfConfigurator() if use_config else None

    e1_hist, anc_hist = [], []
    agent_pos = np.array([3, 3], dtype=np.float32)
    hunger = 0.0
    rate = fixed_rate

    for t in range(n_steps):
        static = (m.env_phase(t) == 0)
        world = m.env_state(t, static)
        obs = np.concatenate([agent_pos, world]).astype(np.float32)
        s = enc.encode(obs)

        hunger = min(1.0, hunger + rate)
        items = m.item_positions(t, static)
        anc_now = m.value_anchor(agent_pos, items)
        if anc_now > 0.7:
            hunger = max(0.0, hunger - 0.05)
        anc = anc_now * hunger

        scores = []
        for a in range(m.N_ACTIONS):
            d_pred = pred.predict_delta(s, a)
            s_next = s + d_pred
            dx_a, dy_a = m.ACTION_DELTA[a]
            next_pos = np.clip(agent_pos + np.array([dx_a, dy_a], dtype=np.float32),
                               0, m.WORLD_N - 1)
            anc_next = m.value_anchor(next_pos, items) * hunger
            fam = mem.familiarity(s_next)
            scores.append(0.6 * anc_next - 0.3 * fam + rng.rand() * 0.05)
        a_sel = int(np.argmax(scores))
        dx, dy = m.ACTION_DELTA[a_sel]
        agent_pos = np.clip(agent_pos + np.array([dx, dy], dtype=np.float32), 0, m.WORLD_N - 1)

        world_next = m.env_state(t + 1, static)
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

        if cfg:
            cfg.observe(anc, t)
            rate = cfg.rate

    return {
        "e1_hist": e1_hist, "anc_hist": anc_hist,
        "wait_ratio": float(np.sum([1 if a == 4 else 0 for a in []]) / 1) if False else 0,
        "reversals": cfg.reversals if cfg else 0,
        "rate_hist": cfg.rate_hist if cfg else [],
        "n_proto": len(mem.prototypes),
    }


def phase_anc(anc_hist, phase):
    idx = [i for i in range(N_STEPS) if m.env_phase(i) == phase]
    return float(np.mean([anc_hist[i] for i in idx]))


def main():
    print("=" * 76)
    print("JPI-11c Configurator v2 — 绩效梯度爬山 (锚值信号, 非 E1)")
    print("=" * 76)

    f_anc, c_anc = [], []
    f_pa, c_pa = [], []   # 静止期锚
    f_pd, c_pd = [], []   # 漂移期锚
    reversals_all = []
    for seed in [1, 7, 42]:
        rf = run(use_config=False, seed=seed)
        rc = run(use_config=True, seed=seed)
        f_anc.append(np.mean(rf['anc_hist'])); c_anc.append(np.mean(rc['anc_hist']))
        f_pa.append(phase_anc(rf['anc_hist'], 0)); c_pa.append(phase_anc(rc['anc_hist'], 0))
        f_pd.append(phase_anc(rf['anc_hist'], 1)); c_pd.append(phase_anc(rc['anc_hist'], 1))
        reversals_all.append(rc['reversals'])
        print(f"\n--- seed {seed} ---")
        print(f"  固定: 锚总 {np.mean(rf['anc_hist']):.4f} | 静止 {phase_anc(rf['anc_hist'],0):.4f} "
              f"漂移 {phase_anc(rf['anc_hist'],1):.4f}")
        print(f"  Config: 锚总 {np.mean(rc['anc_hist']):.4f} | 静止 {phase_anc(rc['anc_hist'],0):.4f} "
              f"漂移 {phase_anc(rc['anc_hist'],1):.4f} | 反转 {rc['reversals']} 次")

    print("\n### 汇总 (3 seed) ###")
    print(f"  锚值: 固定 {np.mean(f_anc):.4f} vs Config {np.mean(c_anc):.4f} "
          f"→ {np.mean(c_anc)/max(np.mean(f_anc),1e-9)*100-100:+.0f}%")
    print(f"  静止期锚: 固定 {np.mean(f_pa):.4f} vs Config {np.mean(c_pa):.4f}")
    print(f"  漂移期锚: 固定 {np.mean(f_pd):.4f} vs Config {np.mean(c_pd):.4f}")
    print(f"  方向反转: {int(np.mean(reversals_all))} 次 (阶段切换 ~3 次时反转 ~6 次合理)")


if __name__ == "__main__":
    main()
