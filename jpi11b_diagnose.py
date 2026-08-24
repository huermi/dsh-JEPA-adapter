"""
JPI-11b Configurator 诊断 — 11 次漂移检测从哪来？
==================================================
诊断问题: 6000 tick 只有 3 次真实阶段切换, 为何检测到 11 次?
假设:
  A. 触发集中在阶段切换附近 (重复触发) → 缺去抖/冷却机制
  B. 触发散布全时间轴 → 信号本身与阶段无关, 需要换信号
  C. 混合 → 两者都要修

输出:
  1. E1 分段统计 (静止/漂移期各自的均值/方差)
  2. 每次触发时刻 vs 真实阶段切换时刻
  3. 饥饿速率轨迹 vs 阶段轨迹
"""
import numpy as np
from collections import deque
import jpi11_configurator_drift as m

PHASE_LEN = m.PHASE_LEN
N_STEPS = 6000


class DiagnosticConfigurator(m.HomeoConfigurator):
    """带日志的 Configurator: 记录每次触发时刻"""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.trigger_times = []
        self.e1_window_data = []  # (t, recent, baseline)

    def observe(self, e1, t):
        self.e1_hist.append(e1)
        if t % self.check_every != 0 or len(self.e1_hist) < self.e1_window:
            return
        recent = np.mean(list(self.e1_hist)[-50:])
        baseline = np.mean(list(self.e1_hist)[:-50])
        self.e1_window_data.append((t, float(recent), float(baseline)))
        if baseline > 1e-9 and recent > baseline * 2.0:
            self.rate = min(self.hi, self.rate * self.up_factor)
            self.drift_events += 1
            self.trigger_times.append(t)
        elif recent < baseline * 0.6 and self.rate > self.base_rate:
            self.rate = max(self.lo, self.rate / self.down_factor)
        self.rate_hist.append(self.rate)


def diagnose(seed=7):
    rng = np.random.RandomState(seed)
    enc = m.Encoder(seed=seed)
    pred = m.Predictor(seed=seed)
    mem = m.Memory()
    cfg = DiagnosticConfigurator()

    e1_hist, rate_hist, phase_hist = [], [], []
    agent_pos = np.array([3, 3], dtype=np.float32)
    hunger = 0.0
    rate = cfg.rate

    for t in range(N_STEPS):
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

        cfg.observe(e1, t)
        rate = cfg.rate
        rate_hist.append(rate)
        phase_hist.append(m.env_phase(t))

    # ── 诊断输出 ──
    e1_arr = np.array(e1_hist)
    print("=" * 76)
    print(f"JPI-11b 诊断 (seed={seed}, 6000 tick, 阶段长度 {PHASE_LEN})")
    print("=" * 76)

    # 1. E1 分段统计
    print("\n[1] E1 分段统计 (按真实阶段):")
    for phase in [0, 1]:
        idx = [i for i in range(N_STEPS) if m.env_phase(i) == phase]
        seg = e1_arr[idx]
        print(f"  阶段{'静止' if phase==0 else '漂移'}: 均值 {np.mean(seg):.5f} "
              f"方差 {np.var(seg):.2e} 中位 {np.median(seg):.5f}")

    # 2. 真实阶段切换 vs 触发时刻
    print("\n[2] 真实阶段切换时刻 vs Configurator 触发时刻:")
    true_switches = [PHASE_LEN, 2*PHASE_LEN, 3*PHASE_LEN]
    print(f"  真实切换: {true_switches}")
    print(f"  触发时刻 ({len(cfg.trigger_times)} 次): {cfg.trigger_times}")
    # 每个触发离最近真实切换的距离
    print("  每个触发 → 最近切换距离:")
    for tt in cfg.trigger_times:
        dist = min(abs(tt - s) for s in true_switches)
        label = "切换附近" if dist < 200 else "远离切换"
        print(f"    t={tt:5d} → 距切换 {dist:4d} tick ({label})")

    # 3. 触发时刻的 E1 窗口数据
    print("\n[3] 触发前后 E1 窗口 (recent vs baseline):")
    for t, recent, baseline in cfg.e1_window_data:
        if t in cfg.trigger_times or abs(t - t) < 50:
            pass
    for t, recent, baseline in cfg.e1_window_data[-30:]:
        marker = " <-- 触发" if t in cfg.trigger_times else ""
        print(f"    t={t:5d} recent={recent:.5f} baseline={baseline:.5f} "
              f"ratio={recent/max(baseline,1e-9):.1f}x{marker}")

    # 4. 饥饿速率 vs 阶段
    print("\n[4] 饥饿速率轨迹 (每 300 tick):")
    for t in range(0, N_STEPS, 300):
        phase = m.env_phase(t)
        print(f"    t={t:5d} 阶段={'静止' if phase==0 else '漂移'} "
              f"rate={rate_hist[t]:.5f}")

    # 5. E1 在切换点附近的行为
    print("\n[5] 切换点 ±200 tick 的 E1 均值 (是否真的突升):")
    for s in true_switches:
        before = np.mean(e1_arr[max(0,s-200):s])
        after = np.mean(e1_arr[s:min(N_STEPS,s+200)])
        print(f"    切换 t={s}: 前200tick E1={before:.5f} → 后200tick E1={after:.5f} "
              f"(变化 {(after/max(before,1e-9)-1)*100:+.0f}%)")


if __name__ == "__main__":
    diagnose(seed=7)
