"""
JPI-12b v6 场景修正探测 — 节奏匹配 + 导航场决策
=================================================
诊断迭代:
  v5: 决策只看一步 → 朝资源的第一步不如 WAIT → agent 永远不动
  v6: 加导航场 — 评分 = 饱足 + FEED_BOOST*exp(-d_nav/SIGMA)
      (预测"到最近就绪资源的距离"的饱足增益, 能闻到远处资源)
      绩效信号 = 采集率 (每 tick 采集滑动平均) — 量纲清晰
"""
import numpy as np

WORLD_N = 12
N_ACTIONS = 5
ACTION_DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
RESOURCES = [(3.0, 3.0), (9.0, 3.0), (3.0, 9.0)]
COLLECT_D = 0.8
REGEN_STATIC = 500
REGEN_DRIFT = 80
FEED_BOOST = 0.5
DEATH_DELAY = 300
NAV_SIGMA = 3.0        # 导航场: 能"闻到"3 格外的就绪资源
SAT_W = 0.8
MOVE_COST = 0.02
PROBE_RATES = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.015, 0.03]
N_STEPS = 6000
SEEDS = [1, 7]


def dist(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32)
                                - np.asarray(b, dtype=np.float32)))


def run_agent(n_steps=N_STEPS, seed=42, fixed_rate=0.001, phase=0):
    """phase=0 静止(再生慢500), phase=1 漂移(再生快80)
    返回: (采集率, 移动率, 死亡次数, 采集历史)"""
    rng = np.random.RandomState(seed)
    regen = REGEN_STATIC if phase == 0 else REGEN_DRIFT
    agent_pos = np.array([6.0, 6.0], dtype=np.float32)
    hunger = 0.0
    rate = fixed_rate
    cooldown = [0, 0, 0]
    death_counter = 0
    n_collect = 0
    n_move = 0
    n_death = 0
    collect_hist = []     # 每 tick 是否采集 (绩效信号)

    for t in range(n_steps):
        hunger = min(1.0, hunger + rate)
        cooldown = [max(0, c - 1) for c in cooldown]

        # 采集
        got = False
        for i, (rx, ry) in enumerate(RESOURCES):
            if dist(agent_pos, (rx, ry)) < COLLECT_D and cooldown[i] <= 0:
                hunger = max(0.0, hunger - FEED_BOOST)
                cooldown[i] = regen
                n_collect += 1
                got = True

        # 死亡
        if hunger >= 0.99:
            death_counter += 1
            if death_counter >= DEATH_DELAY:
                n_death += 1
                death_counter = 0
                agent_pos = np.array([6.0, 6.0], dtype=np.float32)
                hunger = 0.0
        else:
            death_counter = 0

        # 决策: 导航场 (到最近就绪资源的饱足增益) + 移动成本
        scores = []
        for a in range(N_ACTIONS):
            dx_a, dy_a = ACTION_DELTA[a]
            next_pos = np.clip(agent_pos + np.array([dx_a, dy_a], dtype=np.float32),
                               0, WORLD_N - 1)
            # 最近就绪资源的导航距离 (饥饿门控: 饱足时不被吸引)
            d_nav = min(dist(next_pos, (rx, ry))
                        for i, (rx, ry) in enumerate(RESOURCES)
                        if cooldown[i] <= 0) if any(c <= 0 for c in cooldown) else 99.0
            hungry = max(0.0, hunger - 0.3) * 2.0   # 饥饿门控 [0,1.4]
            nav_gain = FEED_BOOST * float(np.exp(-d_nav / NAV_SIGMA)) * hungry
            satiety_next = 1.0 - hunger + nav_gain
            move_pen = MOVE_COST if a != N_ACTIONS - 1 else 0.0
            score = SAT_W * satiety_next - move_pen + rng.rand() * 0.02
            scores.append(score)
        a_sel = int(np.argmax(scores))
        dx, dy = ACTION_DELTA[a_sel]
        if a_sel != N_ACTIONS - 1:
            n_move += 1
        agent_pos = np.clip(agent_pos + np.array([dx, dy], dtype=np.float32),
                            0, WORLD_N - 1)

        collect_hist.append(1.0 if got else 0.0)

    return n_collect / (n_steps / 1000.0), n_move / n_steps, n_death, collect_hist


def net_perf(collect_rate, n_death, n_steps=N_STEPS, death_pen=5.0):
    """净绩效 = 采集率 - 死亡惩罚 (一次死亡 = 损失 5 次采集当量
    ≈ 死亡300tick + 重生导航 ~200tick = 5 次采集的产出)"""
    return collect_rate - death_pen * n_death / (n_steps / 1000.0)


def main():
    print("=" * 78)
    print("JPI-12b v6 场景修正探测 — 节奏匹配 + 导航场")
    print(f"  静止再生={REGEN_STATIC} / 漂移再生={REGEN_DRIFT} | 导航 σ={NAV_SIGMA}")
    print("=" * 78)

    print(f"\n{'rate':>7} | {'静止: 采集率':>10} {'净绩效':>7} {'死亡':>5} | {'漂移: 采集率':>10} {'净绩效':>7} {'死亡':>5} | 峰")
    print("-" * 88)
    peaks = {"static": (None, -1e9), "drift": (None, -1e9)}
    for rate in PROBE_RATES:
        s_c, s_d, d_c, d_d = [], [], [], []
        for seed in SEEDS:
            sc, _, sd, _ = run_agent(seed=seed, fixed_rate=rate, phase=0)
            dc, _, dd, _ = run_agent(seed=seed, fixed_rate=rate, phase=1)
            s_c.append(sc); s_d.append(sd)
            d_c.append(dc); d_d.append(dd)
        scm, dcm = float(np.mean(s_c)), float(np.mean(d_c))
        s_perf = net_perf(scm, np.mean(s_d))
        d_perf = net_perf(dcm, np.mean(d_d))
        tag = " <-- 静止峰" if s_perf > peaks["static"][1] else ""
        tag += " <-- 漂移峰" if d_perf > peaks["drift"][1] else ""
        if s_perf > peaks["static"][1]:
            peaks["static"] = (rate, s_perf)
        if d_perf > peaks["drift"][1]:
            peaks["drift"] = (rate, d_perf)
        print(f"{rate:7.4f} | {scm:10.2f} {s_perf:7.2f} {np.mean(s_d):5.1f} | "
              f"{dcm:10.2f} {d_perf:7.2f} {np.mean(d_d):5.1f} | {tag}")

    r_s, p_s = peaks["static"]
    r_d, p_d = peaks["drift"]
    print("-" * 88)
    print(f"静止期最优 rate = {r_s} (净绩效 {p_s:.2f})")
    print(f"漂移期最优 rate = {r_d} (净绩效 {p_d:.2f})")
    if r_d > r_s * 3 and p_d > p_s:
        print(f"\n✅ 场景有效: 两阶段峰值分离 (漂移 {r_d} >> 静止 {r_s})")
    else:
        print(f"\n❌ 场景无效: 峰值未分离 (静止 {r_s} vs 漂移 {r_d})")


if __name__ == "__main__":
    main()
