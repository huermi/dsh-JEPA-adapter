"""
JPI-6 符号层涌现检验 — 仅行为指纹 (无结构注入) 能否复现增益
============================================================
检验 LeCun 立场: 符号是否可以从"行为观察"涌现, 而不需要"结构注入"?

三模式对照:
  A. 扁平规则表 (30d, 统计特征, jpi3 基线)
  B. 行为指纹 (6d, 短模拟观测统计, 无结构注入 — LeCun 立场候选)
  C. 完整符号 (13d, 静态结构 + 行为指纹, jpi4 增益来源)

验证维度:
  1. 单阶段引导: 行为指纹 vs 完整符号 vs 扁平 (3 seed)
  2. 持续学习: 共享 vs 独立 + 遗忘追踪 (jpi5 框架)
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jpi4_symbolic_bb import (random_rules, run_machine, extract_symbols,
                              SymbolPredictor, N_CANDIDATES, BUDGET, STAGES)


def build_pool(n, seed, mode, n_candidates=N_CANDIDATES):
    rng = np.random.RandomState(seed)
    rules_list = [random_rules(n, rng) for _ in range(n_candidates)]
    steps_list = [run_machine(r, n) for r in rules_list]
    steps_log = np.array([np.log1p(max(s, 0)) for s in steps_list], dtype=np.float32)
    halted = np.array([s >= 0 for s in steps_list])

    if mode == "flat":
        feats = []
        for rules in rules_list:
            f = rules.flatten().astype(np.float32)
            pad = np.zeros(30, dtype=np.float32)
            pad[:len(f)] = f
            feats.append(pad)
    else:
        mode_arg = "behavior" if mode == "behavior" else "full"
        feats = [extract_symbols(r, n, mode=mode_arg) for r in rules_list]
    F = np.array(feats, dtype=np.float32)
    for j in range(F.shape[1]):
        lo, hi = F[:, j].min(), F[:, j].max()
        if hi > lo:
            F[:, j] = (F[:, j] - lo) / (hi - lo)
        else:
            F[:, j] = 0.0
    return F, steps_log, halted


def guided_efficiency(predictor, feats, steps_log, budget=BUDGET, seed=0):
    rng = np.random.RandomState(seed)
    n_cand = len(steps_log)
    halted = steps_log > 0
    true_best = float(np.max(steps_log[halted])) if halted.any() else 1e-9
    warm = rng.choice(n_cand, budget // 4, replace=False)
    for i in warm:
        predictor.step(feats[i], steps_log[i])
    scores = [(predictor.predict(feats[i]), i) for i in range(n_cand) if i not in warm]
    scores.sort(key=lambda x: -x[0])
    best = 0.0
    for _, idx in scores[: (budget - budget // 4)]:
        best = max(best, steps_log[idx])
        predictor.step(feats[idx], steps_log[idx])
    return best / true_best * 100


# ─── 模式对应的预测器维度 ───────────────────────────────────
DIMS = {"flat": 30, "behavior": 6, "full": 13}


def single_stage(mode):
    """单阶段引导 (3 seed 平均)"""
    s_dim = DIMS[mode]
    g_list, r_list = [], []
    for seed in [1, 7, 42]:
        pred = SymbolPredictor(s_dim=s_dim, seed=seed)
        gs, rs = [], []
        for n in STAGES:
            F, slog, _ = build_pool(n, 100 + n * 13, mode)
            gs.append(guided_efficiency(pred, F, slog, seed=1))
            # 随机基线: 用随机预测排序
            rng = np.random.RandomState(seed + n)
            idx = rng.permutation(len(slog))[:BUDGET]
            best_r = float(np.max(slog[idx])) if (slog[idx] > 0).any() else 0
            true_best = float(np.max(slog[slog > 0])) if (slog > 0).any() else 1e-9
            rs.append(best_r / max(true_best, 1e-9) * 100)
        g_list.append(gs); r_list.append(rs)
    return np.mean(g_list, axis=0), np.mean(r_list, axis=0)


def continual(mode):
    """持续学习: 共享 vs 独立 + 遗忘追踪"""
    s_dim = DIMS[mode]
    pools = {n: build_pool(n, 100 + n * 13, mode) for n in STAGES}

    shared = SymbolPredictor(s_dim=s_dim, seed=7)
    history = {}
    for stage_i, n in enumerate(STAGES):
        F_n, slog_n, _ = pools[n]
        guided_efficiency(shared, F_n, slog_n, seed=1)
        eff = {}
        for m in STAGES[: stage_i + 1]:
            F_m, slog_m, _ = pools[m]
            probe = SymbolPredictor(s_dim=s_dim, seed=7)
            probe.W1 = shared.W1.copy()
            probe.W2 = shared.W2.copy()
            eff[m] = guided_efficiency(probe, F_m, slog_m, seed=2)
        history[n] = eff

    indep = {}
    for n in STAGES:
        fresh = SymbolPredictor(s_dim=s_dim, seed=7)
        F_n, slog_n, _ = pools[n]
        indep[n] = guided_efficiency(fresh, F_n, slog_n, seed=1)

    return history, indep


if __name__ == "__main__":
    print("=" * 78)
    print("JPI-6 符号层涌现检验 — 仅行为指纹能否复现增益 (LeCun 立场检验)")
    print("  A=扁平30d  B=行为指纹6d(无结构注入)  C=完整符号13d")
    print("=" * 78)

    results = {}
    for mode, label in [("flat", "A. 扁平规则30d"),
                        ("behavior", "B. 行为指纹6d(无结构注入)"),
                        ("full", "C. 完整符号13d")]:
        g, r = single_stage(mode)
        results[mode] = (g, r)
        print(f"\n--- {label} 单阶段引导 (3 seed) ---")
        for i, n in enumerate(STAGES):
            print(f"  n={n}: 引导 {g[i]:.1f}% vs 随机 {r[i]:.1f}% → {g[i]-r[i]:+.1f}pp")
        print(f"  总体: 引导 {np.mean(g):.1f}% vs 随机 {np.mean(r):.1f}% → {np.mean(g)-np.mean(r):+.1f}pp")

    print("\n" + "=" * 78)
    print("持续学习验证 (共享 vs 独立 + 遗忘追踪)")
    print("=" * 78)
    for mode, label in [("behavior", "B. 行为指纹6d"),
                        ("full", "C. 完整符号13d")]:
        hist, indep = continual(mode)
        print(f"\n--- {label} ---")
        transfer = {n: hist[n][n] - indep[n] for n in STAGES}
        print(f"  迁移增益: " + " | ".join(f"n{n}:{transfer[n]:+.1f}pp" for n in STAGES))
        print(f"  遗忘追踪 (学完n=5后回测):")
        for m in STAGES[:-1]:
            fresh_v = hist[m][m]
            after_v = hist[STAGES[-1]][m]
            d = after_v - fresh_v
            tag = "遗忘" if d < -3 else ("保持" if abs(d) <= 3 else "正迁移")
            print(f"    n={m}: {fresh_v:.0f}% → {after_v:.0f}% ({d:+.1f}pp, {tag})")
