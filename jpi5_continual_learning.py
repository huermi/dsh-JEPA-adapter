"""
JPI-5 持续学习验证 — 结构先验符号层的跨 n 迁移与遗忘
====================================================
基于 jpi4 的符号结构特征, 重点验证持续学习的两个本质问题:
  1. 正迁移: n=2/3/4 学到的结构知识是否帮助 n=5 (共享 vs 独立对照)
  2. 灾难性遗忘: 学完 n=5 后, 回测 n=2/3/4 的能力是否退化

实验设计:
  A. 持续学习 (共享模型): 一个 predictor 依次学 n=2→3→4→5
     - 每阶段末: 回测"当前模型"在已学所有 n 上的引导效率 (遗忘追踪)
  B. 独立训练 (对照): 每阶段新建 predictor, 无迁移
     - 测单阶段上限

指标:
  - 每阶段引导效率 (已有)
  - 遗忘曲线: 学 n=3 后回测 n=2 的效率 vs 学 n=2 后立即回测的效率
  - 迁移增益: 共享模型在 n=5 的效率 - 独立模型在 n=5 的效率
"""
import numpy as np
from jpi4_symbolic_bb import (random_rules, run_machine, extract_symbols,
                              SymbolPredictor, N_CANDIDATES, BUDGET, STAGES)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_pool(n, seed, n_candidates=N_CANDIDATES):
    """构建固定测试池 (每个 n 固定 seed, 用于跨阶段回测)"""
    rng = np.random.RandomState(seed)
    rules_list = [random_rules(n, rng) for _ in range(n_candidates)]
    steps_list = [run_machine(r, n) for r in rules_list]
    steps_log = np.array([np.log1p(max(s, 0)) for s in steps_list], dtype=np.float32)
    halted = np.array([s >= 0 for s in steps_list])
    # 符号特征
    feats = np.array([extract_symbols(r, n) for r in rules_list], dtype=np.float32)
    # 归一化 (每维到 [0,1])
    for j in range(feats.shape[1]):
        lo, hi = feats[:, j].min(), feats[:, j].max()
        if hi > lo:
            feats[:, j] = (feats[:, j] - lo) / (hi - lo)
        else:
            feats[:, j] = 0.0
    return rules_list, feats, steps_log, halted


def guided_efficiency(predictor, feats, steps_log, budget=BUDGET, seed=0):
    """评估: 用预测器引导, 找池中停机最长机器的效率 (预算=budget 次模拟)"""
    rng = np.random.RandomState(seed)
    n_cand = len(steps_log)
    halted = steps_log > 0
    true_best = float(np.max(steps_log[halted])) if halted.any() else 1e-9

    # 预热 (随机学 budget/4)
    warm = rng.choice(n_cand, budget // 4, replace=False)
    for i in warm:
        predictor.step(feats[i], steps_log[i])

    # 引导
    scores = [(predictor.predict(feats[i]), i) for i in range(n_cand) if i not in warm]
    scores.sort(key=lambda x: -x[0])
    best = 0.0
    for _, idx in scores[: (budget - budget // 4)]:
        best = max(best, steps_log[idx])
        predictor.step(feats[idx], steps_log[idx])  # 在线学习
    return best / true_best * 100


def main():
    print("=" * 78)
    print("JPI-5 持续学习验证 — 符号结构特征: 跨 n 迁移 vs 灾难性遗忘")
    print("=" * 78)

    # 固定测试池 (每个 n 独立 seed 构建, 保证公平)
    pools = {n: build_pool(n, seed=100 + n * 13) for n in STAGES}

    # ── A. 持续学习 (共享模型) ──
    print("\n### A. 持续学习 (共享模型, 依次学 n=2→5) ###")
    shared = SymbolPredictor(seed=7)
    history = {}          # n → {学完时各已学 n 的效率}
    for stage_i, n in enumerate(STAGES):
        # 学习当前 n (用它自己的池预热+引导, 相当于在环境中探索)
        _, feats_n, steps_log_n, _ = pools[n]
        guided_efficiency(shared, feats_n, steps_log_n, seed=1)
        # 回测所有已学 n (含当前)
        eff = {}
        for m in STAGES[: stage_i + 1]:
            _, f_m, s_m, _ = pools[m]
            # 用 fresh 副本评估 (避免回测污染模型)
            probe = SymbolPredictor(seed=7)
            probe.W1 = shared.W1.copy()
            probe.W2 = shared.W2.copy()
            eff[m] = guided_efficiency(probe, f_m, s_m, seed=2)
        history[n] = eff
        print(f"  学完 n={n} 后回测: " +
              " | ".join(f"n{m}:{eff[m]:.0f}%" for m in STAGES[: stage_i + 1]))

    # ── B. 独立训练 (对照) ──
    print("\n### B. 独立训练 (每阶段新模型, 无迁移) ###")
    indep = {}
    for n in STAGES:
        fresh = SymbolPredictor(seed=7)
        _, f_n, s_n, _ = pools[n]
        indep[n] = guided_efficiency(fresh, f_n, s_n, seed=1)
        print(f"  n={n} 独立训练: {indep[n]:.0f}%")

    # ── 结论 ──
    print("\n### 持续学习裁决 ###")
    # 迁移增益: 共享在 n 的效率 vs 独立在 n
    for n in STAGES:
        shared_n = history[n][n]
        gain = shared_n - indep[n]
        print(f"  n={n}: 共享 {shared_n:.0f}% vs 独立 {indep[n]:.0f}% → 迁移 {gain:+.1f}pp")
    # 遗忘: 学完 n=5 后, n=2/3/4 的效率 vs 学完各自时
    print("\n  遗忘追踪 (学完 n=5 后回测 vs 刚学完时):")
    for m in STAGES[:-1]:
        fresh_val = history[m][m]
        after_val = history[STAGES[-1]][m]
        delta = after_val - fresh_val
        print(f"    n={m}: 刚学完 {fresh_val:.0f}% → 学完n=5后 {after_val:.0f}% → {delta:+.1f}pp "
              f"({'遗忘' if delta < -3 else '保持' if abs(delta) <= 3 else '正迁移'})")


if __name__ == "__main__":
    main()
