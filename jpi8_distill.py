"""
JPI-8 Mode 蒸馏正式化 — 系统验证蒸馏机制
==========================================
验证维度:
  1. 跨 n (2/3/4/5): 蒸馏是否在各难度都有效
  2. 蒸馏样本数 (20/60/100): 需要多少慢模型知识
  3. 多 seed 鲁棒性
  4. 蒸馏质量: 快模型预测与慢模型的一致性 (蒸馏 vs 直接)
对照: 慢模型(大预算) / 快模型-直接训练(小预算) / 快模型-蒸馏(小预算)
"""
import numpy as np
import jpi6_behavioral_symbols as j6

STAGES = [2, 3, 4, 5]
SLOW_BUDGET = 80
FAST_BUDGET = 15
DISTILL_SAMPLES = [20, 60, 100]


def run_distill(n, seed, n_distill):
    rng = np.random.RandomState(seed)
    F, slog, halted = j6.build_pool(n, 100 + n * 13, "behavior")
    true_best = float(np.max(slog[slog > 0]))

    # Mode-2: 慢模型 (大预算)
    slow = j6.SymbolPredictor(s_dim=6, seed=seed)
    j6.guided_efficiency(slow, F, slog, budget=SLOW_BUDGET, seed=1)
    eff_slow = j6.guided_efficiency(slow, F, slog, budget=SLOW_BUDGET, seed=2)

    # Mode-1 直接: 小预算训练
    fast_direct = j6.SymbolPredictor(s_dim=6, seed=seed)
    j6.guided_efficiency(fast_direct, F, slog, budget=FAST_BUDGET, seed=1)
    eff_direct = j6.guided_efficiency(fast_direct, F, slog, budget=FAST_BUDGET, seed=2)

    # Mode-1 蒸馏: 从慢模型软标签学习
    fast_dist = j6.SymbolPredictor(s_dim=6, seed=seed)
    idx = rng.choice(len(F), n_distill, replace=False)
    for i in idx:
        soft = slow.predict(F[i])
        fast_dist.step(F[i], soft)
    eff_dist = j6.guided_efficiency(fast_dist, F, slog, budget=FAST_BUDGET, seed=2)

    # 蒸馏质量: 快模型与慢模型预测相关性
    corr = np.corrcoef([slow.predict(f) for f in F[:200]],
                       [fast_dist.predict(f) for f in F[:200]])[0, 1]
    return eff_slow, eff_direct, eff_dist, corr


def main():
    print("=" * 78)
    print("JPI-8 Mode 蒸馏正式化 — 跨 n × 蒸馏样本数 × 3 seed")
    print("=" * 78)

    print(f"\n{'n':>3} | {'样本':>4} | {'慢(80)':>8} {'直接(15)':>9} {'蒸馏(15)':>9} | "
          f"{'蒸馏增益':>8} | {'相关性':>7}")
    print("-" * 78)
    for n in STAGES:
        for ns in DISTILL_SAMPLES:
            effs_s, effs_d, effs_dst, corrs = [], [], [], []
            for seed in [1, 7, 42]:
                s, d, dst, c = run_distill(n, seed, ns)
                effs_s.append(s); effs_d.append(d); effs_dst.append(dst); corrs.append(c)
            gain = np.mean(effs_dst) - np.mean(effs_d)
            print(f"{n:>3} | {ns:>4} | {np.mean(effs_s):7.1f}% {np.mean(effs_d):8.1f}% "
                  f"{np.mean(effs_dst):8.1f}% | {gain:+7.1f}pp | {np.mean(corrs):.3f}")
        print("-" * 78)

    # 汇总: 蒸馏最佳样本数
    print("\n### 汇总 (各 n 平均) ###")
    for ns in DISTILL_SAMPLES:
        gains, corrs = [], []
        for n in STAGES:
            for seed in [1, 7, 42]:
                s, d, dst, c = run_distill(n, seed, ns)
                gains.append(dst - d); corrs.append(c)
        print(f"  样本 {ns}: 平均蒸馏增益 {np.mean(gains):+.1f}pp | 相关性 {np.mean(corrs):.3f}")

    print("\n### 结论 ###")
    print("  蒸馏是否稳定有效? 最佳样本数? 增益随 n 如何变化?")


if __name__ == "__main__":
    main()
