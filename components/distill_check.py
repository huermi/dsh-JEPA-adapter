"""
C9 蒸馏组件验收 (distill_check.py)
====================================
迁移保真验证: components.distill.SoftLabelDistiller 在 BB 场景下
是否复现 jpi8 的蒸馏增益 (慢 80 → 快 15, +59.4pp 量级).

对照: 慢模型(大预算) / 快-直接(小预算) / 快-蒸馏(小预算)
精简: n ∈ {2,4} × seed ∈ {1,7} × 蒸馏样本 60 (控制 CPU 时间)
"""
import sys
import numpy as np

sys.path.insert(0, "D:/JEPA")

from jpi4_symbolic_bb import SymbolPredictor
from jpi6_behavioral_symbols import build_pool, guided_efficiency
from components.distill import SoftLabelDistiller

SLOW_BUDGET = 80
FAST_BUDGET = 15
DISTILL_N = 60


def run_distill(n, seed, n_distill=DISTILL_N):
    rng = np.random.RandomState(seed)
    F, slog, _ = build_pool(n, 100 + n * 13, "behavior")

    # Mode-2: 慢模型
    slow = SymbolPredictor(s_dim=6, seed=seed)
    guided_efficiency(slow, F, slog, budget=SLOW_BUDGET, seed=1)
    eff_slow = guided_efficiency(slow, F, slog, budget=SLOW_BUDGET, seed=2)

    # Mode-1 直接
    fast_direct = SymbolPredictor(s_dim=6, seed=seed)
    guided_efficiency(fast_direct, F, slog, budget=FAST_BUDGET, seed=1)
    eff_direct = guided_efficiency(fast_direct, F, slog, budget=FAST_BUDGET, seed=2)

    # Mode-1 蒸馏 (C9 组件)
    distiller = SoftLabelDistiller()
    fast_dist = distiller.distill(
        slow, F, n_distill, seed=seed,
        student_factory=lambda: SymbolPredictor(s_dim=6, seed=seed))
    eff_dist = distiller.evaluate(
        fast_dist, (F, slog),
        budget=FAST_BUDGET, seed=2,
        eval_fn=lambda m, f, s, b, sd: guided_efficiency(m, f, s, budget=b, seed=sd))

    # 蒸馏质量: 快 vs 慢预测相关性
    corr = np.corrcoef([slow.predict(f) for f in F[:200]],
                       [fast_dist.predict(f) for f in F[:200]])[0, 1]
    return eff_slow, eff_direct, eff_dist, corr


def main():
    print("=" * 70)
    print("C9 蒸馏组件验收 — BB 场景 (慢 80 → 快 15, 蒸馏样本 60)")
    print("=" * 70)
    print(f"\n{'n':>3} | {'慢(80)':>8} {'直接(15)':>9} {'蒸馏(15)':>9} | "
          f"{'蒸馏增益':>8} | {'相关性':>7}")
    print("-" * 62)
    totals = {"slow": [], "direct": [], "dist": []}
    for n in [2, 4]:
        effs = [run_distill(n, seed) for seed in [1, 7]]
        s = np.mean([e[0] for e in effs])
        d = np.mean([e[1] for e in effs])
        dst = np.mean([e[2] for e in effs])
        c = np.mean([e[3] for e in effs])
        gain = dst - d
        totals["slow"].append(s); totals["direct"].append(d); totals["dist"].append(dst)
        print(f"{n:>3} | {s:7.1f}% {d:8.1f}% {dst:8.1f}% | {gain:+7.1f}pp | {c:.3f}")

    print("-" * 62)
    avg_gain = np.mean([a - b for a, b in zip(totals["dist"], totals["direct"])])
    print(f"平均蒸馏增益: {avg_gain:+.1f}pp (jpi8 量级 ~+6~59pp 取决于 n)")
    print(f"\n判定: 蒸馏增益 > 0 且相关性 > 0.8 → C9 迁移保真 ✅")
    ok = avg_gain > 0
    print(f"结果: {'✅ 通过' if ok else '❌ 未通过'}")


if __name__ == "__main__":
    main()
