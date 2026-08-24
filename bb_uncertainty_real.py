"""
BB 真实结构池不确定度药方重测 (bb_uncertainty_real.py)
=======================================================
闭合纠正清单待办①: 在真实结构池 (真实计数器注入) 上重测全部方法,
给出 LeCun 药方 (不确定度驱动探索) 的最终裁决.

真实池: bb_human_check 的 build_pool_self + inject_real_counters
对照 (同池 1500 + 4 台真实计数器 log 11.51, 预算 60, 3 seed):
  A  fixed        行为指纹回归 (真实池 50.1% 参考)
  B  egreedy      行为 ε-greedy (真实池 24.8% 参考)
  C1-C3 ucb       NLL σ 头 UCB (β=0.5/1.0/2.0)
  C4 maxσ         纯不确定度
  C5 dist         特征离群度
  C6 ensemble     深度集成 UCB (5 预测器)
  R  ringcontent  环内容分析 (真实池正解 100% 参考)
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import sys
import numpy as np

sys.path.insert(0, REPO_ROOT)
from bb_human_check import build_pool_self, inject_real_counters, static_score_v2
from bb_uncertainty_check import UncertaintyPredictor, predict_mu_sig
from bb_component_check import BBHomeostasis, guided
from jpi4_symbolic_bb import SymbolPredictor

BUDGET = 60
WARM = 10


def run_uncertainty(predictor, F, slog, mode, beta=None, seed=0):
    """不确定度引导 (真实池版, 复用 bb_uncertainty_check 逻辑)"""
    rng = np.random.RandomState(seed)
    n = len(slog)
    true_best = float(np.max(slog[slog > 0]))
    best_found = 0.0
    warm_idx = rng.choice(n, WARM, replace=False)
    seen = set()
    for i in warm_idx:
        if predictor is not None:
            predictor.step(F[i], slog[i])
        best_found = max(best_found, slog[i])
        seen.add(i)
    seen_all = np.zeros(n, dtype=bool)
    seen_all[warm_idx] = True
    for _ in range(BUDGET - WARM):
        unseen = np.where(~seen_all)[0]
        if mode == "dist":
            dists = np.min([np.linalg.norm(F[unseen] - F[i], axis=1)
                            for i in warm_idx], axis=0)
            idx = unseen[int(np.argmax(dists))]
        else:
            out = [predict_mu_sig(predictor, F[i]) for i in unseen]
            mus = np.array([o[0] for o in out])
            sigs = np.array([o[1] for o in out])
            if mode == "ucb":
                score = mus + beta * sigs
            elif mode == "max_sigma":
                score = sigs
            else:
                score = mus
            idx = unseen[int(np.argmax(score))]
        seen_all[idx] = True
        best_found = max(best_found, slog[idx])
        if predictor is not None:
            predictor.step(F[idx], slog[idx])
    return best_found / true_best * 100


def run_ensemble(F, slog, beta, seed=0):
    """深度集成 (5 预测器) UCB"""
    rng = np.random.RandomState(seed)
    n = len(slog)
    true_best = float(np.max(slog[slog > 0]))
    members = [SymbolPredictor(s_dim=6, seed=seed + 10 * k) for k in range(5)]
    best_found = 0.0
    warm_idx = rng.choice(n, WARM, replace=False)
    seen = set()
    for i in warm_idx:
        for m in members:
            m.step(F[i], slog[i])
        best_found = max(best_found, slog[i])
        seen.add(i)
    for _ in range(BUDGET - WARM):
        unseen = [i for i in range(n) if i not in seen]
        mus = np.array([[m.predict(F[i]) for m in members] for i in unseen])
        mean = mus.mean(1)
        unc = mus.std(1)
        score = mean + beta * unc
        idx = unseen[int(np.argmax(score))]
        seen.add(idx)
        best_found = max(best_found, slog[idx])
        for m in members:
            m.step(F[idx], slog[idx])
    return best_found / true_best * 100


def main():
    print("=" * 80)
    print("BB 真实结构池 — 不确定度药方最终裁决 (伪影池修正后)")
    print("=" * 80)

    res = {m: [] for m in ["A fixed", "B egreedy", "C1 ucb0.5", "C2 ucb1.0",
                           "C3 ucb2.0", "C4 maxσ", "C5 dist", "C6 ens1.0",
                           "R ring"]}
    for seed in [1, 7, 42]:
        rules_list, slog, F_beh = build_pool_self(seed)
        inject_real_counters(rules_list, slog, F_beh, seed)
        true_best = float(np.max(slog[slog > 0]))

        # A fixed
        eff, _ = guided(SymbolPredictor(s_dim=6, seed=seed), F_beh, slog,
                        "fixed", seed=seed)
        res["A fixed"].append(eff)
        # B egreedy
        eff, _ = guided(SymbolPredictor(s_dim=6, seed=seed), F_beh, slog,
                        "homeo", BBHomeostasis(hunger_rate=0.05), None, seed=seed)
        res["B egreedy"].append(eff)
        # C1-C4 不确定度
        for tag, beta in [("C1 ucb0.5", 0.5), ("C2 ucb1.0", 1.0),
                          ("C3 ucb2.0", 2.0)]:
            res[tag].append(run_uncertainty(UncertaintyPredictor(seed=seed),
                                             F_beh, slog, "ucb", beta=beta,
                                             seed=seed))
        res["C4 maxσ"].append(run_uncertainty(
            UncertaintyPredictor(seed=seed), F_beh, slog, "max_sigma", seed=seed))
        res["C5 dist"].append(run_uncertainty(None, F_beh, slog, "dist", seed=seed))
        res["C6 ens1.0"].append(run_ensemble(F_beh, slog, 1.0, seed=seed))
        # R 环内容 (正解参考)
        scores = np.array([static_score_v2(r) for r in rules_list])
        order = np.argsort(-scores)
        best = max(slog[i] for i in order[:BUDGET])
        res["R ring"].append(best / true_best * 100)

    print(f"\n{'模式':<14} | {'效率':>7} | 3 seed 明细")
    print("-" * 60)
    for m in res:
        print(f"{m:<14} | {np.mean(res[m]):6.1f}% | {[f'{x:.0f}' for x in res[m]]}")

    r = np.mean(res["R ring"])
    best_u = max(np.mean(res[k]) for k in ["C1 ucb0.5", "C2 ucb1.0", "C3 ucb2.0",
                                           "C4 maxσ", "C5 dist", "C6 ens1.0"])
    best_uk = max(["C1 ucb0.5", "C2 ucb1.0", "C3 ucb2.0", "C4 maxσ", "C5 dist",
                   "C6 ens1.0"], key=lambda k: np.mean(res[k]))
    a = np.mean(res["A fixed"])
    print("\n" + "=" * 80)
    print(f"最终裁决 (真实结构池):")
    print(f"  环内容正解: {r:.1f}% | 不确定度最优 {best_uk}: {best_u:.1f}% | "
          f"回归: {a:.1f}%")
    if best_u >= r - 5:
        print("  ✅ LeCun 药方在真实池成立: 不确定度 ≥ 环内容")
    elif best_u >= a:
        print("  ⚠️ LeCun 药方部分成立: 不确定度 > 回归, 但仍低于环内容")
    else:
        print("  ❌ LeCun 药方在真实池仍失败: 不确定度 ≤ 回归, 环内容独占正解")
    print("  结论: " + ("伪影池裁决被真实池部分推翻, 药方需按场景分级"
                        if best_u >= a else
                        "伪影池裁决在真实池得到确认: 不确定度不是结构长尾的探索信号"))


if __name__ == "__main__":
    main()
