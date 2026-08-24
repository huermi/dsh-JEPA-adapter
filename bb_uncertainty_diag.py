"""
BB 不确定度诊断 + 集成不确定度 (bb_uncertainty_diag.py)
=======================================================
诊断: σ 头 / 特征距离 是否真的给注入的长机器高不确定度?
对照: 深度集成不确定度 (deep ensemble — LeCun 阵营会推荐的标准贝叶斯近似)
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import sys
import numpy as np

sys.path.insert(0, REPO_ROOT)
from jpi6_behavioral_symbols import build_pool
from bb_component_check import inject_long_machines
from bb_uncertainty_check import UncertaintyPredictor, predict_mu_sig, run_guided

N_CAND = 1500
WARM = 10


def diag():
    """诊断: 长机器 (log>=8) vs 普通停机机器的 预测/不确定度/距离"""
    F, slog, _ = build_pool(4, 113, "behavior", n_candidates=N_CAND)
    F, slog = inject_long_machines(F, slog)
    rng = np.random.RandomState(1)

    long_idx = np.where(slog >= 8.0)[0]
    normal_idx = np.where((slog > 0) & (slog < 8.0))[0]
    print(f"长机器: {len(long_idx)} 台 (log {[round(slog[i],1) for i in long_idx]}) | "
          f"普通停机: {len(normal_idx)} 台")

    pred = UncertaintyPredictor(seed=1)
    warm_idx = rng.choice(len(slog), WARM, replace=False)
    for i in warm_idx:
        pred.step(F[i], slog[i])

    # 对长机器和普通机器的 (μ, σ) 预测
    mu_l, sig_l = [], []
    for i in long_idx:
        m, s = pred.predict(F[i])
        mu_l.append(m); sig_l.append(s)
    # 普通停机机器采样 100 台
    samp = rng.choice(normal_idx, 100, replace=False)
    mu_n, sig_n = [], []
    for i in samp:
        m, s = pred.predict(F[i])
        mu_n.append(m); sig_n.append(s)

    # 特征距离: 到 warm 样本的最小距离
    d_l = [min(np.linalg.norm(F[i] - F[j]) for j in warm_idx) for i in long_idx]
    d_n = [min(np.linalg.norm(F[i] - F[j]) for j in warm_idx) for i in samp]

    print("\n--- NLL σ 头诊断 ---")
    print(f"长机器   μ 均值 {np.mean(mu_l):.3f} | σ 均值 {np.mean(sig_l):.3f} (范围 {min(sig_l):.3f}-{max(sig_l):.3f})")
    print(f"普通机器 μ 均值 {np.mean(mu_n):.3f} | σ 均值 {np.mean(sig_n):.3f} (范围 {min(sig_n):.3f}-{max(sig_n):.3f})")
    print(f"→ σ 头是否区分长/普通: {'✅ 是 (长机器 σ 显著高)' if np.mean(sig_l) > np.mean(sig_n) * 1.5 else '❌ 否 (σ 头没学会分布外信号)'}")

    print("\n--- 特征距离 (离群度) 诊断 ---")
    print(f"长机器   最小距离均值 {np.mean(d_l):.3f} (范围 {min(d_l):.3f}-{max(d_l):.3f})")
    print(f"普通机器 最小距离均值 {np.mean(d_n):.3f} (范围 {min(d_n):.3f}-{max(d_n):.3f})")
    print(f"→ 距离是否区分长/普通: {'✅ 是' if np.mean(d_l) > np.mean(d_n) * 1.5 else '❌ 否 (全 1.0 特征不在空间角落)'}")

    # 排序检查: max_dist 会先选到什么?
    dists_all = np.array([min(np.linalg.norm(F[i] - F[j]) for j in warm_idx)
                          for i in range(len(slog))])
    top_dist = np.argsort(-dists_all)[:10]
    print(f"\nmax_dist 前 10 选择: {[(int(i), round(dists_all[i],2), round(slog[i],1)) for i in top_dist]}")
    print(f"  (log>=8 才算命中) → 命中共 {sum(slog[i]>=8 for i in top_dist)} 台")

    # UCB top 10 (β=1)
    scores = np.array([predict_mu_sig(pred, F[i])[0] + 1.0 * predict_mu_sig(pred, F[i])[1]
                       for i in range(len(slog))])
    top_ucb = np.argsort(-scores)[:10]
    print(f"\nUCB(β=1) 前 10 选择: {[(int(i), round(scores[i],2), round(slog[i],1)) for i in top_ucb]}")
    print(f"  → 命中共 {sum(slog[i]>=8 for i in top_ucb)} 台")


def ensemble_variant():
    """深度集成不确定度: 5 个独立预测器, σ = 预测方差 (贝叶斯近似)"""
    from jpi4_symbolic_bb import SymbolPredictor
    F, slog, _ = build_pool(4, 113, "behavior", n_candidates=N_CAND)
    F, slog = inject_long_machines(F, slog)
    n = len(slog)
    true_best = float(np.max(slog[slog > 0]))

    print("\n" + "=" * 78)
    print("深度集成 (5 预测器) — LeCun 阵营标准的贝叶斯不确定度")
    print("=" * 78)

    for beta in [0.5, 1.0, 2.0]:
        effs = []
        for seed in [1, 7, 42]:
            rng = np.random.RandomState(seed)
            members = [SymbolPredictor(s_dim=6, seed=seed + 10 * k) for k in range(5)]
            warm_idx = rng.choice(n, WARM, replace=False)
            seen = set()
            best_found = 0.0
            for i in warm_idx:
                for m in members:
                    m.step(F[i], slog[i])
                best_found = max(best_found, slog[i])
                seen.add(i)
            n_left = 60 - WARM
            for _ in range(n_left):
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
            effs.append(best_found / true_best * 100)
        print(f"ensemble UCB β={beta}: {np.mean(effs):6.1f}% | {[f'{x:.0f}' for x in effs]}")


if __name__ == "__main__":
    diag()
    ensemble_variant()
