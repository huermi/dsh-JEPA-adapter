"""
BB 不确定度驱动探索对照 (bb_uncertainty_check.py)
==================================================
⚠️ 2026-08-24 修正声明: 本实验在"行为伪影注入"池上进行 (见
bb_component_check.py 修正声明). 伪影池上"不确定度失败"的裁决
部分失真; "统计不寻常 vs 结构特殊正交性"结论仍成立 (bb_human_check
的图论 top 全 log 0 是同一现象), 但 LeCun 药方在真实结构池上
未测 = 开放问题. 真实结构池正解: 环内容分析 (bb_human_check.py).

检验 LeCun 的药方: "探索应该去预测不确定度最高的地方, 而不是随机掷骰子"
(对应上一轮交锋: 他说 ε-greedy 是 RL 的坏习惯, 正确做法是不确定度驱动)

对照模式 (同一候选池 1500 + 注入 4 台 log 8-11 长机器, 预算 60, warm 10):
  A  fixed      固定引导 (μ top)                    —— jpi6 基线 24.0%
  B  egreedy    内稳态 ε-greedy (hr=0.05)           —— 最强基线 43.3%
  C1 ucb_0.5    UCB: μ + 0.5σ top
  C2 ucb_1.0    UCB: μ + 1.0σ top
  C3 ucb_2.0    UCB: μ + 2.0σ top
  C4 max_sigma  纯不确定度: σ top
  C5 max_dist   特征空间离群度: 到已见样本最小距离 top (分布不确定度代理, 免训练)

不确定度来源:
  - C1-C4: UncertaintyPredictor 双头 (μ 头 + logσ 头), NLL 损失训练
  - C5:    纯几何 (无预测器), 检验"不确定度不需要学习"的假设

裁决: C 系列是否 ≥ B (43.3%)? → LeCun 药方是否成立
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import sys
import numpy as np

sys.path.insert(0, REPO_ROOT)
from jpi6_behavioral_symbols import build_pool
from bb_component_check import inject_long_machines, BBHomeostasis, guided

N_CAND = 1500
BUDGET = 60
WARM = 10


class UncertaintyPredictor:
    """双头预测器: μ (log 步数) + σ (不确定度). NLL 损失:
    L = log σ + (y-μ)²/(2σ²)  →  最优 σ = |err|"""
    def __init__(self, s_dim=6, h_dim=24, lr=0.05, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(s_dim, h_dim).astype(np.float32) * 0.2
        self.W2_mu = np.zeros(h_dim, dtype=np.float32)
        self.W2_sig = np.zeros(h_dim, dtype=np.float32)
        self.lr = lr

    def predict(self, s):
        """返回 (μ, σ)"""
        h = np.maximum(0, np.dot(s, self.W1))
        mu = float(np.dot(h, self.W2_mu))
        log_sig = float(np.dot(h, self.W2_sig))
        return mu, float(np.exp(np.clip(log_sig, -6, 6)))

    def step(self, s, target):
        h = np.maximum(0, np.dot(s, self.W1))
        mu = float(np.dot(h, self.W2_mu))
        log_sig = float(np.dot(h, self.W2_sig))
        sig = float(np.exp(np.clip(log_sig, -6, 6)))
        err = mu - target
        sig2 = sig * sig + 1e-6
        # NLL 梯度
        g_mu = err / sig2
        g_logsig = 1.0 - err * err / sig2
        self.W2_mu -= self.lr * np.clip(h * g_mu, -0.5, 0.5)
        self.W2_sig -= self.lr * np.clip(h * g_logsig, -0.5, 0.5)
        dH = (self.W2_mu * g_mu + self.W2_sig * g_logsig) * (h > 0)
        self.W1 -= self.lr * np.clip(np.outer(s, dH), -0.5, 0.5)
        return abs(err)


def predict_mu_sig(predictor, s):
    """统一预测接口: 返回 (μ, σ). 单头预测器 σ=0"""
    if hasattr(predictor, "W2_sig"):
        return predictor.predict(s)
    return float(predictor.predict(s)), 0.0


def run_guided(predictor, F, slog, mode, beta=None, seed=0):
    """不确定度驱动引导循环. mode: 'ucb' | 'max_sigma' | 'max_dist' | 'fixed'
    返回 (效率%, 统计)"""
    rng = np.random.RandomState(seed)
    n = len(slog)
    true_best = float(np.max(slog[slog > 0])) if (slog > 0).any() else 1e-9
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

    n_left = BUDGET - WARM
    stats = {"exploit": 0, "explore": 0, "long_found": 0}

    for _ in range(n_left):
        unseen = np.where(~seen_all)[0]
        if mode == "max_dist":
            # 到已见样本的最小特征距离 (离群度) — 免训练分布不确定度
            dists = np.min(
                [np.linalg.norm(F[unseen] - F[i], axis=1) for i in warm_idx],
                axis=0)
            idx = unseen[int(np.argmax(dists))]
            stats["explore"] += 1
        else:
            out = [predict_mu_sig(predictor, F[i]) for i in unseen]
            mus = np.array([o[0] for o in out])
            sigs = np.array([o[1] for o in out])
            if mode == "fixed":
                score = mus
            elif mode == "max_sigma":
                score = sigs
            else:  # ucb
                score = mus + beta * sigs
            idx = unseen[int(np.argmax(score))]
            stats["exploit" if mode == "fixed" else "explore"] += 1

        seen_all[idx] = True
        log_steps = slog[idx]
        best_found = max(best_found, log_steps)
        if log_steps > 4.0:
            stats["long_found"] += 1
        if predictor is not None:
            predictor.step(F[idx], log_steps)

    eff = best_found / true_best * 100
    return eff, stats


def main():
    print("=" * 78)
    print("BB 不确定度驱动探索对照 — LeCun 药方 vs ε-greedy 实证")
    print("=" * 78)
    print("构建候选池 (n=4, behavior, 1500) + 注入 4 台 log 8-11 长机器...")
    F, slog, _ = build_pool(4, 113, "behavior", n_candidates=N_CAND)
    F, slog = inject_long_machines(F, slog)
    true_best = float(np.max(slog[slog > 0]))
    print(f"真实最长 log: {true_best:.2f} | 停机: {(slog>0).sum()}/{N_CAND}\n")

    print(f"{'模式':<14} | {'效率':>7} | {'长机器命中':>8} | 3 seed 明细")
    print("-" * 74)

    results = {}

    # A 固定引导 (普通单头预测器)
    from jpi4_symbolic_bb import SymbolPredictor
    effs, hits = [], []
    for seed in [1, 7, 42]:
        pred = SymbolPredictor(s_dim=6, seed=seed)
        eff, st = run_guided(pred, F, slog, "fixed", seed=seed)
        effs.append(eff); hits.append(st["long_found"])
    results["A fixed"] = np.mean(effs)
    print(f"{'A fixed':<14} | {np.mean(effs):6.1f}% | {int(np.mean(hits)):>5}台 | {[f'{x:.0f}' for x in effs]}")

    # B ε-greedy 内稳态 (最强基线)
    effs, hits = [], []
    for seed in [1, 7, 42]:
        pred = __import__("jpi4_symbolic_bb", fromlist=["SymbolPredictor"]).SymbolPredictor(s_dim=6, seed=seed)
        homeo = BBHomeostasis(hunger_rate=0.05)
        eff, st = guided(pred, F, slog, "homeo", homeo, None, seed=seed)
        effs.append(eff); hits.append(st["feed"])
    results["B egreedy"] = np.mean(effs)
    print(f"{'B egreedy':<14} | {np.mean(effs):6.1f}% | {int(np.mean(hits)):>5}台 | {[f'{x:.0f}' for x in effs]}")

    # C1-C3 UCB
    for beta in [0.5, 1.0, 2.0]:
        effs, hits = [], []
        for seed in [1, 7, 42]:
            pred = UncertaintyPredictor(seed=seed)
            eff, st = run_guided(pred, F, slog, "ucb", beta=beta, seed=seed)
            effs.append(eff); hits.append(st["long_found"])
        tag = f"C{beta} ucb_{beta}"
        results[tag] = np.mean(effs)
        print(f"{tag:<14} | {np.mean(effs):6.1f}% | {int(np.mean(hits)):>5}台 | {[f'{x:.0f}' for x in effs]}")

    # C4 max σ
    effs, hits = [], []
    for seed in [1, 7, 42]:
        pred = UncertaintyPredictor(seed=seed)
        eff, st = run_guided(pred, F, slog, "max_sigma", seed=seed)
        effs.append(eff); hits.append(st["long_found"])
    results["C4 maxσ"] = np.mean(effs)
    print(f"{'C4 maxσ':<14} | {np.mean(effs):6.1f}% | {int(np.mean(hits)):>5}台 | {[f'{x:.0f}' for x in effs]}")

    # C5 max_dist (免训练离群度)
    effs, hits = [], []
    for seed in [1, 7, 42]:
        eff, st = run_guided(None, F, slog, "max_dist", seed=seed)
        effs.append(eff); hits.append(st["long_found"])
    results["C5 dist"] = np.mean(effs)
    print(f"{'C5 dist':<14} | {np.mean(effs):6.1f}% | {int(np.mean(hits)):>5}台 | {[f'{x:.0f}' for x in effs]}")

    # 裁决
    print("\n" + "=" * 78)
    best_c = max({k: v for k, v in results.items() if k.startswith("C")},
                 key=lambda k: results[k])
    base_b = results["B egreedy"]
    print(f"不确定度系列最优: {best_c} = {results[best_c]:.1f}% vs ε-greedy {base_b:.1f}%")
    if results[best_c] >= base_b - 1.0:
        verdict = "✅ LeCun 药方成立/打平: 不确定度驱动无需随机性即可达到或超过 ε-greedy"
    else:
        verdict = "❌ LeCun 药方未达标: 不确定度驱动 < ε-greedy, 随机探索在结构长尾上仍必要"
    print(f"裁决: {verdict}")
    print(f"  (A fixed 基线 {results['A fixed']:.1f}% | B egreedy {base_b:.1f}% | "
          f"C 最优 {results[best_c]:.1f}%)")
    print(f"\n结论: {'支持' if results[best_c] >= base_b - 1.0 else '反对'} "
          f"LeCun 的'不确定度替代随机探索'立场 — "
          f"{'但注意 C5 免训练离群度已够用, 无需 NLL 头' if results.get('C5 dist',0) >= base_b - 1.0 else ''}")


if __name__ == "__main__":
    main()
