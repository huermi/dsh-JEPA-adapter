"""
BB 引导组件 (bb_guided_component.py)
=====================================
纠正清单待办②③: 把 L0 图论粗筛 + L1 环内容排序做成可复用组件,
替代"行为回归 + ε-greedy"作为 BB 结构长尾的引导机制.

组件接口 (面向 JepaAgent 接入):
  BBStructureGuide(n_states, top_k)
    screen_scores(rules_list) -> np.ndarray   # L0+L1 零模拟分数 (全池)
    guide(rules_list, budget, mode) -> list[int]  # 预算分配
        mode="pure"   : 环内容 top budget 直接验证 (100% 正解)
        mode="hybrid" : top_k 子集 + 行为回归细选 (预筛+统计组合)

验收: 3 seed, 真实结构池 (1500 + 4 台真实计数器 log 11.51, 预算 60)
对照: A fixed(回归 50.1%) / B egreedy(24.8%) / R ring(pure 100%)
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import sys
import numpy as np

sys.path.insert(0, REPO_ROOT)
from bb_human_check import build_pool_self, inject_real_counters, static_score_v2
from bb_component_check import BBHomeostasis, guided
from jpi4_symbolic_bb import SymbolPredictor
from components.symbol import RingContentSymbol

BUDGET = 60
WARM = 10


class BBStructureGuide:
    """BB 引导组件: L0 图论粗筛 + L1 环内容排序 + 预算分配

    设计依据 (bb_human_check 实证):
      - 天真图论 (有环/SCC) 无法区分计数器与空转 (图论同构)
      - 环内容 (写1×移动=磁带扩张) 是零模拟可判的结构长尾正解 (100%)
      - 预筛 top_k + 统计细选 (hybrid) 是次优组合 (75%)"""
    def __init__(self, n_states: int = 4, top_k: int = 200):
        self.ring = RingContentSymbol(n_states=n_states)
        self.top_k = top_k

    def screen_scores(self, rules_list: list) -> np.ndarray:
        """L0+L1: 全池零模拟环内容分数 (越大越可能长运行)"""
        return np.array([self.ring.structure_score(r) for r in rules_list])

    def guide(self, rules_list: list, budget: int = BUDGET,
              mode: str = "pure", seed: int = 0) -> list[int]:
        """预算分配: 返回要验证的机器索引列表
        mode='pure'  : 环内容 top budget (零模拟, 无学习, 确定性正解)
        mode='hybrid': 环内容 top_k 子集内随机 budget 个 (预筛+抽查)"""
        scores = self.screen_scores(rules_list)
        order = list(np.argsort(-scores))
        if mode == "pure":
            return order[:budget]
        rng = np.random.RandomState(seed)
        sub = order[:self.top_k]
        rng.shuffle(sub)
        return sub[:budget]


def evaluate_guide(indices, slog):
    """给定验证顺序, 返回效率% 与命中数"""
    true_best = float(np.max(slog[slog > 0]))
    best = max(slog[i] for i in indices[:BUDGET])
    hits = sum(1 for i in indices[:BUDGET] if slog[i] >= 8.0)
    return best / true_best * 100, hits


def main():
    print("=" * 80)
    print("BB 引导组件验收 (bb_guided_component.py) — L0+L1 组件化")
    print("=" * 80)

    guide = BBStructureGuide()
    res = {"A fixed": [], "B egreedy": [], "R pure": [], "R hybrid": []}
    hits = {"R pure": [], "R hybrid": []}

    for seed in [1, 7, 42]:
        rules_list, slog, F_beh = build_pool_self(seed)
        inject_real_counters(rules_list, slog, F_beh, seed)
        true_best = float(np.max(slog[slog > 0]))

        # 对照 A/B
        eff, _ = guided(SymbolPredictor(s_dim=6, seed=seed), F_beh, slog,
                        "fixed", seed=seed)
        res["A fixed"].append(eff)
        eff, _ = guided(SymbolPredictor(s_dim=6, seed=seed), F_beh, slog,
                        "homeo", BBHomeostasis(hunger_rate=0.05), None, seed=seed)
        res["B egreedy"].append(eff)

        # 组件 pure / hybrid
        for mode in ["pure", "hybrid"]:
            indices = guide.guide(rules_list, BUDGET, mode, seed=seed)
            eff, h = evaluate_guide(indices, slog)
            res[f"R {mode}"].append(eff)
            hits[f"R {mode}"].append(h)

    print(f"\n{'模式':<12} | {'效率':>7} | {'命中':>4} | 3 seed 明细")
    print("-" * 64)
    for m in res:
        hs = f" | {int(np.mean(hits[m]))}台" if m in hits else " |  — "
        print(f"{m:<12} | {np.mean(res[m]):6.1f}%{hs} | "
              f"{[f'{x:.0f}' for x in res[m]]}")

    r = np.mean(res["R pure"])
    print("\n" + "=" * 80)
    print(f"组件验收: R pure {r:.1f}% (对照 A {np.mean(res['A fixed']):.1f}% / "
          f"B {np.mean(res['B egreedy']):.1f}%)")
    print(f"{'✅ BB 引导组件化成立 (零模拟, 接口干净, 100% 保持)' if r >= 99 else '❌ 需检查'}")
    print("组件用法: guide.screen_scores(rules_list) 零模拟排序 → "
          "guide.guide(rules_list, budget, 'pure') 直接验证 top")


if __name__ == "__main__":
    main()
