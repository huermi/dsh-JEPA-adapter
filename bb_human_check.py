"""
BB 人类式静态结构预筛实验 (bb_human_check.py)
================================================
检验"人类会怎么做"的实证版: 真实计数器机器 (结构真实, 非行为伪影) 注入,
人类式纯图论预筛 (零模拟) 是否碾压统计方法.

对照 (同一真实结构池, 预算 60 次验证, 3 seed):
  A fixed        行为指纹回归引导 (旧基线)
  B egreedy      行为指纹 ε-greedy 内稳态 (旧最强 43.3%)
  C human_static 人类式: 纯图论分数排序, 预算全花在前 60 高分 (零模拟, 不学习)
  D human+reg    人类式预筛 top 200 → 子集内行为回归引导 (预筛+统计混合)

关键诊断: 注入的计数器机器在静态分数排序中的排名 (人类式预筛能否抓到)
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import sys
import numpy as np

sys.path.insert(0, REPO_ROOT)
from jpi4_symbolic_bb import random_rules, run_machine, extract_symbols, tarjan_scc, SymbolPredictor
from jpi9_configurator import make_counter_machine
from bb_component_check import BBHomeostasis, guided

N_CAND = 1500
BUDGET = 60
WARM = 10
N_INJECT = 4


def build_pool_self(seed):
    """自建池 (完全掌控): 1500 台随机 n=4 机器 + behavior 6 维特征
    slog 为 log1p 尺度 (与 jpi6 build_pool 一致, 注入的 11.5 才是真长尾)"""
    rng = np.random.RandomState(seed)
    rules_list = [random_rules(4, rng) for _ in range(N_CAND)]
    steps = np.array([run_machine(r, 4) for r in rules_list])
    slog = np.array([np.log1p(max(s, 0)) for s in steps], dtype=np.float32)
    F_beh = np.array([extract_symbols(r, 4, mode="behavior") for r in rules_list],
                     dtype=np.float32)
    return rules_list, slog, F_beh


def inject_real_counters(rules_list, slog, F_beh, seed):
    """结构真实注入: 真实计数器机器 (规则表 + 真实特征 + 真实步数)"""
    rng = np.random.RandomState(seed)
    logs = []
    for k in range(N_INJECT):
        idx = rng.randint(len(slog))
        cr = make_counter_machine(4)
        steps = run_machine(cr, 4, max_steps=200000)
        logv = np.log1p(100000) if steps < 0 else np.log1p(steps)
        rules_list[idx] = cr
        slog[idx] = logv
        F_beh[idx] = extract_symbols(cr, 4, mode="behavior")
        logs.append((idx, steps, round(float(logv), 2)))
    return logs


def static_score(rules, n=4):
    """人类式静态分数: 纯图论, 零模拟.
    人类直觉: 循环 + 大 SCC + 少停止出口 + 全可达 → 可能跑久"""
    adj = np.zeros((n, n), dtype=np.float32)
    halt = 0
    for i in range(n * 2):
        st = i // 2
        _, _, nxt = rules[i]
        if nxt < 0:
            halt += 1
        else:
            adj[st, nxt] += 1
    reachable = np.zeros(n, dtype=bool)
    stack = [0]
    while stack:
        u = stack.pop()
        if reachable[u]:
            continue
        reachable[u] = True
        stack.extend(np.where(adj[u] > 0)[0])
    reach_ratio = reachable.sum() / n
    sub = adj[reachable][:, reachable]
    has_cycle = 0.0
    if sub.shape[0] >= 1:
        if (np.trace(sub) > 0 or
                np.linalg.matrix_power(sub, min(8, sub.shape[0])).sum() > sub.shape[0]):
            has_cycle = 1.0
    sccs = tarjan_scc(adj)
    sizes = np.array([len(c) for c in sccs]) if sccs else np.zeros(0)
    max_scc = (sizes.max() / n) if len(sizes) else 0.0
    halt_den = halt / (n * 2)
    return has_cycle * max_scc * (1.0 - halt_den) * reach_ratio


def static_score_v2(rules, n=4):
    """人类式 v2 (反汇编级): 图论 + 环内容分析.
    人类看规则表会追问'这个环在干什么': 环内规则若写1+移动 → 磁带扩张 → 计数器候选;
    若原地写0 → 空转死循环. 这是静态可判的 (无需模拟)"""
    adj = np.zeros((n, n), dtype=np.float32)
    writes = np.zeros((n, n, 2), dtype=np.float32)  # [st, nxt, 0=写1比例,1=移动比例]
    cnt = np.zeros((n, n), dtype=np.float32)
    halt = 0
    for i in range(n * 2):
        st = i // 2
        ws, d, nxt = rules[i]
        if nxt < 0:
            halt += 1
        else:
            adj[st, nxt] += 1
            cnt[st, nxt] += 1
            writes[st, nxt, 0] += (1.0 if ws == 1 else 0.0)
            writes[st, nxt, 1] += (1.0 if d == 1 else 0.0)
    reachable = np.zeros(n, dtype=bool)
    stack = [0]
    while stack:
        u = stack.pop()
        if reachable[u]:
            continue
        reachable[u] = True
        stack.extend(np.where(adj[u] > 0)[0])
    reach_ratio = reachable.sum() / n
    # 最大 SCC
    sccs = tarjan_scc(adj)
    sizes = np.array([len(c) for c in sccs]) if sccs else np.zeros(0)
    max_scc = (sizes.max() / n) if len(sizes) else 0.0
    # 最大 SCC 内的"写1+移动"比例 (环内容: 扩张 vs 空转)
    if len(sizes):
        big = sccs[int(np.argmax(sizes))]
        mask = np.zeros(n, dtype=bool)
        mask[big] = True
        sub_cnt = cnt[mask][:, mask]
        sub_w1 = writes[mask][:, mask, 0]
        sub_mv = writes[mask][:, mask, 1]
        tot = sub_cnt.sum()
        expansion = ((sub_w1.sum() / tot) * (sub_mv.sum() / tot)) if tot > 0 else 0.0
    else:
        expansion = 0.0
    halt_den = halt / (n * 2)
    return max_scc * reach_ratio * (1.0 - halt_den) * expansion


def run_static(scores, slog, budget=BUDGET):
    """人类式 C: 纯静态分数排序, 预算花在前 N 高分 (零模拟, 无学习)"""
    order = np.argsort(-scores)
    best = 0.0
    hits = 0
    for i in order[:budget]:
        best = max(best, slog[i])
        if slog[i] >= 8.0:
            hits += 1
    return best, hits


def run_static_regress(F, slog, scores, seed=0):
    """人类式 D: 静态预筛 top 200 → 子集内行为回归引导 60 次"""
    top = np.argsort(-scores)[:200]
    sub_idx = set(top.tolist())
    rng = np.random.RandomState(seed)
    pred = SymbolPredictor(s_dim=6, seed=seed)
    best = 0.0
    hits = 0
    warm_idx = rng.choice(list(sub_idx), WARM, replace=False)
    seen = set()
    for i in warm_idx:
        pred.step(F[i], slog[i])
        best = max(best, slog[i])
        if slog[i] >= 8.0:
            hits += 1
        seen.add(i)
    for _ in range(BUDGET - WARM):
        cand = [i for i in sub_idx if i not in seen]
        if not cand:
            break
        scores_pred = [(pred.predict(F[i]), i) for i in cand]
        scores_pred.sort(key=lambda x: -x[0])
        idx = scores_pred[0][1]
        seen.add(idx)
        best = max(best, slog[idx])
        if slog[idx] >= 8.0:
            hits += 1
        pred.step(F[idx], slog[idx])
    return best, hits


def main():
    print("=" * 78)
    print("BB 人类式静态结构预筛 — 真实计数器机器注入 (非行为伪影)")
    print("=" * 78)

    all_res = {m: [] for m in ["A fixed", "B egreedy", "C human_static",
                               "C2 human_v2", "D human+reg"]}
    all_hits = {m: [] for m in all_res}
    ranks_all = []

    for seed in [1, 7, 42]:
        rules_list, slog, F_beh = build_pool_self(seed)
        inj = inject_real_counters(rules_list, slog, F_beh, seed)
        true_best = float(np.max(slog[slog > 0]))
        if seed == 1:
            print(f"\n池: 真实最长 log {true_best:.2f} | 注入: "
                  f"{[(round(x[2],1)) for x in inj]}")
            print(f"  注入机器实测步数: {[x[1] for x in inj]} "
                  f"(-1 = 超 200000 步不停机, 真长尾)")

        # 静态分数 (全池零模拟)
        scores = np.array([static_score(r) for r in rules_list])
        scores2 = np.array([static_score_v2(r) for r in rules_list])
        order = np.argsort(-scores)
        order2 = np.argsort(-scores2)
        if seed == 1:
            inj_idx = [x[0] for x in inj]
            ranks = [int(np.where(order == i)[0][0]) for i in inj_idx]
            ranks2 = [int(np.where(order2 == i)[0][0]) for i in inj_idx]
            ranks_all.append(ranks)
            print(f"注入机器在静态排序中的排名: {ranks} (共 {N_CAND})")
            print(f"注入机器在 v2(环内容) 排序中的排名: {ranks2}")
            top_ranks = {i: (round(float(scores[i]), 3), round(float(slog[i]), 1))
                         for i in order[:8]}
            print(f"静态前 8: {top_ranks}")

        # A 固定引导 (行为回归)
        pred = SymbolPredictor(s_dim=6, seed=seed)
        eff, st = guided(pred, F_beh, slog, "fixed", seed=seed)
        all_res["A fixed"].append(eff)
        all_hits["A fixed"].append(st["feed"] if "feed" in st else 0)

        # B ε-greedy (行为回归 + 内稳态)
        pred = SymbolPredictor(s_dim=6, seed=seed)
        homeo = BBHomeostasis(hunger_rate=0.05)
        eff, st = guided(pred, F_beh, slog, "homeo", homeo, None, seed=seed)
        all_res["B egreedy"].append(eff)
        all_hits["B egreedy"].append(st["feed"])

        # C 人类式纯静态预筛
        best, hits = run_static(scores, slog)
        all_res["C human_static"].append(best / true_best * 100)
        all_hits["C human_static"].append(hits)

        # C2 人类式 v2 (环内容分析)
        best, hits = run_static(scores2, slog)
        all_res["C2 human_v2"].append(best / true_best * 100)
        all_hits["C2 human_v2"].append(hits)

        # D 预筛 + 回归
        best, hits = run_static_regress(F_beh, slog, scores, seed=seed)
        all_res["D human+reg"].append(best / true_best * 100)
        all_hits["D human+reg"].append(hits)

    print("\n" + "=" * 78)
    print(f"{'模式':<16} | {'效率':>7} | {'长机器命中':>8} | 3 seed 明细")
    print("-" * 78)
    for m in all_res:
        print(f"{m:<16} | {np.mean(all_res[m]):6.1f}% | "
              f"{int(np.mean(all_hits[m])):>5}台 | "
              f"{[f'{x:.0f}' for x in all_res[m]]}")
    print("-" * 78)
    print(f"注入机器静态排名 (seed=1): {ranks_all[0] if ranks_all else 'N/A'}")

    c = np.mean(all_res["C human_static"])
    c2 = np.mean(all_res["C2 human_v2"])
    a = np.mean(all_res["A fixed"])
    b = np.mean(all_res["B egreedy"])
    d = np.mean(all_res["D human+reg"])
    print(f"\n裁决: C(静态) {c:.1f}% | C2(环内容) {c2:.1f}% | D(预筛+回归) {d:.1f}% | "
          f"A(全池回归) {a:.1f}% | B(ε-greedy) {b:.1f}%")
    if c2 >= a and c2 >= b:
        print("✅ 人类式反汇编 (环内容分析) 零模拟超越统计方法 — 结构分析正解成立")
    elif d >= a and d >= b:
        print("✅ 预筛+回归组合 (人类式正确打开方式) 超越单一统计方法")
    else:
        print("⚠️ 人类式方法未超越 — 需要进一步分析")


if __name__ == "__main__":
    main()
