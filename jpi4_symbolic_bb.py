"""
JPI-4 符号层验证 — 图灵机结构反汇编 → BB 理解与持续学习
=======================================================
设计: 符号层 = 把图灵机规则表"反汇编"为 n 无关的结构描述符:
  - 静态结构: halt 边密度 / 自环 / 可达性 / 有环 / SCC
  - 行为指纹: 短模拟 (150步) 的访问状态数 / 磁带扩张 / 停机标志
核心优势: 结构特征在不同 n 下语义一致 → 跨分布迁移 (这是人类
理解 BB(n) 能迁移到 BB(n+1) 的机制, 也是符号层 vs 扁平规则的本质区别)

对照:
  A. 扁平规则表 (jpi3 现状, 统计引导不稳定)
  B. 符号结构特征 (本实验)
评估: 引导 vs 随机 (相同预算), 跨 n=2→5 持续学习, 预测误差随 n 的迁移
"""
import numpy as np
from collections import deque

MAX_STEPS = 5000
N_CANDIDATES = 2000
BUDGET = 30
STAGES = [2, 3, 4, 5]


# ─── 图灵机 ─────────────────────────────────────────────────
def random_rules(n: int, rng) -> np.ndarray:
    n_rules = n * 2
    rules = np.zeros((n_rules, 3), dtype=np.int32)
    for i in range(n_rules):
        rules[i, 0] = rng.randint(0, 2)
        rules[i, 1] = rng.randint(0, 2)
        rules[i, 2] = rng.randint(-1, n)
    return rules


def run_machine(rules: np.ndarray, n: int, max_steps: int = MAX_STEPS) -> int:
    tape = {}
    pos, state = 0, 0
    for step in range(max_steps):
        sym = tape.get(pos, 0)
        idx = state * 2 + sym
        if idx >= len(rules):
            return step
        new_sym, d, nxt = rules[idx]
        tape[pos] = new_sym
        pos += 1 if d == 1 else -1
        if nxt < 0:
            return step + 1
        state = nxt
    return -1


# ─── 符号层: 结构反汇编 (n 无关描述符) ──────────────────────
def tarjan_scc(adj: np.ndarray) -> list:
    """强连通分量 (迭代 Tarjan 简化版: 用可达互达近似, 小规模够用)"""
    n = adj.shape[0]
    sccs = []
    visited = np.zeros(n, dtype=bool)
    for start in range(n):
        if visited[start]:
            continue
        # 前向可达
        fwd = np.zeros(n, dtype=bool)
        stack = [start]
        while stack:
            u = stack.pop()
            if fwd[u]:
                continue
            fwd[u] = True
            stack.extend(np.where(adj[u] > 0)[0])
        # 反向可达
        bwd = np.zeros(n, dtype=bool)
        stack = [start]
        while stack:
            u = stack.pop()
            if bwd[u]:
                continue
            bwd[u] = True
            stack.extend(np.where(adj[:, u] > 0)[0])
        comp = np.where(fwd & bwd)[0]
        sccs.append(comp)
        visited[comp] = True
    return sccs


def extract_symbols(rules: np.ndarray, n: int, sim_steps: int = 150,
                    mode: str = "full") -> np.ndarray:
    """符号层: 规则表 → n 无关结构描述符 (13 维)
    mode='full'     : 静态结构 + 行为指纹 (13 维)
    mode='behavior' : 仅行为指纹 (6 维, 系统通过短模拟观测即可获得, 无结构注入)"""
    n_rules = n * 2
    feat = np.zeros(13, dtype=np.float32)

    # 1. 静态结构
    halt_count, self_loop = 0, 0
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n_rules):
        st, sym = i // 2, i % 2
        _, _, nxt = rules[i]
        if nxt < 0:
            halt_count += 1
        else:
            adj[st, nxt] += 1
            if nxt == st:
                self_loop += 1
    feat[0] = halt_count / n_rules            # halt 边密度
    feat[1] = self_loop / max(n, 1)           # 自环密度

    # 可达状态 (从状态 0 BFS)
    reachable = np.zeros(n, dtype=bool)
    stack = [0]
    while stack:
        u = stack.pop()
        if reachable[u]:
            continue
        reachable[u] = True
        stack.extend(np.where(adj[u] > 0)[0])
    feat[2] = reachable.sum() / n             # 可达比例

    # 可达子图有环 (非平凡)
    sub = adj[reachable][:, reachable]
    cyc = 0.0
    if sub.shape[0] >= 1:
        d = np.trace(sub) / max(1, sub.shape[0])
        cyc = 1.0 if (np.trace(sub) > 0 or
                      np.linalg.matrix_power(sub, min(8, sub.shape[0])).sum() > sub.shape[0]) else 0.0
    feat[3] = cyc                              # 可达图有环

    # SCC
    sccs = tarjan_scc(adj)
    sizes = np.array([len(c) for c in sccs]) if sccs else np.zeros(0)
    feat[4] = len(sizes) / max(n, 1)          # SCC 数密度
    feat[5] = (sizes.max() / n) if len(sizes) else 0.0  # 最大 SCC

    # 2. 行为指纹: 短模拟 (比完整运行便宜得多)
    tape = {}
    pos, state = 0, 0
    visited_states = set()
    visited_pos = set()
    halted_sim = 0.0
    for step in range(sim_steps):
        sym = tape.get(pos, 0)
        idx = state * 2 + sym
        if idx >= n_rules:
            halted_sim = 1.0
            break
        new_sym, d, nxt = rules[idx]
        visited_states.add(state)
        visited_pos.add(pos)
        tape[pos] = new_sym
        pos += 1 if d == 1 else -1
        if nxt < 0:
            halted_sim = 1.0
            break
        state = nxt
    feat[6] = len(visited_states) / max(n, 1)       # 访问状态多样度
    feat[7] = len(visited_pos) / sim_steps            # 磁带扩张率
    feat[8] = halted_sim                               # 短模拟内停机
    if visited_pos:
        spread = (max(visited_pos) - min(visited_pos)) / sim_steps
        feat[9] = spread                               # 磁带范围扩散
    # 3. 长程迹象: 短模拟最后 30 步是否仍在访问新状态 (增长迹象)
    # 简化: 磁带位置数是否接近 sim_steps (线性扩张 → 计数器候选)
    feat[10] = 1.0 if len(visited_pos) > sim_steps * 0.6 else 0.0  # 线性扩张迹象
    # 4. 状态回归性: 短模拟中状态访问的熵近似 (循环多样性)
    from collections import Counter
    state_counts = Counter()
    pos2, state2 = 0, 0
    tape2 = {}
    for _ in range(min(sim_steps, 60)):
        sym = tape2.get(pos2, 0)
        idx = state2 * 2 + sym
        if idx >= n_rules or rules[idx][2] < 0:
            break
        state_counts[state2] += 1
        new_sym, d, nxt = rules[idx]
        tape2[pos2] = new_sym
        pos2 += 1 if d == 1 else -1
        state2 = nxt
    if state_counts:
        probs = np.array(list(state_counts.values()), dtype=np.float32)
        probs = probs / probs.sum()
        feat[11] = -float(np.sum(probs * np.log(probs + 1e-9))) / np.log(max(n, 2))
    else:
        feat[11] = 0.0
    # 5. 停机边在可达状态中的比例 (可达范围内多少规则能停)
    reach_rules = 0
    for st in np.where(reachable)[0]:
        for sym in [0, 1]:
            idx = st * 2 + sym
            if idx < n_rules and rules[idx][2] < 0:
                reach_rules += 1
    feat[12] = reach_rules / max(2 * max(1, reachable.sum()), 1)

    if mode == "behavior":
        # 仅行为指纹 (6 维): 系统通过短模拟观测即可获得, 无任何结构注入
        return feat[6:12]
    return feat


# ─── 预测器 (在符号特征上学习) ──────────────────────────────
class SymbolPredictor:
    """预测器: 符号特征 → log 运行步数. 符号是 n 无关的, 跨阶段共享"""
    def __init__(self, s_dim: int = 13, h_dim: int = 24, lr: float = 0.05,
                 seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(s_dim, h_dim).astype(np.float32) * 0.2
        self.W2 = np.zeros(h_dim, dtype=np.float32)
        self.lr = lr

    def predict(self, s: np.ndarray) -> float:
        h = np.maximum(0, np.dot(s, self.W1))
        return float(np.dot(h, self.W2))

    def step(self, s: np.ndarray, target: float):
        h = np.maximum(0, np.dot(s, self.W1))
        pred = float(np.dot(h, self.W2))
        err = pred - target
        self.W2 -= self.lr * np.clip(h * err, -0.5, 0.5)
        dH = self.W2 * err * (h > 0)
        self.W1 -= self.lr * np.clip(np.outer(s, dH), -0.5, 0.5)
        return abs(err)


class FlatPredictor:
    """对照: 扁平规则表预测器 (jpi3 现状, 特征随 n 变化维度不同 → 无法跨 n 迁移)"""
    def __init__(self, s_dim: int = 30, h_dim: int = 16, lr: float = 0.05,
                 seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(s_dim, h_dim).astype(np.float32) * 0.2
        self.W2 = np.zeros(h_dim, dtype=np.float32)
        self.lr = lr

    def predict(self, s: np.ndarray) -> float:
        h = np.maximum(0, np.dot(s, self.W1))
        return float(np.dot(h, self.W2))

    def step(self, s: np.ndarray, target: float):
        h = np.maximum(0, np.dot(s, self.W1))
        pred = float(np.dot(h, self.W2))
        err = pred - target
        self.W2 -= self.lr * np.clip(h * err, -0.5, 0.5)
        dH = self.W2 * err * (h > 0)
        self.W1 -= self.lr * np.clip(np.outer(s, dH), -0.5, 0.5)
        return abs(err)


# ─── 实验 ───────────────────────────────────────────────────
def run_stage(n, rng, predictor, use_symbols, budget=BUDGET,
              n_candidates=N_CANDIDATES):
    rules_list = [random_rules(n, rng) for _ in range(n_candidates)]
    steps_list = [run_machine(r, n) for r in rules_list]
    steps_log = np.array([np.log1p(max(s, 0)) for s in steps_list], dtype=np.float32)
    halted = np.array([s >= 0 for s in steps_list])

    # 特征: 符号 or 扁平 (扁平需 padding 到固定 30 维, 因为 n 变化)
    if use_symbols:
        feats = [extract_symbols(r, n) for r in rules_list]
    else:
        feats = []
        for rules in rules_list:
            f = rules.flatten().astype(np.float32)
            pad = np.zeros(30, dtype=np.float32)
            pad[:len(f)] = f
            feats.append(pad)
    # 归一化符号特征 (每维 z-score 到 [0,1])
    F = np.array(feats, dtype=np.float32)
    if use_symbols:
        for j in range(F.shape[1]):
            lo, hi = F[:, j].min(), F[:, j].max()
            if hi > lo:
                F[:, j] = (F[:, j] - lo) / (hi - lo)
            else:
                F[:, j] = 0.0

    # 预热
    warm = list(range(budget // 4))
    for i in warm:
        err = predictor.step(F[i], steps_log[i])

    # 引导组
    pred_scores = []
    for i in range(n_candidates):
        if i in warm:
            continue
        pred_scores.append((predictor.predict(F[i]), i))
    pred_scores.sort(key=lambda x: -x[0])
    guided_best = 0.0
    for pred, idx in pred_scores[: (budget - budget // 4)]:
        guided_best = max(guided_best, steps_log[idx])
        predictor.step(F[idx], steps_log[idx])

    # 随机组
    unseen = [i for i in range(n_candidates) if i not in warm]
    rng.shuffle(unseen)
    rand_best = 0.0
    for i in unseen[: (budget - budget // 4)]:
        rand_best = max(rand_best, steps_log[i])

    true_best_halted = float(np.max(steps_log[halted])) if halted.any() else 0.0
    return {
        "guided_best": float(guided_best),
        "rand_best": float(rand_best),
        "true_best_halted": true_best_halted,
        "n_halted": int(halted.sum()),
    }


def main():
    print("=" * 76)
    print("JPI-4 符号层验证 — 结构反汇编 vs 扁平规则 (BB 引导 + 跨 n 持续学习)")
    print("=" * 76)

    for use_sym, label in [(False, "A. 扁平规则表 (jpi3 现状)"),
                           (True,  "B. 符号结构特征 (本设计)")]:
        # 3 seed 平均
        g_avg, r_avg = [], []
        errs_by_n = []
        for seed in [1, 7, 42]:
            predictor = (SymbolPredictor(seed=seed) if use_sym
                         else FlatPredictor(seed=seed))
            gs, rs = [], []
            for n in STAGES:
                rng = np.random.RandomState(n * 100 + seed)
                r = run_stage(n, rng, predictor, use_sym)
                gs.append(r["guided_best"] / max(r["true_best_halted"], 1e-9) * 100)
                rs.append(r["rand_best"] / max(r["true_best_halted"], 1e-9) * 100)
            g_avg.append(gs); r_avg.append(rs)
        g_m = np.mean(g_avg, axis=0); r_m = np.mean(r_avg, axis=0)
        g_s = np.std(g_avg, axis=0); r_s = np.std(r_avg, axis=0)

        print(f"\n--- {label} ---")
        print(f"  {'n':>3} | {'引导%':>8}±{'':<5} {'随机%':>8}±{'':<5} | {'净增益':>7}")
        for i, n in enumerate(STAGES):
            gain = g_m[i] - r_m[i]
            print(f"  {n:>3} | {g_m[i]:6.1f}±{g_s[i]:4.1f} {r_m[i]:6.1f}±{r_s[i]:4.1f} | {gain:+6.1f}pp")
        overall = np.mean(g_m) - np.mean(r_m)
        print(f"  总体: 引导 {np.mean(g_m):.1f}% vs 随机 {np.mean(r_m):.1f}% → 净增益 {overall:+.1f}pp")


if __name__ == "__main__":
    main()
