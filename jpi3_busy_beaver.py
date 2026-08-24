"""
JPI-3 自主持续学习验证 — Busy Beaver（忙碌海狸）环境
====================================================
用图灵机最大运行步数问题作为持续学习环境:
  - 每阶段 n 递增 (2→3→4→5), 每个 n 是全新的规则分布
  - 长运行机器极端稀有 (BB(2)=6, BB(3)=21, BB(4)=107, BB(5)=47M)
  - 判断"是否停机/运行多久"是真实难题 (停机问题不可判定)
  - agent 任务: 在线学习预测机器运行步数, 用世界模型引导探索

验证目标:
  1. 预测器能否学会"规则模式→运行步数" (E1 下降)
  2. 随 n 递增, 系统能否持续适应新分布 (无灾难性遗忘)
  3. 自适应原型能否涌现 (长运行 vs 短运行机器分簇)
  4. 世界模型能否引导找到池中最长运行的机器
"""
import numpy as np
from collections import deque

MAX_STEPS = 5000          # 死循环超时阈值 (近似停机判定)
N_CANDIDATES = 300        # 每阶段候选机器数
N_STAGES = 4              # n = 2,3,4,5
STAGES = [2, 3, 4, 5]


# ─── 图灵机模拟器 ───────────────────────────────────────────
def random_rules(n: int, rng) -> np.ndarray:
    """随机 n 状态 2 符号图灵机规则表.
    规则: rules[state][sym] = (new_sym, dir, next_state)
    编码为 3 数组: 每个 (state,sym) 对应 [new_sym(0/1), dir(0=L/1=R), next(-1=halt..n-1)]"""
    n_rules = n * 2
    rules = np.zeros((n_rules, 3), dtype=np.int32)
    for i in range(n_rules):
        rules[i, 0] = rng.randint(0, 2)          # new symbol
        rules[i, 1] = rng.randint(0, 2)          # direction
        rules[i, 2] = rng.randint(-1, n)         # next state (-1 = halt)
    return rules


def run_machine(rules: np.ndarray, n: int, max_steps: int = MAX_STEPS) -> int:
    """运行图灵机, 返回停机步数; 超时返回 -1 (视为不停机)"""
    tape = {}          # 稀疏磁带
    pos = 0
    state = 0
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
    return -1  # 超时, 视为不停机 (近似)


def rules_feature(rules: np.ndarray, n: int) -> np.ndarray:
    """规则表 → 特征向量 (固定长度, 供编码器)"""
    # 展平 + 填充到固定长度 (5 状态 × 2 符号 × 3 = 30)
    flat = rules.flatten().astype(np.float32)
    feat = np.zeros(30, dtype=np.float32)
    feat[:len(flat)] = flat
    return feat


# ─── 编码器 (感知层) ────────────────────────────────────────
class Encoder:
    def __init__(self, d_in: int = 30, d_out: int = 8, lr: float = 0.01, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(d_in, d_out).astype(np.float32) * 0.1
        self.b = np.zeros(d_out, dtype=np.float32)
        self.lr = lr

    def encode(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(np.dot(x, self.W) + self.b)
        return h / (np.linalg.norm(h) + 1e-6)

    def step(self, x: np.ndarray, grad: np.ndarray):
        g = np.clip(grad, -0.5, 0.5)
        self.W -= self.lr * g
        self.b -= self.lr * np.sum(grad, axis=0) if grad.ndim > 1 else self.lr * grad


# ─── 预测器 (世界模型: 规则 → 运行步数) ─────────────────────
class Predictor:
    def __init__(self, s_dim: int = 8, h_dim: int = 16,
                 lr: float = 0.05, seed: int = 42):
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


# ─── 记忆 (自适应原型 + surprise 门控) ──────────────────────
class Memory:
    def __init__(self, cap: int = 200, proto_quantile: float = 0.9):
        self.cap = cap
        self.proto_quantile = proto_quantile
        self.items = []              # (s, steps_log, t)
        self.prototypes = []         # 长运行机器原型
        self.short_protos = []       # 短运行机器原型
        self.err_history = deque(maxlen=500)
        self.dist_history = deque(maxlen=1000)

    def write(self, s, pred_err, steps_log, t):
        self.err_history.append(pred_err)
        if len(self.err_history) < 30:
            return
        thresh = float(np.percentile(self.err_history, 85))
        if pred_err > thresh and len(self.items) < self.cap:
            self.items.append((s.copy(), steps_log, t))
            # 长运行机器原型 (steps_log > 阈值)
            bucket = self.prototypes if steps_log > 2.0 else self.short_protos
            if not bucket:
                bucket.append(s.copy())
            else:
                d = float(min(np.linalg.norm(s - p) for p in bucket))
                self.dist_history.append(d)
                if len(self.dist_history) >= 30:
                    d_thresh = float(np.percentile(self.dist_history,
                                                   self.proto_quantile * 100))
                    if d > d_thresh and len(bucket) < 20:
                        bucket.append(s.copy())

    def familiarity(self, s) -> float:
        all_p = self.prototypes + self.short_protos
        if not all_p:
            return 0.0
        d = min(np.linalg.norm(s - p) for p in all_p)
        return 1.0 / (1.0 + d)


# ─── 主实验 ─────────────────────────────────────────────────
def run_stage(n: int, rng, encoder, predictor, memory,
              n_candidates: int = N_CANDIDATES, budget: int = 80):
    """单个 n 阶段: 世界模型引导 vs 随机采样 对照.
    总模拟预算 = budget 台机器 (引导组: 预测选 top; 随机组: 随机选)
    返回两组各自找到的最长停机机器"""
    # 1. 生成候选机器 (含真实步数, 作为 ground truth 评估)
    rules_list = [random_rules(n, rng) for _ in range(n_candidates)]
    steps_list = [run_machine(r, n) for r in rules_list]
    steps_log = np.array([np.log1p(max(s, 0)) for s in steps_list], dtype=np.float32)
    halted = np.array([s >= 0 for s in steps_list])

    # 2. 预热: 随机学 budget/4 台 (让预测器有基本模型)
    warm = list(range(budget // 4))
    for i in warm:
        feat = rules_feature(rules_list[i], n)
        s = encoder.encode(feat)
        err = predictor.step(s, steps_log[i])
        memory.write(s, err, steps_log[i], 0)

    # 3. 引导组: 预测全部未见机器, 模拟预测 top 的剩余预算
    pred_scores = []
    for i in range(n_candidates):
        if i in warm:
            continue
        feat = rules_feature(rules_list[i], n)
        s = encoder.encode(feat)
        pred = predictor.predict(s)
        pred_scores.append((pred, i))
    pred_scores.sort(key=lambda x: -x[0])
    guided_best = 0.0
    guided_halted = 0
    for pred, idx in pred_scores[: (budget - budget // 4)]:
        guided_best = max(guided_best, steps_log[idx])
        guided_halted += int(halted[idx])
        # 引导发现后在线学习 (强化)
        feat = rules_feature(rules_list[idx], n)
        s = encoder.encode(feat)
        err = predictor.step(s, steps_log[idx])
        memory.write(s, err, steps_log[idx], 0)

    # 4. 随机组: 相同预算随机模拟 (对照)
    unseen_idx = [i for i in range(n_candidates) if i not in warm]
    rng.shuffle(unseen_idx)
    rand_best = 0.0
    rand_halted = 0
    for i in unseen_idx[: (budget - budget // 4)]:
        rand_best = max(rand_best, steps_log[i])
        rand_halted += int(halted[i])

    # 5. 全池统计 (ground truth)
    true_best = float(np.max(steps_log))
    true_best_halted = float(np.max(steps_log[halted])) if halted.any() else 0.0

    return {
        "n": n,
        "guided_best": float(guided_best),
        "rand_best": float(rand_best),
        "guided_halted": guided_halted,
        "rand_halted": rand_halted,
        "true_best": true_best,
        "true_best_halted": true_best_halted,
        "n_halted": int(halted.sum()),
        "n_protos_long": len(memory.prototypes),
        "n_protos_short": len(memory.short_protos),
        "pred_mean_err": float(np.mean(list(memory.err_history)[-100:])) if memory.err_history else 0,
    }


if __name__ == "__main__":
    print("=" * 72)
    print("JPI-3 自主持续学习验证 — Busy Beaver 环境 (n=2→5 逐级递增)")
    print("  世界模型引导 vs 随机采样 对照 (相同模拟预算)")
    print("=" * 72)

    # 跨阶段共享 → 持续学习
    encoder = Encoder(seed=7)
    predictor = Predictor(seed=7)
    memory = Memory()

    print(f"{'n':>3} | {'引导最长':>8} {'随机最长':>8} {'真实最长':>8} | "
          f"{'引导停机':>6} {'随机停机':>6} | {'预测误差':>8} | {'原型长/短':>9}")
    print("-" * 80)
    for n in STAGES:
        rng = np.random.RandomState(n * 100)
        r = run_stage(n, rng, encoder, predictor, memory)
        guided_pct = r["guided_best"] / max(r["true_best_halted"], 1e-9) * 100
        rand_pct = r["rand_best"] / max(r["true_best_halted"], 1e-9) * 100
        print(f"{n:>3} | {r['guided_best']:8.2f} {r['rand_best']:8.2f} {r['true_best_halted']:8.2f} | "
              f"{r['guided_halted']:6d} {r['rand_halted']:6d} | "
              f"{r['pred_mean_err']:8.4f} | {r['n_protos_long']}/{r['n_protos_short']}")
        print(f"    引导效率 {guided_pct:.0f}% vs 随机效率 {rand_pct:.0f}% (对停机机器中最长)")

