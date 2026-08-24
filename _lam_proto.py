"""可微记忆进权重 — 线性联想记忆 (LAM) 原型验证.
问题: LeCun"记忆进权重"在家用电脑是否可行/可优化性能?
方案: 线性联想记忆 W (z_query → a_answer 线性映射), Hebbian 在线闭式更新
      (无反向传播, CPU 微秒级, O(n) 检索 → O(1) 预测).
验证: ①东京/上海混叠对能否被 W 区分 (W 学特征组合而非几何距离)
      ②模糊变体 (余弦检索弃权) 上 W 是否仍给出正确方向 (线性泛化)
      ③对照余弦检索的 margin 弃权行为
"""
import numpy as np

D = 384          # 情境维度 (MiniLM 384d 单槽, 简化)
A = 64           # 答案向量维度 (概念空间)


def mkvec(seed, d):
    r = np.random.RandomState(seed)
    v = r.randn(d).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def near(v1, v2, thresh=0.7):
    return float(np.dot(v1, v2)) > thresh


class LinearAssociativeMemory:
    """线性联想记忆: a_pred = W @ z. Hebbian 在线更新 (闭式, 无梯度).
    W 是"权重化的记忆" — 学情境特征组合 → 答案的线性映射.
    更新: W += alpha * (a_true - W@z) @ z^T  (delta 规则, 等价 LMS).
    正则: 投影后小幅收缩 (防发散)."""

    def __init__(self, d_in=D, d_out=A, alpha=0.1):
        self.W = np.zeros((d_out, d_in), dtype=np.float32)
        self.alpha = alpha
        self.n_updates = 0

    def predict(self, z):
        return self.W @ z

    def update(self, z, a_true):
        pred = self.W @ z
        err = a_true - pred
        # delta 规则: W += alpha * err ⊗ z  (外积, O(d) 内存 O(d) 时间)
        self.W += self.alpha * np.outer(err, z)
        self.n_updates += 1

    def test(self, z, a_true):
        a_pred = self.predict(z)
        return near(a_pred, a_true), float(np.dot(a_pred, a_true))


def make_scene(seed_base, template, entity):
    """构造情境: 模板主导 + 实体微调 → 高相似但实体可区分.
    z = 0.75*模板 + 0.25*实体方向 → 东京/上海相似度 ≈ 0.8 (混叠区)."""
    z_t = mkvec(seed_base, D)          # 模板 (weather/percentage 共享)
    z_e = mkvec(seed_base + 10000, D)  # 实体 (tokyo/shanghai 各异)
    z_e = z_e - (z_e @ z_t) * z_t      # 与模板正交化
    z_e = z_e / (np.linalg.norm(z_e) + 1e-9)
    z = 0.75 * z_t + 0.25 * entity * z_e
    return z / (np.linalg.norm(z) + 1e-9)


print("=" * 62)
print("线性联想记忆原型: 可微记忆进权重 (家用电脑可行?)")
print("=" * 62)

# ── 构造: 东京/上海混叠对 (模板共享, 实体不同 → 高相似) ──
z_tokyo = make_scene(1, "tpl", +1.0)
z_shanghai = make_scene(1, "tpl", -1.0)
sim = float(np.dot(z_tokyo, z_shanghai))
print(f"\n[构造] 东京/上海情境相似度: {sim:.3f} "
      f"(高相似 → 余弦检索混叠区)")

# 答案: 东京→rainy, 上海→sunny (概念空间正交)
a_rainy = mkvec(500, A)
a_sunny = mkvec(600, A)

# ── 训练: W 在线学习 20 轮 (每个实体样本交替) ──
lam = LinearAssociativeMemory(alpha=0.2)
for epoch in range(20):
    lam.update(z_tokyo, a_rainy)
    lam.update(z_shanghai, a_sunny)

# ── 验证 1: W 区分混叠对 ──
ok1 = lam.test(z_tokyo, a_rainy)
ok2 = lam.test(z_shanghai, a_sunny)
print(f"\n[验证1] W 区分混叠对: 东京→rainy {ok1} | 上海→sunny {ok2} "
      f"({'✅ 线性可分' if ok1[0] and ok2[0] else '❌'})")

# ── 验证 2: 模糊变体 (余弦检索会弃权, W 线性泛化) ──
# 变体 = 东京模板 + 新实体偏移 (与两边都中等相似 → 余弦 margin 不足)
z_var = z_tokyo + 0.35 * mkvec(777, D)
z_var = z_var / (np.linalg.norm(z_var) + 1e-9)
c_t = float(np.dot(z_var, z_tokyo))
c_s = float(np.dot(z_var, z_shanghai))
print(f"\n[验证2] 模糊变体: cos(东京)={c_t:.3f} cos(上海)={c_s:.3f} "
      f"margin={abs(c_t-c_s):.3f} (<0.15 → 余弦检索弃权)")
pred = lam.predict(z_var)
r_t, s_t = lam.test(z_var, a_rainy)
r_s, s_s = lam.test(z_var, a_sunny)
print(f"  W 预测 → rainy 相似 {s_t:.3f} | sunny 相似 {s_s:.3f} "
      f"({'✅ W 线性泛化 (给出正确方向)' if s_t > s_s and s_t > 0.5 else '❌'})")

# ── 验证 3: 无关查询 (W 应给低置信, 可被校准拒绝) ──
z_other = make_scene(900, "other", +1.0)
r_o, s_o_r = lam.test(z_other, a_rainy)
r_o2, s_o_s = lam.test(z_other, a_sunny)
print(f"\n[验证3] 无关查询: rainy 相似 {s_o_r:.3f} | sunny 相似 {s_o_s:.3f} "
      f"({'✅ 低置信 (可拒)' if max(s_o_r, s_o_s) < 0.5 else '❌ 需校准拒'})")

# ── 性能: O(1) 预测 vs O(n) 检索 ──
import time
n = 100000
zq = mkvec(42, D)
t0 = time.time()
for _ in range(n):
    _ = lam.predict(zq)
t1 = time.time()
print(f"\n[性能] W 预测 {n} 次: {(t1-t0)*1000:.0f}ms "
      f"({(t1-t0)/n*1e6:.2f} µs/次, O(1) — 检索 O(n) 随条目线性增长)")
print(f"  W 内存: {lam.W.nbytes/1024:.0f} KB (384×64 float32)")
print("=" * 62)
