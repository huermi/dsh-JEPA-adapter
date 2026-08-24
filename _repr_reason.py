"""表征空间推理加工验证: MiniLM 句向量空间的"关系位移"是否可转移.
问题: 能否在权重/记忆之间做推理泛化 (类似人脑内思维) —
  数学前提: 关系位移向量可转移 (v(A_rainy) - v(A) + v(B) ≈ v(B_rainy)).
验证: ①同模板不同实体 (东京/上海天气) 的关系转移
      ②不同关系类型 (天气/首都/语言) 的位移一致性
      ③转移质量 (cos 相似度 vs 基线)
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import sys, time
import numpy as np
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "body"))
from mini_encoder import get_encoder

enc = get_encoder()
t0 = time.time()
ok = enc.ensure_loaded()
print(f"MiniLM 加载: {ok} ({time.time()-t0:.1f}s)", flush=True)

def v(s):
    return enc.encode(s)

print("\n" + "=" * 62)
print("表征空间推理: 关系位移可转移性 (MiniLM 384d)")
print("=" * 62)

# ── 验证 1: 同模板不同实体 (东京/上海 天气) ──
pairs = [
    # (实体A情境, 实体A+关系, 实体B情境, 实体B+关系, 关系名)
    ("the weather in tokyo", "the weather in tokyo is rainy",
     "the weather in shanghai", "the weather in shanghai is rainy", "weather-rainy"),
    ("the weather in tokyo", "the weather in tokyo is sunny",
     "the weather in shanghai", "the weather in shanghai is sunny", "weather-sunny"),
    ("the capital of japan", "the capital of japan is tokyo",
     "the capital of france", "the capital of france is paris", "capital"),
    ("the language of japan", "the language of japan is japanese",
     "the language of france", "the language of france is french", "language"),
]
print("\n[验证1] 关系位移转移: v(A+关系) - v(A) + v(B) ≈ v(B+关系)?")
results = []
for a, a_r, b, b_r, rel in pairs:
    va, var, vb, vbr = v(a), v(a_r), v(b), v(b_r)
    # 关系位移: d = v(A+关系) - v(A)   →  转移: v(B) + d
    d = var - va
    transfer = vb + d
    transfer = transfer / (np.linalg.norm(transfer) + 1e-9)
    vbr_n = vbr / (np.linalg.norm(vbr) + 1e-9)
    cos = float(np.dot(transfer, vbr_n))
    # 基线: 直接 v(B) 与 v(B+关系) 的相似度 (无推理, 仅语义重叠)
    base = float(np.dot(vb / (np.linalg.norm(vb) + 1e-9), vbr_n))
    results.append((rel, cos, base))
    gain = cos - base
    print(f"  [{rel:<14}] 转移 {cos:.3f} | 基线 {base:.3f} | "
          f"推理增益 {gain:+.3f} {'✅' if gain > 0.02 else ''}")

# ── 验证 2: 类比运算 king-man+woman 式 (实体对之间) ──
print("\n[验证2] 类比运算: v(japan) - v(tokyo) + v(paris) ≈ v(france)?")
vt = v("tokyo"); vj = v("japan"); vp = v("paris"); vf = v("france")
# 首都关系: japan:tokyo :: france:paris → tokyo - japan + france ≈ paris
analogy = vj - vt + vp
analogy = analogy / (np.linalg.norm(analogy) + 1e-9)
vf_n = vf / (np.linalg.norm(vf) + 1e-9)
vp_n = vp / (np.linalg.norm(vp) + 1e-9)
print(f"  japan-tokyo+france vs paris: {float(np.dot(analogy, vp_n)):.3f}")
print(f"  japan-tokyo+france vs france: {float(np.dot(analogy, vf_n)):.3f}")
print(f"  (应 paris 更高 = 首都关系转移)")

# ── 验证 3: 巩固重放 (无外部输入, 仅内部运算) ──
print("\n[验证3] 重放巩固: 已学样本无外部输入重复更新 → 表征收敛")
from respond_learner import RespondLearner
import numpy as np
r = RespondLearner(min_sim=0.45, cap=50, seed=0)
z1 = v("the weather in tokyo is rainy"); z2 = v("the weather in shanghai is sunny")
r.learn(z1, "rainy"); r.learn(z2, "sunny")
# 重放: 不接触外部, 内部再编码+软校准 (巩固)
for _ in range(5):
    for zq, ans in [(z1, "rainy"), (z2, "sunny")]:
        txt = r.respond(zq)
        r.report_outcome(txt == ans)
print(f"  重放后条目数: {r.n()} | 通过分沉淀: {len(r.core_pairs)} "
      f"| hits/misses: {r.stats['hits']}/{r.stats['misses']}")
print("  (巩固 = 内部重放更新权重, 无外部输入 — 睡眠重放的检索式等价)")
print("=" * 62)
