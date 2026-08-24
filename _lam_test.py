"""LAM 接入验证: 真实 MiniLM, 教材内化 → 未见变体 W 兜底."""
import sys
import numpy as np
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "body"))
from mini_encoder import get_encoder
from respond_learner import RespondLearner

enc = get_encoder()
enc.ensure_loaded()
r = RespondLearner(min_sim=0.45, cap=200, seed=0)

# 教材: 同模板不同实体 (W 学功能映射)
pairs = [
    ("the weather in tokyo is rainy", "rainy"),
    ("the weather in shanghai is sunny", "sunny"),
    ("the capital of japan is tokyo", "tokyo"),
    ("the capital of france is paris", "paris"),
]
for q, a in pairs:
    r.learn(enc.encode(q), a)

print(f"W: {r.W.shape if r.W is not None else None} | "
      f"answer_vecs: {len(r.answer_vecs)} | 条目: {r.n()}")

# 未见变体: 检索可能 miss (同模板新实体) → W 兜底
tests = [
    ("the weather in tokyo is rainy today", "rainy", "同模板变体"),
    ("the weather in shanghai is sunny today", "sunny", "同模板变体"),
    ("the weather in beijing is rainy", "rainy", "新实体变体(未学)"),
    ("the capital of japan is tokyo city", "tokyo", "同模板变体"),
    ("the capital of france is paris city", "paris", "同模板变体"),
    ("what is the population of mars", None, "无关查询"),
]
for q, expect, tag in tests:
    out = r.respond(enc.encode(q), task_text=q)
    hit = "✅" if (expect is None and out is None) or (expect and out == expect) else "❌"
    print(f"  {hit} [{tag}] {q[:38]} → {out!r} (应 {expect})")
print(f"lam_hits: {r.stats.get('lam_hits', 0)}")
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
