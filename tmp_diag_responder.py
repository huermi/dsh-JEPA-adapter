"""诊断: 200 条知识内化后, responder 对题目的检索行为"""
import sys
import numpy as np

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "body"))

from kernel import JepaBody
from plugin_config import PluginConfig
from benchmark_check import KNOWLEDGE

body = JepaBody(seed=42, config=PluginConfig(seed=42, benchmark_mode=True,
                                             respond_cap=2000,
                                             respond_min_sim=0.28))
body.ensure_semantic()
for q, a in KNOWLEDGE:
    body.learn_response({"task": q}, a)
print(f"responder pairs: {body.responder.n()}")

# 手动检索: 看每题 top-3 相似条目
test_questions = [
    "Which one of the following groups is excluded from the caste system of Hinduism?",
    "In Python 3, which of the following function sets the integer starting value used in generating random numbers?",
    "What is the value of p in 24 = 2p?",
    "Which of the following religions developed first?",
    "What is the mean of the scores 76, 80, 83, 71, 80, and 78?",
]
for q in test_questions:
    z = body._situation_vec(q, "", "")
    zn = z / (np.linalg.norm(z) + 1e-9)
    scored = []
    for pz, text in body.responder.pairs:
        pzn = np.asarray(pz, np.float32)
        pzn = pzn / (np.linalg.norm(pzn) + 1e-9)
        scored.append((float(np.dot(zn, pzn)), text[:60]))
    scored.sort(key=lambda x: -x[0])
    print(f"\nQ: {q[:60]}")
    for s, t in scored[:3]:
        print(f"   {s:.3f}  {t}")
    # 实际走 chat_completion
    resp = body.chat_completion([{"role": "user", "content": q}], [])
    print(f"   via={resp.get('via')} content={resp.get('content','')[:60]}")
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
