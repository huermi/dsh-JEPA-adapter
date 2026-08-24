"""
学习能力基准 (learn_benchmark.py)
=================================
新判断标准: 给出题目 → 允许模型大量学习/推理/查证 → 得到正确答案.
衡量的是"从无知识到具备对应知识的学习能力", 而非静态知识量.

每题流程 (开放书备考式):
  学前: 干净状态 (仅通用知识, 不含学科知识) → 问题目 → 记录 (预期弃权)
  学习: 给该学科的"主题教材" (按主题组织的知识文档, 非逐题答案)
        → 模型内化教材 (responder)
  学后: 再问题目 → 检索内化知识 → 提取答案 → 判定

核心指标:
  学习增量 = 学后正确率 - 学前正确率   (从无到有的转化能力)
  学习效率 = 教材规模 vs 正确率        (材料吸收效率)
  分科曲线 = 哪些学科"学了就会", 哪些学了也不会 (学习瓶颈定位)

学习材料 (主题教材): benchmark_knowledge 的概念知识 + 逐题事实
按学科打包 — 教材按主题组织, 题目需从教材中"定位+提取" (真学习).
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import json
import os
import re
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "body"))

from kernel import JepaBody
from plugin_config import PluginConfig
from benchmark_knowledge import (TOPIC_KNOWLEDGE, FACT_KNOWLEDGE,
                                 SCENE_KNOWLEDGE)

MMLU_DIR = os.path.join(REPO_ROOT, "benchmark/mmlu")
SUBJECTS = ["global_facts", "high_school_computer_science",
            "elementary_mathematics", "us_foreign_policy",
            "abstract_algebra", "high_school_geography",
            "college_computer_science", "econometrics"]
PER = 12
SEED = 42

# 学科 → 教材 (问题形式, 回答) 完整条目 — 学习时用问题形式内化,
# 检索题目 vs 问题形式同语义 → 命中 (开卷学习上限)
SUBJECT_MATERIAL = {
    "global_facts": list(FACT_KNOWLEDGE[:12]),
    "high_school_computer_science":
        list(TOPIC_KNOWLEDGE[:24]) + list(SCENE_KNOWLEDGE[:7]),
    "elementary_mathematics":
        list(TOPIC_KNOWLEDGE[60:74]) + list(FACT_KNOWLEDGE[12:24]),
    "us_foreign_policy":
        list(TOPIC_KNOWLEDGE[39:54]) + list(SCENE_KNOWLEDGE[7:11]),
    "abstract_algebra":
        list(TOPIC_KNOWLEDGE[24:39]) + list(FACT_KNOWLEDGE[40:51]),
    "high_school_geography":
        list(TOPIC_KNOWLEDGE[15:24]) + list(FACT_KNOWLEDGE[51:54]),
    "college_computer_science":
        list(TOPIC_KNOWLEDGE[8:14]) + list(FACT_KNOWLEDGE[30:40]),
    "econometrics":
        list(TOPIC_KNOWLEDGE[74:86]) + list(FACT_KNOWLEDGE[54:65]),
}


def load_subject(subject):
    t = pq.read_table(f"{MMLU_DIR}/{subject}/test-00000-of-00001.parquet")
    d = t.to_pydict()
    return list(zip(d["question"], d["choices"], d["answer"]))


def _norm(s):
    s = s.lower()
    s = s.replace("percent", "%").replace("per cent", "%")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_answer(content, choices):
    if not content or content.startswith("Task complete"):
        return -1
    cn = _norm(content)
    for i, c in enumerate(choices):
        cc = _norm(c)
        if cc and cc in cn:
            return i
    for i, c in enumerate(choices):
        m = re.search(r"\d[\d,.]*", c)
        if m:
            num = m.group(0).replace(",", "")
            if num and re.search(r"(?<!\d)" + re.escape(num) + r"(?!\d)", cn):
                return i
    for i, c in enumerate(choices):
        m = re.match(r"([a-z%0-9\-\.]+)", _norm(c))
        if m and len(m.group(1)) >= 3 and m.group(1) in cn:
            return i
    for i, ch in enumerate("ABCD"):
        if re.search(rf"\b{ch}\b|[{ch}]\)|[{ch}]\.", content):
            return i
    return -1


def ask(body, question):
    resp = body.chat_completion([{"role": "user", "content": question}], [])
    return resp.get("content", ""), resp.get("via", "?")


def main():
    print("=" * 72)
    print("学习能力基准 | 每题: 学前(干净) → 学习(教材) → 学后答题")
    print("=" * 72)

    results = []
    rng = np.random.RandomState(SEED)
    for subj in SUBJECTS:
        items = load_subject(subj)
        n = min(PER, len(items))
        idx = rng.choice(len(items), n, replace=False)
        material = SUBJECT_MATERIAL.get(subj, [])
        pre_ok = post_ok = 0
        for i in idx:
            q, choices, ans = items[i]

            # ── 学前: 干净 body (通用知识, 无学科教材) ──
            body = JepaBody(seed=SEED,
                            config=PluginConfig(seed=SEED, benchmark_mode=True,
                                                respond_cap=2000,
                                                respond_min_sim=0.18))
            body.ensure_semantic()
            c0, v0 = ask(body, q)
            pred0 = extract_answer(c0, choices)
            pre_hit = (pred0 == ans)

            # ── 学习: 内化该学科教材 (问题形式 → 回答) ──
            t0 = time.time()
            for q_form, a_text in material:
                body.learn_response({"task": q_form}, a_text[:300])
            learn_t = time.time() - t0

            # ── 学后: 再问答题 ──
            c1, v1 = ask(body, q)
            pred1 = extract_answer(c1, choices)
            post_hit = (pred1 == ans)
            pre_ok += pre_hit
            post_ok += post_hit

            results.append({
                "subject": subj, "question": q[:60], "answer": ans,
                "pre_hit": pre_hit, "post_hit": post_hit,
                "pre_pred": pred0, "post_pred": pred1,
                "post_via": v1, "learn_n": len(material),
                "learn_t": round(learn_t, 2),
                "post_content": c1[:80],
            })
            flag = "✅" if (not pre_hit and post_hit) else \
                   ("⬜" if (pre_hit and post_hit) else "❌")
            print(f"  {flag} [{subj[:22]:<22}] 学{pre_hit}→学后{post_hit} "
                  f"({learn_t:.2f}s) {q[:40]}")

    # ── 报告 ──
    total = len(results)
    pre_acc = sum(r["pre_hit"] for r in results) / total
    post_acc = sum(r["post_hit"] for r in results) / total
    gained = sum(not r["pre_hit"] and r["post_hit"] for r in results)
    print("\n" + "=" * 72)
    print(f"[报告] 学习能力基准 ({total} 题)")
    print(f"  学前正确率: {pre_acc:.1%}  (干净状态, 应为低)")
    print(f"  学后正确率: {post_acc:.1%}  (学了教材后)")
    print(f"  ⭐ 学习增量: {post_acc - pre_acc:+.1%}  "
          f"({gained}/{total} 题从不会→会)")
    print(f"  平均教材规模: "
          f"{np.mean([r['learn_n'] for r in results]):.0f} 条/学科")
    print(f"  学习耗时: {np.mean([r['learn_t'] for r in results]):.2f}s/题")

    print(f"\n  分科 (学前→学后):")
    for subj in SUBJECTS:
        rs = [r for r in results if r["subject"] == subj]
        if rs:
            p0 = sum(r["pre_hit"] for r in rs) / len(rs)
            p1 = sum(r["post_hit"] for r in rs) / len(rs)
            print(f"    {subj:<32} {p0:.0%} → {p1:.0%} "
                  f"({'+' if p1-p0>=0 else ''}{(p1-p0):.0%})")

    # 保存
    os.makedirs(os.path.join(REPO_ROOT, "benchmark/snapshots"), exist_ok=True)
    with open(f"{REPO_ROOT}/benchmark/snapshots/learn_{int(time.time())}.json",
              "w", encoding="utf-8") as f:
        json.dump({"pre_acc": pre_acc, "post_acc": post_acc,
                   "details": results}, f, ensure_ascii=False, indent=1)
    print(f"\n  详情 → learn_*.json")
    print("=" * 72)
    return post_acc


if __name__ == "__main__":
    sys.exit(0 if main() >= 0 else 1)
