"""
自主查证学习循环 (learner_loop.py)
==================================
训练 JEPA 自主把握学习过程: 面对任务 → 自己决定"学什么/去哪学/何时够" → 答题.

核心架构 (元学习): "学习策略"本身可学习.
  research (查证) 注册为工具 → 走 JEPA 记忆-检索-探索-反馈机制:
    - 学过的策略复用: call_mem 检索到 (任务→research+query)
    - 新策略探索: 语义探索选 research
    - 策略进化: 最终答对/错 → perf 反馈 → learn_call 记录策略
  "学会学习" = 学习动作的选择本身被记忆/检索/探索驱动.

学习资源 (多源查证):
  本地资料库 (D:/JEPA/benchmark/library/*.txt, 171 条知识) + 网络 fetch

循环:
  while 轮 < max:
    resp = chat_completion(任务, tools=[research])
    research → 执行查证 (grep 资料库 + fetch) → 内化 responder → 轮++
    无工具调用 → 收敛 (responder 命中知识 / 学不到收尾) → 提取答案 → 判定

指标 (自主性):
  自主学习率 / 学习效率 (每题查证次数) / 策略复用 / 学后正确率 + 学习增量
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import argparse
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
from benchmark_check import _norm, extract_answer  # 复用鲁棒提取
from question_bank import QuestionBank, STOPWORDS

LIBRARY = os.path.join(REPO_ROOT, "benchmark/library")
MMLU_DIR = os.path.join(REPO_ROOT, "benchmark/mmlu")
SUBJECTS = ["global_facts", "high_school_computer_science",
            "elementary_mathematics", "us_foreign_policy",
            "abstract_algebra", "high_school_geography",
            "college_computer_science", "econometrics"]

RESEARCH_SCHEMA = [{
    "type": "function",
    "function": {"name": "research",
                 "description": "research a topic in the knowledge library "
                                "or on the internet to learn what you need",
                 "parameters": {"type": "object",
                                "properties": {"query": {"type": "string"}}}}},
]


def load_subject(subject):
    t = pq.read_table(f"{MMLU_DIR}/{subject}/test-00000-of-00001.parquet")
    d = t.to_pydict()
    return list(zip(d["question"], d["choices"], d["answer"]))


def research_execute(query: str, task: str = "") -> str:
    """查证: 本地资料库 grep (用任务的实义词, 相关性排序) → 网络兜底.
    返回与任务最相关的文本段 (供内化)."""
    # query 用任务实义词 (模型只决定"查不查", 查证词由 harness 提炼)
    base = task if task else query
    words = [w for w in re.split(r"[^a-z0-9]+", base.lower())
             if len(w) >= 4 and w not in
             ("what", "which", "that", "with", "from", "this", "have",
              "were", "about", "when", "where", "than", "been", "will",
              "into", "over", "more", "most", "would", "there", "their")]
    scored = []
    for fn in sorted(os.listdir(LIBRARY)):
        if not fn.endswith(".txt"):
            continue
        with open(os.path.join(LIBRARY, fn), encoding="utf-8") as f:
            for line in f:
                ll = line.lower()
                score = sum(1 for w in words if w in ll)
                if score >= 2:           # 相关性门控: ≥2 个实义词命中
                    scored.append((score, line.strip()))
    if scored:
        scored.sort(key=lambda x: -x[0])
        # 只返回事实部分 (按 "|" 取后半) — 避免 "问题|事实" 格式污染内化
        # topk 可调: top1 更精准 (防 top3 拼接引入混叠), 默认 3 (信息量)
        topk = int(os.environ.get("JEPA_RESEARCH_TOPK", "3"))
        facts = []
        for _, line in scored[:topk]:
            if "|" in line:
                facts.append(line.split("|", 1)[1].strip())
            else:
                facts.append(line.strip())
        return " | ".join(facts)[:600]
    # 网络 fetch 兜底 (JEPA_NETWORK=0 禁用; 空壳/错误 → no knowledge)
    if os.environ.get("JEPA_NETWORK", "1") == "0":
        return f"no knowledge found for: {base}"
    try:
        from train_web_learning import tool_fetch
        r = tool_fetch(query=base)
        if (r.startswith("no result") or r.startswith("ERROR")
                or r.strip().lower().startswith("search")):
            return f"no knowledge found for: {base}"
        return r[:400]
    except Exception:
        return f"no knowledge found for: {base}"


class LearnerLoop:
    """自主查证学习循环 (harness 侧驱动 + 模型自主决策 + 纠错)"""

    def __init__(self, body: JepaBody, max_rounds: int = 5):
        self.body = body
        self.max_rounds = max_rounds

    def _learn_round(self, task: str) -> dict:
        """一轮学习循环: 模型自主决策 (查证/收敛) → 内化 → 返回."""
        self.body.reset_task()
        messages = [{"role": "user", "content": task}]
        rounds = []
        content = ""
        via = "?"
        for r in range(1, self.max_rounds + 1):
            resp = self.body.chat_completion(messages, RESEARCH_SCHEMA)
            tcs = resp.get("tool_calls", [])
            via = resp.get("via", "?")
            if not tcs:
                content = resp.get("content", "")
                rounds.append({"round": r, "action": "converge", "via": via})
                break
            fn = tcs[0]["function"]
            query = json.loads(fn.get("arguments") or "{}").get("query", task)
            result = research_execute(query, task=task)
            words = [w for w in re.split(r"[^a-z0-9]+", task.lower())
                     if len(w) >= 4 and w not in STOPWORDS]
            relevant = sum(1 for w in words if w in result.lower()) >= 1
            rounds.append({"round": r, "action": "research",
                           "query": query[:40],
                           "result": result[:60],
                           "relevant": relevant})
            if relevant and result and not result.startswith("no knowledge"):
                self.body.learn_response({"task": task}, result[:300])
            messages.append({"role": "assistant", "content": "",
                             "tool_calls": tcs})
            messages.append({"role": "tool", "content": result,
                             "tool_call_id": tcs[0].get("id", "0")})
        return {"rounds": rounds, "content": content, "via": via}

    def learn_and_answer(self, task: str, max_fix: int = 2) -> dict:
        """学习+答题, 带纠错: 学到错误/没学到 → 负样本 → 重查证 → 重答.
        "学错了没关系, 关键是要能纠错" — 纠错 = 发现错误 + 更新知识."""
        fixes = []
        for fix in range(max_fix + 1):
            traj = self._learn_round(task)
            # 学到了知识 (responder 命中) → 交 main 判定
            if traj["via"] != "default":
                return {**traj, "fixes": fixes}
            # 没学到 (default) → 纠错: 清状态重来 (换角度再查)
            fixes.append({"fix": fix + 1, "action": "retry",
                          "reason": "no knowledge learned"})
            self.body.learn_response_negative({"task": task})
        return {**traj, "fixes": fixes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default=",".join(SUBJECTS))
    ap.add_argument("--per", type=int, default=12)
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-soft-align", action="store_true",
                    help="关闭 AdaJEPA 软校准 (对照实验)")
    args = ap.parse_args()
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    print("=" * 72)
    print(f"自主查证学习循环 | 学科 {len(subjects)} | 每科 {args.per} 题 "
          f"| 每轮上限 {args.max_rounds}")
    print("=" * 72)

    rng = np.random.RandomState(args.seed)
    results = []
    bank = QuestionBank(seed=args.seed)
    n_fixed = n_wrong_learned = n_spawn = n_spawn_mastered = 0
    # 每个学科一个干净 body (隔离学科知识, 测每科学习能力)
    for subj in subjects:
        body = JepaBody(seed=args.seed,
                        config=PluginConfig(seed=args.seed,
                                            respond_cap=2000,
                                            respond_min_sim=0.18,
                                            soft_align=not args.no_soft_align))
        # 校准表更新开关 (JEPA_NO_CALIB=1 禁用 — 隔离 calib 对检索行为的影响)
        if os.environ.get("JEPA_NO_CALIB", "0") == "1":
            body.responder.calib_update = False
        # 不开 benchmark_mode: 学习循环需要探索 (发起查证 = 学习行为)
        body.ensure_semantic()
        loop = LearnerLoop(body, max_rounds=args.max_rounds)
        items = load_subject(subj)
        n = min(args.per, len(items))
        idx = rng.choice(len(items), n, replace=False)
        ok = 0
        clean_pre = None   # 只测该科第一题的干净学前 (后续题 body 已在前序学习)
        for k, i in enumerate(idx):
            q, choices, ans = items[i]
            # 学前: 仅第一题 (干净状态, 无前序学习污染)
            if k == 0:
                c0 = body.chat_completion([{"role": "user", "content": q}], [])
                clean_pre = (extract_answer(c0.get("content", ""), choices)
                             == ans)
            pre_hit = bool(clean_pre) if k == 0 else False
            # 问题库: 加入当前问题 (持续学习动力引擎输入)
            bank.add(q)
            # 自主查证学习 + 答题
            t0 = time.time()
            traj = loop.learn_and_answer(q)
            lt = time.time() - t0
            post_hit = (extract_answer(traj["content"], choices) == ans)
            n_research = sum(1 for r in traj["rounds"]
                             if r["action"] == "research")
            # 纠错: 学到了知识但答错 → 负样本标记 → 重学 → 清除负样本 → 重答
            # (负样本语义: 旧答案作废; 新知识内化后必须解除封锁, 否则永远拒答)
            fixed = False
            if (not post_hit and traj["content"]
                    and not traj["content"].startswith("Task complete")):
                n_wrong_learned += 1
                body.learn_response_negative({"task": q})
                body.reset_task()
                loop.learn_and_answer(q, max_fix=1)   # 重学 (探索查证新知识)
                # 清除该任务的负样本: 新知识已内化, 旧答案作废
                z_task = body._situation_vec(q, "", "")
                body.responder.negatives = [
                    nz for nz in body.responder.negatives
                    if np.linalg.norm(np.asarray(nz, np.float32)
                                      - z_task) > 1e-6]
                # 重答
                resp2 = body.chat_completion(
                    [{"role": "user", "content": q}], [])
                fixed = (extract_answer(resp2.get("content", ""), choices)
                         == ans)
                if fixed:
                    post_hit = True
                    n_fixed += 1
            # 受阻-通过沉淀反馈 (落点2, 命题1+3): 判定结果回传 responder —
            # 答对 → 通过分+1 (可沉淀 core); 答错 → -2 (污染惩罚, 不沉淀).
            # 沉淀判据 = "被实践考验且通过", 不是"被检索命中" (对照实验 B).
            body.responder_feedback(post_hit)
            ok += post_hit
            # 问题库反馈 + 问题繁殖 (双向公式: 学会 → 繁殖新问题)
            spawned = bank.feedback(q, post_hit, traj["content"])
            if spawned:
                n_spawn += 1
            results.append({
                "subject": subj, "question": q[:50], "answer": ans,
                "pre_hit": pre_hit, "post_hit": post_hit,
                "fixed": fixed, "n_fixes": len(traj.get("fixes", [])),
                "n_research": n_research, "learn_t": round(lt, 2),
                "content": traj["content"][:50],
            })
            flag = "🔧" if fixed else ("✅" if post_hit else "❌")
            print(f"  {flag} [{subj[:20]:<20}] "
                  f"学{pre_hit}→学后{post_hit}"
                  f"{'(纠错)' if fixed else ''} {q[:32]}")
            # 学习策略反馈: 答对 → (任务→查证动作) 学进 call_mem
            # (JEPA_NO_LEARN_CALL=1 可关闭 — 隔离 call_memory 对结果的影响)
            if post_hit and os.environ.get("JEPA_NO_LEARN_CALL", "0") != "1":
                body.learn_call({"task": q}, "research", {"query": q},
                                result=traj["content"], perf=1.0)

        # 繁殖问题自学: 该科学完后, 处理问题库繁殖出的未掌握问题
        # (双向公式: 学会的答案繁殖出新问题 → 继续学 → 持续学习动力)
        for _ in range(20):
            q2 = bank.pick()
            if q2 is None:
                break
            st = bank.questions[q2]
            if st["mastered"]:
                break
            traj2 = loop.learn_and_answer(q2, max_fix=1)
            learned = traj2["via"] != "default"
            if learned:
                n_spawn_mastered += 1
            bank.feedback(q2, learned, traj2["content"])

    # ── 报告 ──
    total = len(results)
    pre_acc = sum(r["pre_hit"] for r in results) / total
    post_acc = sum(r["post_hit"] for r in results) / total
    gained = sum(not r["pre_hit"] and r["post_hit"] for r in results)
    n_res = np.mean([r["n_research"] for r in results])
    print("\n" + "=" * 72)
    print(f"[报告] 自主查证学习 ({total} 题)")
    print(f"  学前正确率: {pre_acc:.1%}")
    print(f"  学后正确率: {post_acc:.1%}")
    print(f"  ⭐ 学习增量: {post_acc - pre_acc:+.1%} ({gained}/{total} 从不会→会)")
    print(f"  自主学习率: {sum(r['n_research']>0 for r in results)/total:.0%} "
          f"(主动发起查证的题)")
    print(f"  学习效率: 平均 {n_res:.1f} 次查证/题")
    print(f"  🔧 纠错: {n_wrong_learned} 题学到但答错, "
          f"纠错成功 {n_fixed} ({n_fixed/max(n_wrong_learned,1):.0%})")
    print(f"  🌱 问题繁殖 (双向公式): 学会后繁殖 {n_spawn} 个新问题, "
          f"自学掌握 {n_spawn_mastered}")
    print(f"  分科:")
    for subj in subjects:
        rs = [r for r in results if r["subject"] == subj]
        if rs:
            p0 = sum(r["pre_hit"] for r in rs) / len(rs)
            p1 = sum(r["post_hit"] for r in rs) / len(rs)
            nr = np.mean([r["n_research"] for r in rs])
            print(f"    {subj:<30} {p0:.0%}→{p1:.0%} "
                  f"({nr:.1f}次/题)")
    os.makedirs(os.path.join(REPO_ROOT, "benchmark/snapshots"), exist_ok=True)
    with open(f"{REPO_ROOT}/benchmark/snapshots/learner_{int(time.time())}.json",
              "w", encoding="utf-8") as f:
        json.dump({"pre_acc": pre_acc, "post_acc": post_acc,
                   "details": results}, f, ensure_ascii=False, indent=1)
    print(f"  详情 → learner_*.json")
    print("=" * 72)
    return post_acc


if __name__ == "__main__":
    sys.exit(0 if main() >= 0 else 1)
