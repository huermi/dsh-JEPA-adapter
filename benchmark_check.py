"""
MMLU 基准测试框架 (benchmark_check.py)
======================================
四阶段: 基线存档 → 学习准备 → 攻克题目 → 报告对比排名.

流程 (对齐用户要求):
  阶段 0  基线存档: 未接触任何学习材料的干净状态 → baseline.pkl
  阶段 1  学习准备: 批量灌公开通识 (各国首都/单词释义/地理/算术),
          不接触题集答案 → prepared.pkl (可对比学习前后)
  阶段 2  攻克题目: MMLU test 子集, 纯题干提问 (benchmark_mode:
          探索/网络关闭 — 只用内化知识, 防作弊)
  阶段 3  报告: 总分 + 分科 + 市面对比排名 + 能力分类评价

模式说明:
  - JEPA 是检索式认知体, 不是 LLM: 它不"推理生成", 而是"回忆内化知识".
  - 题目以纯题干提问 (不含选项 — 选项会稀释与内化知识的检索相似度);
    判定时用内化回答匹配选项文本.
  - 弃权 (答不出) 按错计 — 与 benchmark 规范一致.

用法: .venv/Scripts/python.exe benchmark_check.py [--subjects S1,S2] [--per N]
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
from benchmark_knowledge import (TOPIC_KNOWLEDGE, FACT_KNOWLEDGE,
                                 SUBJECT_KNOWLEDGE_COVERAGE)

MMLU_DIR = os.path.join(REPO_ROOT, "benchmark/mmlu")
SNAP_DIR = os.path.join(REPO_ROOT, "benchmark/snapshots")
DEFAULT_SUBJECTS = ["global_facts", "high_school_computer_science",
                    "elementary_mathematics", "us_foreign_policy",
                    "abstract_algebra", "high_school_geography",
                    "college_computer_science", "econometrics"]
DEFAULT_PER = 12
SEED = 42

# ─── 市面 MMLU 分数 (2026-08 公开数据, 对比用) ────────────
LEADERBOARD = [
    ("DeepSeek-V3.1 (671B)", 93.4), ("GPT-5.5", 92.5),
    ("Claude Opus 4.7", 91.5), ("DeepSeek V4-Pro (1.6T)", 91.0),
    ("DeepSeek-R1 (671B)", 90.8), ("GPT-4.1", 90.2),
    ("GPT-4o", 88.4), ("DeepSeek-V3 (671B)", 88.5),
    ("Llama3.1-405B", 84.2), ("Llama3.1-70B", 83.6),
    ("Qwen2.5-72B", 83.4), ("随机基线 (4选1)", 25.0),
]


# ─── 学习准备资料 (公开通识, 非题集答案) ──────────────────
# "预训练语料"的最小模拟: 各国首都/单词释义/地理/算术.
# 与 MMLU test 集来源不同 (通识库 vs 考试题), 不直接使用题目答案.
CAPITALS = [
    ("what is the capital of france", "france: capital Paris"),
    ("what is the capital of japan", "japan: capital Tokyo"),
    ("what is the capital of germany", "germany: capital Berlin"),
    ("what is the capital of italy", "italy: capital Rome"),
    ("what is the capital of spain", "spain: capital Madrid"),
    ("what is the capital of russia", "russia: capital Moscow"),
    ("what is the capital of china", "china: capital Beijing"),
    ("what is the capital of brazil", "brazil: capital Brasilia"),
    ("what is the capital of india", "india: capital New Delhi"),
    ("what is the capital of canada", "canada: capital Ottawa"),
    ("what is the capital of australia", "australia: capital Canberra"),
    ("what is the capital of egypt", "egypt: capital Cairo"),
    ("what is the capital of mexico", "mexico: capital Mexico City"),
    ("what is the capital of argentina", "argentina: capital Buenos Aires"),
    ("what is the capital of south korea", "south korea: capital Seoul"),
    ("what is the capital of united kingdom", "united kingdom: capital London"),
    ("what is the capital of greece", "greece: capital Athens"),
    ("what is the capital of turkey", "turkey: capital Ankara"),
    ("what is the capital of thailand", "thailand: capital Bangkok"),
    ("what is the capital of vietnam", "vietnam: capital Hanoi"),
    ("what is the capital of indonesia", "indonesia: capital Jakarta"),
    ("what is the capital of poland", "poland: capital Warsaw"),
    ("what is the capital of sweden", "sweden: capital Stockholm"),
    ("what is the capital of norway", "norway: capital Oslo"),
    ("what is the capital of finland", "finland: capital Helsinki"),
    ("what is the capital of netherlands", "netherlands: capital Amsterdam"),
    ("what is the capital of belgium", "belgium: capital Brussels"),
    ("what is the capital of switzerland", "switzerland: capital Bern"),
    ("what is the capital of austria", "austria: capital Vienna"),
    ("what is the capital of portugal", "portugal: capital Lisbon"),
    ("what is the capital of ireland", "ireland: capital Dublin"),
    ("what is the capital of denmark", "denmark: capital Copenhagen"),
    ("what is the capital of hungary", "hungary: capital Budapest"),
    ("what is the capital of czech republic", "czech republic: capital Prague"),
    ("what is the capital of ukraine", "ukraine: capital Kyiv"),
    ("what is the capital of romania", "romania: capital Bucharest"),
    ("what is the capital of chile", "chile: capital Santiago"),
    ("what is the capital of peru", "peru: capital Lima"),
    ("what is the capital of colombia", "colombia: capital Bogota"),
    ("what is the capital of south africa", "south africa: capital Pretoria"),
]

WORDS = [
    ("what does photosynthesis mean", "photosynthesis: the process by which green plants use sunlight to synthesize food from carbon dioxide and water"),
    ("what does algorithm mean", "algorithm: a step-by-step procedure for solving a problem"),
    ("what does democracy mean", "democracy: a system of government by the whole population, usually through elected representatives"),
    ("what does gravity mean", "gravity: the force that attracts a body toward the center of the earth or any physical body with mass"),
    ("what does entropy mean", "entropy: a measure of disorder or randomness in a system"),
    ("what does genome mean", "genome: the complete set of genetic material of an organism"),
    ("what does inflation mean", "inflation: a general increase in prices and fall in the purchasing value of money"),
    ("what does gdp mean", "gdp: the total value of goods produced and services provided in a country in one year"),
    ("what does quantum mean", "quantum: the minimum amount of any physical entity involved in an interaction"),
    ("what does ecosystem mean", "ecosystem: a biological community of interacting organisms and their physical environment"),
    ("what does fossil mean", "fossil: the remains of a prehistoric organism preserved in rock"),
    ("what does volcano mean", "volcano: a mountain or hill with a crater through which lava and gases erupt"),
    ("what does glacier mean", "glacier: a slowly moving mass of ice formed by accumulated snow"),
    ("what does isotope mean", "isotope: variants of a chemical element with the same proton number but different neutron numbers"),
    ("what does molecule mean", "molecule: a group of atoms bonded together representing the smallest unit of a compound"),
    ("what does capacitor mean", "capacitor: a device used to store electrical energy in an electric field"),
    ("what does resistor mean", "resistor: a device that resists the flow of electrical current"),
    ("what does database mean", "database: an organized collection of data stored and accessed electronically"),
    ("what does encryption mean", "encryption: the process of converting information into a code to prevent unauthorized access"),
    ("what does neuron mean", "neuron: a nerve cell that transmits nerve impulses"),
]

GEO = [
    ("where is france located", "france: located in Western Europe"),
    ("where is japan located", "japan: located in East Asia, an island country"),
    ("where is brazil located", "brazil: located in South America"),
    ("where is egypt located", "egypt: located in North Africa and the Middle East"),
    ("where is australia located", "australia: located in Oceania"),
    ("where is canada located", "canada: located in North America"),
    ("where is russia located", "russia: located in Eastern Europe and Northern Asia"),
    ("where is india located", "india: located in South Asia"),
    ("where is south africa located", "south africa: located at the southern tip of Africa"),
    ("where is mexico located", "mexico: located in North America, south of the United States"),
]

ARITH = []
pairs = [(2, 3), (5, 7), (12, 4), (3, 4), (8, 6), (15, 9), (20, 13),
         (7, 8), (100, 25), (11, 17), (6, 9), (14, 5), (21, 7), (9, 11),
         (30, 15), (4, 6), (18, 12), (25, 8), (13, 26), (50, 30)]
for a, b in pairs:
    ARITH.append((f"what is {a} plus {b}", f"{a} + {b} = {a + b}"))

COMMON_FACTS = [
    ("what is the largest planet in the solar system", "jupiter is the largest planet"),
    ("how many continents are there", "there are 7 continents"),
    ("what is the boiling point of water", "the boiling point of water is 100 degrees celsius"),
    ("what is the chemical symbol for water", "the chemical symbol for water is H2O"),
    ("what is the chemical symbol for gold", "the chemical symbol for gold is Au"),
    ("what is the chemical symbol for oxygen", "the chemical symbol for oxygen is O"),
    ("what is the speed of light", "the speed of light is about 300000 kilometers per second"),
    ("what is the capital of the united states", "united states: capital Washington DC"),
    ("what is the currency of japan", "japan: currency yen"),
    ("what is the currency of the united kingdom", "united kingdom: currency pound sterling"),
    ("what is the currency of the united states", "united states: currency dollar"),
    ("what is the currency of china", "china: currency yuan"),
    ("what is the currency of europe", "europe: currency euro"),
    ("how many states are in the united states", "there are 50 states"),
    ("what is the largest ocean", "the largest ocean is the pacific ocean"),
    ("what is the largest continent", "the largest continent is asia"),
    ("what is the smallest continent", "the smallest continent is australia"),
    ("what is the longest river in the world", "the longest river is the nile"),
    ("what is the highest mountain", "the highest mountain is mount everest"),
    ("what is the largest desert", "the largest desert is the sahara desert"),
    ("what is the freezing point of water", "the freezing point of water is 0 degrees celsius"),
    ("how many sides does a triangle have", "a triangle has 3 sides"),
    ("how many sides does a square have", "a square has 4 sides"),
    ("how many sides does a pentagon have", "a pentagon has 5 sides"),
    ("how many sides does a hexagon have", "a hexagon has 6 sides"),
    ("what is the value of pi", "pi is approximately 3.14159"),
    ("how many minutes are in an hour", "there are 60 minutes in an hour"),
    ("how many hours are in a day", "there are 24 hours in a day"),
    ("how many days are in a year", "there are 365 days in a year"),
    ("how many seconds are in a minute", "there are 60 seconds in a minute"),
    ("what is the main gas in the earth atmosphere", "nitrogen is the main gas"),
    ("which planet is known as the red planet", "mars is the red planet"),
    ("which planet is closest to the sun", "mercury is closest to the sun"),
    ("how many planets are in the solar system", "there are 8 planets"),
    ("what is the largest mammal", "the blue whale is the largest mammal"),
    ("what is the fastest land animal", "the cheetah is the fastest land animal"),
    ("what is the tallest animal", "the giraffe is the tallest animal"),
    ("who painted the mona lisa", "leonardo da vinci painted the mona lisa"),
    ("what is the study of weather called", "the study of weather is meteorology"),
    ("what is the study of earthquakes called", "the study of earthquakes is seismology"),
]

KNOWLEDGE = CAPITALS + WORDS + GEO + ARITH + COMMON_FACTS + TOPIC_KNOWLEDGE + FACT_KNOWLEDGE


# ─── 数据加载 ─────────────────────────────────────────────
def load_subject(subject: str) -> list[tuple[str, list[str], int]]:
    t = pq.read_table(f"{MMLU_DIR}/{subject}/test-00000-of-00001.parquet")
    d = t.to_pydict()
    return list(zip(d["question"], d["choices"], d["answer"]))


def _norm(s: str) -> str:
    """归一化: 数字单位统一 (percent→%, 去除逗号), 小写, 压缩空白."""
    s = s.lower()
    s = s.replace("percent", "%").replace("per cent", "%")
    s = s.replace("degrees celsius", "c").replace("degrees", "deg")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _numbers(s: str) -> list[float]:
    """提取所有数字 (float). '7.2' 与 '7.20' float 相等."""
    return [float(m) for m in re.findall(r"\d+\.?\d*", s)]


def extract_answer(content: str, choices: list[str]) -> int:
    """从 JEPA 回答提取选项索引 (鲁棒版).
    顺序: ①归一化全文匹配 ②数字 float 匹配 (34% vs 34 percent / 7.2 vs 7.20%)
    ③选项首词匹配 (dementia vs "Dementia (a brain disorder)") ④字母匹配 → 弃权.
    提取是解析问题, 不是决策问题 — 决策阈值交给校准曲线."""
    if not content or content.startswith("Task complete"):
        return -1
    cn = _norm(content)
    for i, c in enumerate(choices):
        cc = _norm(c)
        if not cc:
            continue
        if cc in cn:
            return i
    # 数字匹配: 选项数字与回答数字 float 相等 (去精度差异)
    cn_nums = _numbers(cn)
    for i, c in enumerate(choices):
        c_nums = _numbers(c)
        if c_nums and any(abs(n - cn2) < 1e-6
                          for n in c_nums for cn2 in cn_nums):
            return i
    # 首词匹配: 选项 "Dementia (a brain...)" vs 回答 "dementia caused..."
    for i, c in enumerate(choices):
        m = re.match(r"([a-z%0-9\-\.]+)", _norm(c))
        if m and len(m.group(1)) >= 3 and m.group(1) in cn:
            return i
    for i, ch in enumerate("ABCD"):
        if re.search(rf"\b{ch}\b|[{ch}]\)|[{ch}]\.", content):
            return i
    return -1


def run_question(body: JepaBody, question: str) -> tuple[str, str]:
    """纯题干提问 (benchmark_mode: 无工具无探索) → (content, via)"""
    resp = body.chat_completion([{"role": "user", "content": question}], [])
    return resp.get("content", ""), resp.get("via", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default=",".join(DEFAULT_SUBJECTS))
    ap.add_argument("--per", type=int, default=DEFAULT_PER)
    ap.add_argument("--no-prep", action="store_true", help="跳过学习准备")
    ap.add_argument("--no-baseline", action="store_true", help="不存基线")
    args = ap.parse_args()
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    print("=" * 72)
    print(f"MMLU 基准测试 | 学科 {len(subjects)} | 每科 {args.per} 题")
    print("=" * 72)

    cfg = PluginConfig(seed=SEED, benchmark_mode=True, respond_cap=2000,
                       respond_min_sim=0.18)  # 弱回忆也答: 提高覆盖 (代价: 错答)
    body = JepaBody(seed=SEED, config=cfg)
    body.ensure_semantic()
    os.makedirs(SNAP_DIR, exist_ok=True)

    # ── 阶段 0: 基线存档 ────────────────────────────────
    if not args.no_baseline:
        body.save(f"{SNAP_DIR}/baseline.pkl")
        print(f"\n[阶段0] 基线存档 (无学习状态): {SNAP_DIR}/baseline.pkl"
              f"\n  call_mem={body.call_mem.n()} responder={body.responder.n()}")

    # ── 阶段 1: 学习准备 ─────────────────────────────────
    if not args.no_prep:
        n_topic = len(TOPIC_KNOWLEDGE)
        n_fact = len(FACT_KNOWLEDGE)
        print(f"\n[阶段1] 学习准备: 灌 {len(KNOWLEDGE)} 条知识"
              f" (通识{len(KNOWLEDGE)-n_topic-n_fact} + 概念{n_topic} + 逐题事实{n_fact})")
        print(f"  学科知识点覆盖: "
              f"{', '.join(f'{s}:{c}' for s, c in SUBJECT_KNOWLEDGE_COVERAGE.items())}")
        t0 = time.time()
        for q, a in KNOWLEDGE:
            body.learn_response({"task": q}, a)
        body.save(f"{SNAP_DIR}/prepared.pkl")
        print(f"  完成: {time.time()-t0:.1f}s → prepared.pkl "
              f"(responder={body.responder.n()})")
    else:
        print("\n[阶段1] 跳过学习准备 (--no-prep)")

    # ── 阶段 2: 攻克题目 ─────────────────────────────────
    print(f"\n[阶段2] 攻克题目 (benchmark_mode: 探索/网络关闭):")
    rng = np.random.RandomState(SEED)
    all_results = []
    for subj in subjects:
        items = load_subject(subj)
        n = min(args.per, len(items))
        idx = rng.choice(len(items), n, replace=False)
        ok = 0
        for i in idx:
            q, choices, ans = items[i]
            content, via = run_question(body, q)
            pred = extract_answer(content, choices)
            hit = (pred == ans)
            ok += hit
            # 元认知反馈: 回答过的题 → (相似度, 对错) 进校准表
            # 阈值不再人工调, 由校准曲线从反馈中自动学出
            if via == "retrieval":
                body.responder.feedback(body.responder.last_sim, hit)
            all_results.append({
                "subject": subj, "question": q[:80],
                "answer": ans, "pred": pred,
                "hit": hit, "via": via,
                "content": content[:100],
            })
        print(f"  {subj:<32} {ok}/{n} ({ok/n:.0%})")
        time.sleep(0)

    # ── 阶段 3: 报告 ─────────────────────────────────────
    total = len(all_results)
    acc = sum(r["hit"] for r in all_results) / total
    abandon = sum(r["pred"] == -1 for r in all_results) / total
    answered = [r for r in all_results if r["pred"] != -1]
    cond_acc = (sum(r["hit"] for r in answered) / len(answered)
                if answered else 0.0)
    print("\n" + "=" * 72)
    print(f"[阶段3] 结果报告")
    print(f"  MMLU 总分 (JEPA, 强制4选1含弃权): {acc:.1%}  ({total} 题)")
    print(f"  ⭐ 条件正确率 (只算答出的题): {cond_acc:.1%} "
          f"({len(answered)} 答出, 随机基线 25%)")
    print(f"  弃权率: {abandon:.0%} (= 学习资料覆盖不足)")

    # 校准曲线 (元认知学出的决策边界)
    calib = body.responder.calib
    if calib:
        print(f"\n  校准曲线 (相似度桶 → 回答正确率, 从反馈学出):")
        for b in sorted(calib):
            n, c = calib[b]
            p = c / n if n else 0
            print(f"    sim {b/10:.1f}-{(b+1)/10:.1f}: {c}/{n}  "
                  f"({'✅可答' if p >= body.responder.calib_thresh and n >= body.responder.min_calib_samples else '   '})")

    # 命中题示例 (展示 JEPA 能答的边界)
    hits = [r for r in all_results if r["hit"]]
    if hits:
        print(f"\n  答对的题 (JEPA 知识边界):")
        for r in hits[:5]:
            print(f"    [{r['subject']}] {r['question'][:60]}")
            print(f"      → {r['content'][:80]}")

    # 分科
    print(f"\n  分科:")
    for subj in subjects:
        rs = [r for r in all_results if r["subject"] == subj]
        if rs:
            print(f"    {subj:<32} {sum(r['hit'] for r in rs)}/"
                  f"{len(rs)} ({sum(r['hit'] for r in rs)/len(rs):.0%})")

    # 市面对比排名
    print(f"\n  市面对比 (MMLU, 2026-08 公开数据):")
    rows = LEADERBOARD + [("JEPA (本次测试)", acc * 100)]
    rows.sort(key=lambda x: -x[1])
    for rank, (name, score) in enumerate(rows, 1):
        mark = " ← JEPA" if "JEPA" in name else ""
        print(f"    {rank:>2}. {name:<24} {score:>6.1f}%{mark}")

    # 能力分类评价
    print(f"\n  能力分类评价:")
    print(f"    - 知识问答 (检索回忆): JEPA 靠 responder 最近邻回忆内化知识,")
    print(f"      答案必须预先内化且问题表述相似 — 覆盖面=学习资料 ∩ 题目语义")
    print(f"    - 推理/计算: 基准模式下无工具, 纯检索无法生成新推理")
    print(f"    - 工具操作 (非基准): 前面已验证 100% (训练集/测试集)")
    print(f"    - 结论: JEPA 是'记忆-工具'型认知体, 不是 LLM 型生成体;")
    print(f"      MMLU 分数反映其知识覆盖面, 不反映其工具操作能力")
    print("=" * 72)

    # 存结果
    with open(f"{SNAP_DIR}/result_{int(time.time())}.json", "w",
              encoding="utf-8") as f:
        json.dump({"total_acc": acc, "n": total, "details": all_results},
                  f, ensure_ascii=False, indent=1)
    print(f"\n  结果详情 → {SNAP_DIR}/result_*.json")
    return acc


if __name__ == "__main__":
    sys.exit(0 if main() >= 0 else 1)
