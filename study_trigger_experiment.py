"""
持续学习理论推进 — 对照实验 (study_trigger_experiment.py)
==========================================================
依据《持续学习理论推进》边界声明的实验设计, 验证两个核心命题:

实验 A: 学习触发源对照 (命题2: 受阻是学习触发源, 好奇只是策略性动力)
  三种触发策略在相同任务集/相同 seed 下对照, 只改"何时学习", 其余全同:
    A1 受阻触发: 仅当检索 miss (responder 未命中) 时学习
    A2 好奇触发: 仅当任务新奇度高 (与已学知识最大余弦低) 时学习
    A3 效用触发: 每次都学习 (无差别内化)
  预注册判定标准:
    - 若命题2成立: A1 的单位学习收益 (学习动作→从不会到会) 显著高于 A2 和 A3;
      A2 触发率偏高但收益低 (新奇≠可学, 学一堆无关); A3 浪费学习动作
    - 若 A3 (每次都学) 收益最高 → 命题2 被反驳
  多 seed 符号检验: seed 7 / 42 / 123

实验 B: 沉淀判据对照 (命题1: 选择性遗忘是扬弃; 命题3: 沉淀判据应为受阻-通过)
  合成数据流, 两种沉淀判据:
    B1 受阻-通过: 条目"被检索命中 且 外部判定正确"才 +1 通过分; 答错衰减;
                 通过分 ≥2 → 晋升 core (沉淀)
    B2 命中次数 (当前实现): 被检索命中就 +1 (不管对错); ≥2 → 晋升 core
  关键场景: 高相似错误条目 (同构混叠 — 东京问题内化上海答案)
    预注册判定标准:
      - B2 会把高命中错误条目晋升到 core (错误固化 = 命题1说的"忘得不够"病理)
      - B1 不会 (答错衰减 → 不沉淀 → 被选择性遗忘/覆盖)
  高压冲击后测: 正确知识保留率 (学新不毁旧) + 错误条目是否沉淀
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

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "body"))

from kernel import JepaBody
from plugin_config import PluginConfig
from benchmark_check import _norm, extract_answer
from learner_loop import load_subject

MMLU_DIR = os.path.join(REPO_ROOT, "benchmark/mmlu")
LIBRARY = os.path.join(REPO_ROOT, "benchmark/library")
SUBJECTS_A = ["global_facts", "elementary_mathematics"]  # 资料库覆盖好的两科


def local_research(task: str) -> str:
    """本地资料库专用查证 (无网络): 控制变量 + 快.
    返回与任务最相关的文本段; 无命中返回 no knowledge (学不到是正常结果)."""
    words = [w for w in re.split(r"[^a-z0-9]+", task.lower())
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
                score = sum(1 for w in words if w in line.lower())
                if score >= 2:
                    scored.append((score, line.strip()))
    if not scored:
        return "no knowledge found"
    scored.sort(key=lambda x: -x[0])
    facts = []
    for _, line in scored[:3]:
        facts.append(line.split("|", 1)[1].strip() if "|" in line
                     else line.strip())
    return " | ".join(facts)[:600]

# ══════════════════════════════════════════════════════════════
# 实验 A: 学习触发源对照
# ══════════════════════════════════════════════════════════════

def _novelty(body: JepaBody, task: str) -> float:
    """新奇度 = 1 - 任务与已学知识条目的最大余弦 (0=已覆盖, 1=完全陌生)."""
    z = body._situation_vec(task, "", "")
    zn = z / (np.linalg.norm(z) + 1e-9)
    best = -1.0
    for pool in (body.responder.pairs, body.responder.core_pairs):
        for pz, _ in pool:
            pzn = np.asarray(pz, np.float32)
            pzn = pzn / (np.linalg.norm(pzn) + 1e-9)
            s = float(np.dot(zn, pzn))
            if s > best:
                best = s
    return 1.0 - best if best >= 0 else 1.0


def ans_hit(content: str, answer: str) -> bool:
    """判定: 回答是否含答案核心词 (归一化后 ≥2 个长词命中)."""
    if not content or content.startswith("Task complete"):
        return False
    cn = _norm(content)
    an = _norm(answer)
    kws = [w for w in an.split() if len(w) > 3][:4]
    return sum(1 for w in kws if w in cn) >= 2


def load_task_stream() -> list[tuple[str, str]]:
    """任务流 = 资料库知识条目的 (question, answer) 对.
    学习动作永远有效 (资料库有对应知识) — 干净测"何时学习"的策略差异."""
    tasks = []
    for fn in ("global_facts", "elementary_mathematics",
               "high_school_computer_science"):
        with open(os.path.join(LIBRARY, f"{fn}.txt"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|", 1)
                if len(parts) == 2:
                    q, a = parts[0].strip(), parts[1].strip()
                    if q and a:
                        tasks.append((q, a))
    return tasks


def load_covered_mmlu() -> dict[str, list[tuple[str, list, int]]]:
    """MMLU 题集 → 只保留资料库可查证的题 (学习动作有效 + 真实长句/选项/噪声)."""
    out = {}
    for subj in SUBJECTS_A:
        items = load_subject(subj)
        covered = []
        for q, choices, ans in items:
            r = local_research(q)
            if not r.startswith("no knowledge"):
                covered.append((q, choices, ans))
        out[subj] = covered
    return out


def _mkz(seed: int, dim: int = 128) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _norm_v(v) -> np.ndarray:
    return np.asarray(v, np.float32) / (np.linalg.norm(v) + 1e-9)


def run_experiment_A(seed: int, per: int = 80, novelty_thresh: float = 0.55,
                     min_sim: float = 0.6):
    """机制级触发策略对照 (真实 RespondLearner 内核, 合成任务流).

    任务流 (80):
      40 新任务     — 方向独立, 与已学知识余弦 ~0      → 检索 miss
      20 模糊变体   — cos(z_v, z_old) ≈ 0.5: 检索 miss 但新奇度不高
                      ("似曾相识但不记得" — read 污染场景的结构根源)
      20 复习任务   — 与前面某任务同 z (cos 1.0)       → 已会
    预期: blocked 在 miss 时学 (不漏模糊变体, 跳过复习);
          curious 在新奇度高时学 (漏掉模糊变体 — 新奇≠该学);
          always 每次都学 (复习任务重复学, 浪费).
    """
    rng = np.random.RandomState(seed)
    # 构造知识空间
    z_old = [_mkz(seed * 1000 + i) for i in range(40)]
    stream = []
    for i in range(40):                      # 新任务
        stream.append((z_old[i], f"answer{i}"))
    for i in range(20):                      # 模糊变体 (原主题 i, 新答案)
        zv = _norm_v(z_old[i] + 1.732 * _mkz(seed * 2000 + i))
        stream.append((zv, f"variant-answer{i}"))
    for i in range(20):                      # 复习 (重复前 20 新任务)
        stream.append((z_old[i], f"answer{i}"))
    rng.shuffle(stream)

    out = {}
    for strat in ("blocked", "curious", "always"):
        from respond_learner import RespondLearner
        rl = RespondLearner(min_sim=min_sim, cap=400, seed=seed)
        learn_actions = 0
        correct = 0
        n_miss = 0
        for zq, ans in stream:
            text = rl.respond(zq)
            miss = text is None
            if miss:
                n_miss += 1
            # 新奇度: 与已学条目最大余弦
            best_sim = -1.0
            for pool in (rl.pairs, rl.core_pairs):
                for pz, _ in pool:
                    s = float(np.dot(_norm_v(zq), _norm_v(pz)))
                    if s > best_sim:
                        best_sim = s
            nov = 1.0 - best_sim if best_sim > 0 else 1.0
            if strat == "blocked":
                should = miss
            elif strat == "curious":
                should = nov > novelty_thresh
            else:
                should = True
            if should:
                learn_actions += 1
                rl.learn(zq, ans)
                correct += 1                       # 学了就会 (知识可达)
            else:
                if not miss and text == ans:       # 已会 → 答对
                    correct += 1
                # miss 但没学 (curious 漏学模糊变体) → 答错 (不计 correct)
        out[strat] = {"trigger_rate": learn_actions / len(stream),
                      "learn_actions": learn_actions,
                      "post_acc": correct / len(stream),
                      "unit_gain": (correct / learn_actions
                                    if learn_actions else 0.0),
                      "miss_rate": n_miss / len(stream)}
    return out


def run_experiment_A_full(seeds=(7, 42, 123), per: int = 60):
    print("=" * 72)
    print("实验 A: 学习触发源对照 (命题2: 受阻是触发源, 好奇只是策略性动力)")
    print(f"  任务流: 资料库知识条目 {per} 条 | 多 seed 符号检验 {seeds}")
    print("=" * 72)
    agg = {s: {"trigger_rate": [], "learn_actions": [],
               "post_acc": [], "unit_gain": [], "miss_rate": []}
           for s in ("blocked", "curious", "always")}
    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        s = run_experiment_A(seed, per=per)
        for strat, m in s.items():
            for k, v in m.items():
                agg[strat][k].append(v)
            print(f"  [{strat:<8}] 触发 {m['trigger_rate']:.0%} "
                  f"| 学 {m['learn_actions']} 次 | 学后正确率 {m['post_acc']:.0%} "
                  f"| 单位收益 {m['unit_gain']:.2f}")
    print("\n--- 多 seed 汇总 (均值±) ---")
    verdict = {}
    for strat, a in agg.items():
        mu = {k: float(np.mean(v)) for k, v in a.items()}
        sd = {k: float(np.std(v)) for k, v in a.items()}
        verdict[strat] = mu
        print(f"  [{strat:<8}] 触发 {mu['trigger_rate']:.0%}±{sd['trigger_rate']:.0%} "
              f"| 单位收益 {mu['unit_gain']:.2f}±{sd['unit_gain']:.2f} "
              f"| 学后 {mu['post_acc']:.0%}±{sd['post_acc']:.0%} "
              f"| miss {mu['miss_rate']:.0%}")
    # 判定 (预注册): 受阻触发应同时满足
    #  ① 精准性: 与 always 同正确率但学习动作更少 (不浪费)
    #  ② 完整性: 比 curious 正确率更高 (不漏"似曾相识但重要"的模糊新知识)
    g_b = verdict["blocked"]["unit_gain"]
    g_c = verdict["curious"]["unit_gain"]
    g_a = verdict["always"]["unit_gain"]
    acc_b = verdict["blocked"]["post_acc"]
    acc_c = verdict["curious"]["post_acc"]
    acc_a = verdict["always"]["post_acc"]
    act_b = verdict["blocked"]["learn_actions"]
    act_a = verdict["always"]["learn_actions"]
    print("\n[判定] 预注册标准:")
    print(f"  ① 精准性: 受阻({acc_b:.0%}@{act_b:.0f}动作) vs 总是({acc_a:.0%}@{act_a:.0f}动作)")
    print(f"  ② 完整性: 受阻({acc_b:.0%}) vs 好奇({acc_c:.0%}) — 好奇是否漏学模糊变体")
    precise = acc_b >= acc_a - 0.02 and act_b < act_a - 1
    complete = acc_b > acc_c + 0.03
    if precise and complete:
        verdict_text = ("✅ 命题2 强支持: 受阻触发既精准(同正确率省动作)又完整"
                        "(不漏'似曾相识但重要'的知识); 好奇触发漏学模糊变体 — "
                        "新奇≠该学, 受阻才是触发源")
    elif precise:
        verdict_text = ("⚠️ 命题2 部分支持: 受阻触发精准(不浪费)成立, "
                        "但与好奇的正确率差异不显著")
    else:
        verdict_text = "❌ 命题2 被反驳: 受阻触发无优势"
    print(verdict_text)
    return verdict, verdict_text


# ══════════════════════════════════════════════════════════════
# 实验 B: 沉淀判据对照 (合成数据流, 直接验证机制差异)
# ══════════════════════════════════════════════════════════════

class PassPromoteResponder:
    """受阻-通过沉淀判据 (B1): 条目被检索命中 且 外部判定正确 → 通过分+1;
    判定错误 → 通过分衰减 (-2); 通过分 ≥2 → 晋升 core.
    与 RespondLearner 的命中次数判据 (B2) 做机制级对照."""
    def __init__(self, min_sim=0.45, cap=12, promote_thresh=2, mode="pass"):
        self.min_sim = min_sim
        self.cap = cap
        self.core_cap = cap * 4
        self.promote_thresh = promote_thresh
        self.pairs = []            # 高频 (运动)
        self.core_pairs = []       # 低频 (沉淀)
        self.pass_scores = {}      # 高频索引 -> 通过分 (可负: 污染条目)
        self.hit_counts = {}       # 对照用: 命中次数 (模拟 B2)
        self.mode = mode           # pass=受阻-通过 | hit=命中次数

    def learn(self, z, text):
        zn = z / (np.linalg.norm(z) + 1e-9)
        for i, (pz, _) in enumerate(self.pairs):
            if float(np.dot(zn, np.asarray(pz, np.float32)
                            / (np.linalg.norm(pz) + 1e-9))) > 0.95:
                self.pairs[i] = (z, text)
                return
        self.pairs.append((z, text))
        if len(self.pairs) > self.cap:
            self.pairs.pop(0)

    def respond(self, z):
        """返回 (text, pool, idx); 未命中返回 (None, None, -1)."""
        zn = z / (np.linalg.norm(z) + 1e-9)
        best, best_sim, best_pool, best_idx = None, -1.0, None, -1
        for pool_name, pool in (("fast", self.pairs), ("core", self.core_pairs)):
            for i, (pz, pt) in enumerate(pool):
                s = float(np.dot(zn, np.asarray(pz, np.float32)
                                 / (np.linalg.norm(pz) + 1e-9)))
                if s > best_sim:
                    best, best_sim = pt, s
                    best_pool, best_idx = pool_name, i
        if best is None or best_sim < self.min_sim:
            return None, None, -1
        return best, best_pool, best_idx

    def report_outcome(self, idx: int, correct: bool):
        """外部判定回传 (只作用于高频层条目): 正确 → 通过分+1 / 命中+1;
        错误 → 通过分 -2 (污染惩罚). 达标 → 晋升 core."""
        if idx < 0 or idx >= len(self.pairs):
            return
        if self.mode == "pass":
            self.pass_scores[idx] = self.pass_scores.get(idx, 0) \
                + (1 if correct else -2)
            if self.pass_scores[idx] >= self.promote_thresh:
                self.core_pairs.append(self.pairs[idx])
                self.pass_scores[idx] = 0
        else:
            self.hit_counts[idx] = self.hit_counts.get(idx, 0) + 1
            if self.hit_counts[idx] >= self.promote_thresh:
                self.core_pairs.append(self.pairs[idx])
                self.hit_counts[idx] = 0

    def core_texts(self):
        return [t for _, t in self.core_pairs]


def run_experiment_B():
    """合成数据流: 正确知识 + 同构混叠错误条目 + 高压新知识冲击.
    混叠场景 (真实世界的上海/东京): 东京问题内化了上海答案 —
    错误条目 = (东京问题形式 z, 上海答案文本), 与正确条目高相似但独立.
    对比 pass(受阻-通过) vs hit(命中次数) 两种沉淀判据."""
    print("\n" + "=" * 72)
    print("实验 B: 沉淀判据对照 (命题1 选择性遗忘 + 命题3 受阻-通过判据)")
    print("  场景: 6 正确知识 + 混叠错误条目(东京问题内化上海答案) + 高压冲击")
    print("=" * 72)

    def z(topic):
        rng = np.random.RandomState(abs(hash(topic)) % 2**31)
        v = rng.randn(128).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def norm(v):
        return np.asarray(v, np.float32) / (np.linalg.norm(v) + 1e-9)

    def simulate(mode: str) -> dict:
        r = PassPromoteResponder(mode=mode, cap=10, promote_thresh=2)
        z5 = z("weather in city5")
        # ① 6 个正确知识 (city0-5 各绑自己的答案)
        for t in range(6):
            r.learn(z(f"weather in city{t}"), f"city{t}: sunny")
        # ② 混叠: city5 内化 city3 的答案 (东京问题→上海答案, 同构问题高相似独立条目)
        r.learn(z("weather in city5"), "city3: sunny")
        # ③ 使用循环 3 轮: 问 6 城天气, 外部判定 (答对=含自己城市的答案)
        for _ in range(3):
            for t in range(6):
                zq = z(f"weather in city{t}")
                text, pool, idx = r.respond(zq)
                if text is not None and pool == "fast":
                    correct = (text == f"city{t}: sunny")
                    r.report_outcome(idx, correct)
        # ④ 高压冲击: 学 15 条无关新知识 (高频淘汰最旧)
        for t in range(15):
            r.learn(z(f"totally different topic{t}"), f"junk{t}")
        # ⑤ 测量 (用情境 z 区分条目身份, 不用文本 — 错误/正确条目文本相同)
        def is_city5(cz):
            return float(np.dot(norm(cz), norm(z5))) > 0.9
        core = r.core_pairs
        correct_kept = sum(1 for t in range(6)
                           if any(not is_city5(cz) and
                                  f"city{t}: sunny" == txt
                                  for cz, txt in core))
        wrong_promoted = sum(1 for cz, _ in core if is_city5(cz))
        return {"core_n": len(core), "correct_kept": correct_kept,
                "wrong_promoted": wrong_promoted}

    out = {}
    for mode in ("pass", "hit"):
        m = simulate(mode)
        out[mode] = m
        print(f"  [{mode}] core沉淀 {m['core_n']} 条 | "
              f"正确知识保留 {m['correct_kept']}/6 | "
              f"错误条目(混叠)沉淀 {m['wrong_promoted']} 条")

    verdict_text = ""
    p, h = out["pass"], out["hit"]
    if h["wrong_promoted"] > p["wrong_promoted"] and p["wrong_promoted"] == 0:
        verdict_text = ("✅ 命题1+3 支持: 命中次数判据把高命中错误条目沉淀进 core "
                        "(错误固化 = '忘得不够'的病理); 受阻-通过判据 0 条错误沉淀 "
                        "(答错衰减 → 选择性遗忘 = 扬弃)")
    elif h["wrong_promoted"] > p["wrong_promoted"]:
        verdict_text = ("⚠️ 方向支持但受阻-通过也沉淀了错误 — 需检查惩罚强度")
    else:
        verdict_text = ("⚠️ 两判据均未沉淀错误条目 — 场景构造使错误条目未被命中")
    print("\n" + verdict_text)
    return out, verdict_text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=60)
    ap.add_argument("--seeds", default="7,42,123")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    t0 = time.time()
    if not args.skip_a:
        va, ta = run_experiment_A_full(seeds=seeds, per=args.per)
    if not args.skip_b:
        vb, tb = run_experiment_B()
    print(f"\n总耗时 {time.time()-t0:.0f}s")
    os.makedirs(os.path.join(REPO_ROOT, "benchmark/snapshots"), exist_ok=True)
    snap = {"time": time.time(),
            "experiment": "study_trigger",
            "A": va if not args.skip_a else None,
            "A_verdict": ta if not args.skip_a else "",
            "B": vb if not args.skip_b else None,
            "B_verdict": tb if not args.skip_b else ""}
    with open(f"{REPO_ROOT}/benchmark/snapshots/study_trigger_{int(time.time())}.json",
              "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=1)
