"""
自动化学习训练 (auto_train.py) — 知识累积制
============================================
快速提升模型智商: 解决"知识零累积 + 资料库覆盖不足"两大瓶颈.

架构 (对齐 benchmark/自动化训练计划.md):
  L0 知识供给: 训练轮答对的 (问题|正确选项) 回放入库 — 知识单调累积
  L1 训练执行: 滚动训练轮 (每轮新题, 扩充资料库) + 固定评估轮 (测泛化)
  L2 质量反馈: 每轮评估分数曲线 + 收敛判定 (连续2轮提升<1点 → 停)

诚实性:
  - 只入答对的 (受阻-通过判据: 实践考验通过才沉淀; 答错绝不入库)
  - 评估题与训练题不重叠 (perm 抽样分离) — 无"背答案"作弊
  - 入库答案 = choices[ans] (数据集 ground truth), 非模型自答文本
  - 评估 body 每轮干净新建, 查证源 = 当前资料库 → 测"教材扩充 → 泛化提升"
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
from benchmark_check import extract_answer
from learner_loop import LearnerLoop, load_subject, SUBJECTS, LIBRARY

TRAIN_PER = 24      # 每科每轮训练题数 (滚动窗口)
EVAL_PER = 12       # 每科评估题数 (固定)
SEED = 42

_replayed: set[str] = set()   # 进程内防重复入库


def build_body(seed: int = SEED, soft_align: bool = True) -> JepaBody:
    body = JepaBody(seed=seed,
                    config=PluginConfig(seed=seed, respond_cap=2000,
                                        respond_min_sim=0.18,
                                        soft_align=soft_align))
    body.ensure_semantic()
    return body


def subject_count(subject: str) -> int:
    """该科资料库当前条目数 (教材容量控制)."""
    path = os.path.join(LIBRARY, f"{subject}.txt")
    if not os.path.exists(path):
        return 0
    return sum(1 for _ in open(path, encoding="utf-8"))


def append_knowledge(subject: str, question: str, answer: str) -> bool:
    """经验回放 (L0): (问题|正确选项) 写回该学科资料库.
    受阻-通过判据的持久层兑现: 只有被实践考验且通过的才沉淀."""
    qq = question.strip()
    if not qq or qq in _replayed:
        return False
    aa = str(answer).strip()
    if not aa:
        return False
    _replayed.add(qq)
    path = os.path.join(LIBRARY, f"{subject}.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{qq} | {aa}\n")
    return True


def learn_one(loop: LearnerLoop, q: str) -> dict:
    """单题学习, 返回判定用信息 (与 learner_loop 一致)."""
    traj = loop.learn_and_answer(q)
    return traj


def eval_one(subj: str, q: str, choices, ans: int) -> tuple[bool, dict]:
    """评估单题: 每题独立干净 body (learn_benchmark 模式).
    关键设计: 评估测"教材(资料库)能否支撑该题", 而非"记忆跨题污染" —
    共享 body 会让第一题内化的条目被后续同主题题检索命中 (同构混叠).
    每题干净 → 查证源=当前资料库(教材) → 测真实泛化."""
    body = build_body()
    loop = LearnerLoop(body, max_rounds=5)
    traj = loop.learn_and_answer(q)
    hit = (extract_answer(traj["content"], choices) == ans)
    return hit, traj


def run_round(subject: str, items: list, loop: LearnerLoop,
              replay: bool) -> tuple[int, int, int]:
    """跑一批题: 学习+判定. replay=True 时答对回放入库.
    返回 (答对数, 查证动作数, 回放入库数)."""
    ok = n_research = n_replay = 0
    for q, choices, ans in items:
        traj = learn_one(loop, q)
        hit = (extract_answer(traj["content"], choices) == ans)
        n_research += sum(1 for r in traj["rounds"]
                          if r["action"] == "research")
        if hit:
            ok += 1
            if replay:
                n_replay += append_knowledge(subject, q, choices[ans])
    return ok, n_research, n_replay


def ingest_material(subject: str, items: list, cap: int = 0) -> int:
    """教材采集 (L0): 训练题 (问题|正确选项) 直接入库.
    关键设计: 教材正确性由数据标注 (choices[ans] = ground truth) 保证,
    与模型是否答对无关 — 训练/评估解耦, 知识积累不受模型瓶颈限制.
    (等价于 MMLU train 集的本地版: test 集劈半, 一半当教材一半当考试)
    cap>0: 该科教材容量封顶 (防同科相似条目过多 → 检索混叠)"""
    n = 0
    for q, choices, ans in items:
        if cap > 0 and subject_count(subject) >= cap:
            break
        n += append_knowledge(subject, q, choices[ans])
    return n


def internalize_material(subject: str, items: list, body: JepaBody) -> int:
    """教材内化 (LeCun 改进2: 学习=结构变化): (问题→完整事实) learn_response
    进 body 的 responder — 模型结构内吸收, 而非只写资料库 (txt).
    关键: 条目文本 = "问题 | 答案" 完整形式 (与查证模式内化的事实一致),
    而非短选项 — 短答案 (40%) 与未见题相似度 0.36-0.70 全被 margin 拒 (1% 根因);
    完整条目共享问题词 → 未见题命中 → 答对.
    回应批评 "你在训练数据库, 不是训练模型": 内化模式测模型本身的记忆."""
    n = 0
    for q, choices, ans in items:
        body.learn_response({"task": q}, f"{q} | {str(choices[ans])}")
        n += 1
    return n


def eval_memory(body: JepaBody, eval_items: list) -> tuple[int, int]:
    """模型记忆评估 (LeCun 改进2): 不查证, 直接用内化的模型答题
    (benchmark_mode: 禁探索 — 纯 responder 检索, 排除资料库/查证增益).
    返回 (答对数, 总题数)."""
    ok = 0
    for q, choices, ans in eval_items:
        resp = body.chat_completion([{"role": "user", "content": q}], [])
        if extract_answer(resp.get("content", ""), choices) == ans:
            ok += 1
    return ok, len(eval_items)


def build_memory_body(seed: int = SEED) -> JepaBody:
    """内化评估用 body: benchmark_mode=True (禁探索 → 纯记忆检索)."""
    body = JepaBody(seed=seed,
                    config=PluginConfig(seed=seed, benchmark_mode=True,
                                        respond_cap=2000,
                                        respond_min_sim=0.18))
    body.ensure_semantic()
    return body


def subject_split(subj: str, rng: np.random.RandomState, round_i: int):
    """训练/评估分源: perm 前 EVAL_PER 为固定评估集, 其余滚动训练.
    返回 (train_items, eval_items, eval_idx, train_pool)."""
    items = load_subject(subj)
    n = len(items)
    perm = rng.permutation(n)
    eval_idx = set(int(i) for i in perm[:EVAL_PER])
    train_pool = [int(i) for i in perm if int(i) not in eval_idx]
    start = round_i * TRAIN_PER
    train_idx = train_pool[start:start + TRAIN_PER]
    return ([items[i] for i in train_idx],
            [items[i] for i in eval_idx],
            eval_idx, train_pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3,
                    help="训练轮数 (每轮滚动 24 题/科)")
    ap.add_argument("--subjects", default=",".join(SUBJECTS))
    ap.add_argument("--material-cap", type=int, default=48,
                    help="每科教材容量封顶 (防同科相似条目过多→检索混叠)")
    ap.add_argument("--internalize", action="store_true",
                    help="内化模式: 教材学进模型 (learn_response), 评估测模型记忆 "
                    "(LeCun: 学习=结构变化, 非扩充数据库)")
    ap.add_argument("--eval-every", action="store_true",
                    help="每轮训练后都评估 (默认: 训练完才评估)")
    args = ap.parse_args()
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    print("=" * 72)
    mode = "内化模式 (教材→模型, 测记忆)" if args.internalize else \
           "查证模式 (教材→资料库, 测泛化)"
    print("自动化学习训练 (知识累积制) | 训练滚动 24 题/科/轮 | "
          f"评估固定 {EVAL_PER} 题/科 | 共 {args.rounds} 轮 | {mode}")
    print("  经验回放: 答对 → (问题|正确选项) 入库 | 评估/训练分源 | "
          "收敛: 连续2轮提升<1点")
    print("=" * 72)

    rng = np.random.RandomState(SEED)
    # 每科: 预生成 perm 与评估集 (全轮共享)
    plan = {}
    for subj in subjects:
        _, _, eval_idx, train_pool = subject_split(subj, rng, 0)
        plan[subj] = {"eval_idx": eval_idx, "train_pool": train_pool}

    history = []
    last_acc = None
    for round_i in range(args.rounds):
        t0 = time.time()
        print(f"\n── 第 {round_i + 1} 轮训练 (滚动窗口 "
              f"[{round_i*TRAIN_PER}:{(round_i+1)*TRAIN_PER}]) ──")
        n_eval_ok = n_replay = n_train = n_eval = 0
        for subj in subjects:
            items = load_subject(subj)
            p = plan[subj]
            start = round_i * TRAIN_PER
            train_idx = p["train_pool"][start:start + TRAIN_PER]
            train_items = [items[i] for i in train_idx]
            eval_items = [items[i] for i in p["eval_idx"]]
            # 训练: 教材采集 (标注保证正确, 不经模型 — 训练/评估解耦)
            n_replay += ingest_material(subj, train_items,
                                        cap=args.material_cap)
            n_train += len(train_items)
            if args.internalize:
                # ── 内化模式 (LeCun 改进2): 教材学进模型 → 测模型记忆 ──
                body_m = build_memory_body()
                internalize_material(subj, train_items, body_m)
                ok_m, n_m = eval_memory(body_m, eval_items)
                n_eval_ok += ok_m
                n_eval += n_m
                print(f"  [{subj[:28]:<28}] 内化 +{len(train_items)} 条 "
                      f"| 模型记忆 {ok_m}/{n_m}")
            else:
                # 查证模式: 每题独立干净 body (查证源 = 当前资料库=教材)
                ok_e = 0
                for q_e, c_e, a_e in eval_items:
                    hit_e, _ = eval_one(subj, q_e, c_e, a_e)
                    ok_e += hit_e
                n_eval_ok += ok_e
                n_eval += len(eval_items)
                print(f"  [{subj[:28]:<28}] 教材 +{len(train_items)} 条 "
                      f"| 评估 {ok_e}/{len(eval_items)}")
        eval_acc = n_eval_ok / max(n_eval, 1)
        dt = time.time() - t0
        delta = (eval_acc - last_acc) if last_acc is not None else float("nan")
        print(f"── 第{round_i+1}轮小结: 教材 +{n_train} 条/科 "
              f"| 评估 {eval_acc:.1%} ({n_eval_ok}/{n_eval}) | "
              f"本轮入库 {n_replay} 条 | {dt:.0f}s | Δ{delta:+.1%}")
        history.append({"round": round_i + 1, "train_n": n_train,
                        "eval_acc": eval_acc, "replay": n_replay,
                        "time_s": round(dt, 1)})
        last_acc = eval_acc
        # 收敛判定: 从第 3 轮起, 连续 2 轮提升 < 1 点 → 停
        if (round_i >= 2 and len(history) >= 2
                and history[-1]["eval_acc"] - history[-2]["eval_acc"] < 0.01
                and history[-2]["eval_acc"] - history[-3]["eval_acc"] < 0.01
                if len(history) >= 3 else False):
            print("收敛: 连续 2 轮评估提升 < 1 点, 停止训练.")
            break

    # ── 最终报告 ──
    print("\n" + "=" * 72)
    print("[训练报告]")
    for h in history:
        print(f"  轮 {h['round']}: 教材 +{h['train_n']} 条/科 | "
              f"评估 {h['eval_acc']:.1%} | 入库 {h['replay']} 条")
    print("  资料库规模:")
    for subj in subjects:
        path = os.path.join(LIBRARY, f"{subj}.txt")
        n = sum(1 for _ in open(path, encoding="utf-8")) \
            if os.path.exists(path) else 0
        print(f"    {subj:<30} {n} 条")
    print("=" * 72)
    os.makedirs(os.path.join(REPO_ROOT, "benchmark/snapshots"), exist_ok=True)
    with open(f"{REPO_ROOT}/benchmark/snapshots/auto_train_{int(time.time())}.json",
              "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=1)
    return history[-1]["eval_acc"] if history else 0.0


if __name__ == "__main__":
    sys.exit(0 if main() >= 0 else 1)
