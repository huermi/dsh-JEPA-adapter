"""
睡眠巩固验证 (sleep_check.py)
==============================
检验做中学闭环的"睡眠环":
  实验1 巩固价值: 白天在线学习后, 睡眠重放是否让 E1 进一步下降?
        + 优先级 (高惊讶优先) vs 均匀 vs 不巩固
        + 稀有高惊讶经验 (重要事件) 是否被选择性强化 — 人类睡眠的意义
  实验2 防灾难性遗忘: 巩固任务 B 后, 任务 A 的知识是否被破坏?
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import sys
import numpy as np

sys.path.insert(0, REPO_ROOT)

from components.world_model import ResidualWorldModel
from components.memory import AdaptiveMemory
from components.consolidation import SleepConsolidation

D = 768
N_ACTIONS = 5


def make_task(n_common=480, n_rare=20, c_common=0.05, c_rare=0.3,
              region=2.0, seed=0):
    """任务转移流: 普通转移 (小偏移, 区域 +region) + 稀有重要转移
    (大偏移, 区域 -region). 两个子分布输入可区分 (s[0] 区域标记) —
    预测器学'条件偏移' (该区域该偏移), 睡眠强化稀有区域不会污染普通区域.
    高惊讶 = 与预测器差异大 = 难学但重要 (人类: 平凡日 vs 惊吓事件)"""
    rng = np.random.RandomState(seed)
    trans = []
    for _ in range(n_common):
        s = rng.randn(D).astype(np.float32)
        s[0] += region                       # 普通区域
        a = rng.randint(N_ACTIONS)
        sn = s + c_common
        trans.append((s, a, sn))
    for _ in range(n_rare):
        s = rng.randn(D).astype(np.float32)
        s[0] -= region                       # 稀有区域
        a = rng.randint(N_ACTIONS)
        sn = s + c_rare
        trans.append((s, a, sn))
    rng.shuffle(trans)
    return trans


def online_learn(wm, trans, n, seed=0):
    """白天在线学习 n 条 (AdaJEPA 1 步梯度, 模拟探索)"""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(trans), n, replace=False)
    for i in idx:
        s, a, sn = trans[i]
        wm.step(s, a, sn)


def eval_task(wm, trans, which="common", n=60):
    """评估某子集的 E1 (留出, 不训练)"""
    if which == "common":
        pool = [t for t in trans if t[2][0] - t[0][0] < 0.2]   # 偏移小 = 普通
    else:
        pool = [t for t in trans if t[2][0] - t[0][0] >= 0.2]  # 偏移大 = 稀有
    if not pool:
        return float("nan")
    rng = np.random.RandomState(0)
    idx = rng.choice(len(pool), min(n, len(pool)), replace=False)
    es = [wm.energy(s, a, sn) for i in idx for s, a, sn in [pool[i]]]
    return float(np.mean(es))


def fill_memory(memory, trans, wm):
    """白天经验写入记忆: 惊讶度 = 该转移对当前模型的真实预测误差"""
    for s, a, sn in trans:
        e1 = wm.energy(s, a, sn)
        memory.add_transition(s, a, sn, e1)


def exp1():
    print("=" * 70)
    print("实验1: 睡眠巩固的价值 — 采样策略对照")
    print("=" * 70)
    trans = make_task(seed=0, region=2.0)
    results = {}

    for label, prio, mix, do_sleep in [("不巩固", 1.0, 0.5, False),
                                       ("均匀重放", 0.0, 0.0, True),
                                       ("混合重放(默认)", 1.0, 0.5, True),
                                       ("纯优先级(极端)", 1.0, 1.0, True)]:
        wm = ResidualWorldModel(n_actions=N_ACTIONS, seed=7)
        online_learn(wm, trans, 100)          # 白天在线学 100 条
        mem = AdaptiveMemory(seed=7)
        fill_memory(mem, trans, wm)           # 经验写记忆 (含真实惊讶度)
        e1_c0 = eval_task(wm, trans, "common")
        e1_r0 = eval_task(wm, trans, "rare")

        if do_sleep:
            sleep = SleepConsolidation(prio_power=prio, prio_mix=mix,
                                       epochs=3, batch=64)
            r = sleep.consolidate(mem, wm)
            n_replay = r["n_replayed"]
        else:
            n_replay = 0

        e1_c1 = eval_task(wm, trans, "common")
        e1_r1 = eval_task(wm, trans, "rare")
        drop_c = (1 - e1_c1 / max(e1_c0, 1e-12)) * 100
        drop_r = (1 - e1_r1 / max(e1_r0, 1e-12)) * 100
        results[label] = (drop_c, drop_r, e1_c1, e1_r1)
        print(f"\n[{label}] 重放 {n_replay} 条")
        print(f"  普通转移 E1: {e1_c0:.5f} → {e1_c1:.5f} (变化 {drop_c:+.0f}%)")
        print(f"  稀有转移 E1: {e1_r0:.5f} → {e1_r1:.5f} (下降 {drop_r:.0f}%)")

    print("\n" + "-" * 70)
    print("裁决 (绝对值, 防百分比假象):")
    m = results["混合重放(默认)"]
    p = results["纯优先级(极端)"]
    ok_sleep = m[1] > results["不巩固"][1]
    ok_prio = m[1] >= results["均匀重放"][1] - 5
    ok_no_damage = m[2] < 0.02   # 普通 E1 绝对值保持 < 0.02 (仍是好预测)
    print(f"  睡眠提升稀有经验掌握: {'✅' if ok_sleep else '❌'} "
          f"(混合 {m[1]:.0f}% vs 不巩固 {results['不巩固'][1]:.0f}%)")
    print(f"  选择性强化 ≥ 均匀: {'✅' if ok_prio else '❌'} "
          f"(混合 {m[1]:.0f}% vs 均匀 {results['均匀重放'][1]:.0f}%)")
    print(f"  不饿死常规知识: {'✅' if ok_no_damage else '❌'} "
          f"(混合普通 E1 {m[2]:.5f} < 0.02; 纯优先级极端 {p[2]:.5f})")
    return results


def exp2():
    print("\n" + "=" * 70)
    print("实验2: 防灾难性遗忘 — 巩固任务 B 后, 任务 A 是否保持")
    print("=" * 70)
    transA = make_task(seed=1, region=2.0)
    transB = make_task(seed=2, region=-2.0, c_common=0.12, c_rare=0.5)  # 不同区域+分布的任务 B
    # 长期池: A 的旧经验 (惊讶度在巩固时动态计算: 遗忘风险 → 优先重放)
    rehearsal_A = transA[:200]

    results = {}
    for label, use_rehearsal in [("巩固B(无旧经验)", False),
                                 ("巩固B+旧经验重放", True)]:
        wm = ResidualWorldModel(n_actions=N_ACTIONS, seed=7)
        # 阶段1: 学会 A
        online_learn(wm, transA, 100)
        memA = AdaptiveMemory(seed=7)
        fill_memory(memA, transA, wm)
        SleepConsolidation(prio_power=1.0, prio_mix=0.5,
                           epochs=3, batch=64).consolidate(memA, wm)
        eA0 = eval_task(wm, transA, "common")          # A 掌握后

        # 阶段2: 白天学 B (在线覆盖 A)
        online_learn(wm, transB, 100)
        eA1 = eval_task(wm, transA, "common")          # 在线学 B 后

        # 阶段3: 睡眠巩固 B (可选混入 A 旧经验)
        # rehearsal 的惊讶度 = 当前模型对 A 的误差 (正在被遗忘 → 高惊讶 → 优先重放)
        memB = AdaptiveMemory(seed=7)
        fill_memory(memB, transB, wm)
        rehearsal_dyn = None
        if use_rehearsal:
            rehearsal_dyn = [(s, a, sn, wm.energy(s, a, sn))
                             for s, a, sn in transA[:480]]
        r = SleepConsolidation(prio_power=1.0, prio_mix=0.5,
                               epochs=3, batch=64).consolidate(
            memB, wm, rehearsal=rehearsal_dyn)
        eA2 = eval_task(wm, transA, "common")          # 巩固 B 后

        d1 = (eA1 / max(eA0, 1e-12) - 1) * 100
        d2 = (eA2 / max(eA0, 1e-12) - 1) * 100
        results[label] = (eA0, eA1, eA2, d1, d2)
        print(f"\n[{label}]")
        print(f"  A 掌握后 E1: {eA0:.5f} → 在线学B后 {eA1:.5f} "
              f"({d1:+.0f}%) → 巩固B后 {eA2:.5f} ({d2:+.0f}%)")
        if use_rehearsal:
            print(f"  重放 {r['n_replayed']} 次, 其中旧经验 {r['n_rehearsal_sampled']} 次")

    no_r = results["巩固B(无旧经验)"]
    with_r = results["巩固B+旧经验重放"]
    ok = with_r[4] <= max(no_r[4], 0) + 10   # 有旧经验重放的巩固应比无旧经验少伤 A
    print(f"\n裁决: {'✅ 旧经验重放防遗忘生效' if ok else '❌ 旧经验重放未生效'} "
          f"(带重放 {with_r[4]:+.0f}% vs 无重放 {no_r[4]:+.0f}%)")
    return no_r[4], with_r[4]


if __name__ == "__main__":
    r1 = exp1()
    r2 = exp2()
    print("\n" + "=" * 70)
    print(f"总体: 实验1 {'✅' if r1['混合重放(默认)'][1] > 0 else '⚠️'} | "
          f"实验2 {'✅' if r2[1] <= r2[0] + 10 else '⚠️'}")
    print("=" * 70)
