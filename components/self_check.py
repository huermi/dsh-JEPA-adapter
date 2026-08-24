"""
组装自检 (components/self_check.py)
====================================
验证:
  1. 包可导入, 全部组件接口健全 (ABC 可实例化 via 默认实现)
  2. 默认实现组装成 JepaAgent 并跑通 N tick (三环驱动)
  3. 事件流八元组完整, E1/E2/绩效/调制全部生效
  4. Configurator 调制链路: observe → get_config → _apply_config
"""
import sys
import numpy as np

sys.path.insert(0, "D:/JEPA")

from components.core import D, Event, ParamKind, HealthStats, cosine
from components.perception import DummyPerception
from components.world_model import DummyWorldModel
from components.energy import CuriosityEnergy
from components.value import HomeostaticValue
from components.memory import AdaptiveMemory
from components.goal import ValueAlignedGoal
from components.planner import GreedyPlanner
from components.symbol import BehavioralSymbol
from components.configurator import PerfConfigurator
from components.system import JepaAgent, build_default_system, DemoEnv


def check_components():
    """1. 组件可实例化 (默认实现)"""
    names = []
    for obj, n in [(DummyPerception(), "C1"), (DummyWorldModel(), "C2"),
                   (CuriosityEnergy(), "C3"), (HomeostaticValue(), "C4"),
                   (AdaptiveMemory(), "C5"), (ValueAlignedGoal(), "C6"),
                   (GreedyPlanner(), "C7"), (BehavioralSymbol(), "C8"),
                   (PerfConfigurator(), "C10")]:
        names.append(n)
    print(f"[1] 组件实例化: {len(names)}/{len(names)} OK ({','.join(names)} + C9 占位)")
    return True


def check_types():
    """2. 类型系统健全"""
    ev = Event.blank(t=5)
    assert ev.s.shape == (D,), "表征维度必须 768"
    assert isinstance(ev.e1, float) or ev.e1 == 0
    assert ParamKind.HOMEOSTASIS_RATE in ParamKind
    c = cosine(np.ones(D), np.ones(D))
    assert abs(c - 1.0) < 1e-6, "cosine 自相似必须 1"
    print(f"[2] 类型系统: Event 八元组 + ParamKind({len(list(ParamKind))} 项) + cosine OK")
    return True


def check_run(n_ticks: int = 200):
    """3. 组装运行: 三环驱动"""
    agent = build_default_system(seed=7)
    env = DemoEnv(seed=7)
    events = agent.run(env, n_ticks)

    assert len(events) == n_ticks, "事件数必须等于 tick 数"
    e1s = [e.e1 for e in events]
    e2s = [e.e2 for e in events]
    perfs = [e.perf for e in events]
    # 三环信号都活跃 (E2 是行动理由, 可正可负 — 活性 = 非恒定)
    assert max(e1s) > 0, "E1 必须非零"
    assert max(e2s) > min(e2s), "E2 必须活跃 (非恒定)"
    assert max(perfs) > 0, "绩效必须非零"
    # 记忆写入
    assert len(agent.memory.items) > 0, "记忆必须写入"
    print(f"[3] 三环运行 {n_ticks} tick: E1 均值 {np.mean(e1s):.4f} | "
          f"E2 范围 [{min(e2s):.3f}, {max(e2s):.3f}] | 绩效均值 {np.mean(perfs):.4f}")
    print(f"    记忆 {len(agent.memory.items)} 条 | 原型 {len(agent.memory.prototypes)} 个 | "
          f"Configurator 速率 {agent.configurator.rate:.4f}")
    return True


def check_configurator():
    """4. Configurator 调制链路"""
    cfg = PerfConfigurator()
    for t in range(400):
        cfg.observe(perf=5.0 + (t % 100) * 0.01, t=t)   # 低绩效 → 爬山
    c = cfg.get_config()
    rate = c.get(ParamKind.HOMEOSTASIS_RATE)
    assert rate != 0.001, "调制后速率必须变化"
    # 死亡急刹车
    cfg.observe(perf=0.0, t=500, died=True)
    rate2 = cfg.get_config().get(ParamKind.HOMEOSTASIS_RATE)
    assert rate2 < rate, "死亡后速率必须下降 (急刹车)"
    print(f"[4] Configurator: 速率 {rate:.4f} → 急刹车后 {rate2:.4f} | "
          f"反转 {cfg.reversals} | 急刹车 {cfg.panics} OK")
    return True


if __name__ == "__main__":
    print("=" * 66)
    print("JEPA AI 组件组装自检")
    print("=" * 66)
    ok = all([check_components(), check_types(), check_run(), check_configurator()])
    print("=" * 66)
    print(f"自检 {'全部通过 ✅' if ok else '失败 ❌'}")
    sys.exit(0 if ok else 1)
