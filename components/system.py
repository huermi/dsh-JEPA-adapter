"""
三环组装器 (system.py) — JepaAgent
====================================
把 C1-C10 组件按三环架构驱动:

  内环 (认知): perception → world_model → energy(E1) → 学习
  中环 (行动): memory(写) → goal → planner → 动作 → 环境 → value(E2)
  外环 (元):   configurator.observe(perf, t, died) → 调制参数 → 各组件

事件流: Event 八元组 (t, s, a, s_next, e1, e2, perf, died)

用法:
  agent = build_default_system()
  for t in range(N):
      a, s = agent.decide(obs)           # 中环决策
      obs_next, reward, died = env.step(a)
      agent.learn(obs, a, obs_next, reward, died)   # 内环+外环
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .core import D, Event, ParamKind, ComponentConfig, Obs, Embed
from .perception import PerceptionEncoder, DummyPerception
from .world_model import WorldModel, DummyWorldModel
from .energy import EnergySystem, CuriosityEnergy
from .value import ValueSystem, HomeostaticValue
from .memory import MemorySystem, AdaptiveMemory
from .goal import GoalGenerator, ValueAlignedGoal
from .planner import Planner, GreedyPlanner
from .symbol import SymbolLayer, BehavioralSymbol
from .configurator import Configurator, PerfConfigurator


class JepaAgent:
    """三环驱动组装器"""

    def __init__(self, perception: PerceptionEncoder, world_model: WorldModel,
                 energy: EnergySystem, value: ValueSystem, memory: MemorySystem,
                 goal: GoalGenerator, planner: Planner, symbol: Optional[SymbolLayer] = None,
                 configurator: Optional[Configurator] = None):
        self.perception = perception
        self.world_model = world_model
        self.energy = energy
        self.value = value
        self.memory = memory
        self.goal = goal
        self.planner = planner
        self.symbol = symbol
        self.configurator = configurator
        self.t = 0
        self._cur_goal: Optional[Embed] = None
        self._goal_hold = 0

    # ── 中环: 决策 ────────────────────────────────────────
    def decide(self, obs: Obs) -> tuple[int, Embed]:
        """编码 → 目标 → 规划 → 动作 (MPC 第一步). 返回 (a, s)."""
        s = self.perception.encode(obs)
        if self._cur_goal is None:
            self._cur_goal = self.goal.next_goal(self.memory, s)
        a = self.planner.select_action(s, self._cur_goal)
        self._goal_hold += 1
        return a, s

    # ── 内环+外环: 学习与调制 ─────────────────────────────
    def learn(self, obs: Obs, a: int, obs_next: Obs, perf: float, died: bool = False) -> Event:
        """观测 → E1 学习 → E2 价值 → 记忆写 → Configurator 调制."""
        s = self.perception.encode(obs)
        s_next = self.perception.encode(obs_next)

        # 内环: 世界模型 1 步梯度 (AdaJEPA)
        e1 = self.world_model.step(s, a, s_next)

        # 中环: 价值 + 记忆
        anchor_v = self.value.anchor(s_next, demand=1.0)
        e2 = self.energy.update(e1, anchor_v)
        self.memory.write(s, e1, e2)
        self.memory.add_transition(s, a, s_next, e1)

        # 目标坚持: 能量趋势 + 坚持预算
        _, e1_trend, _ = self.energy.get_stats()
        if self.goal.should_switch(e1_trend, self._goal_hold):
            self._cur_goal = self.goal.next_goal(self.memory, s)
            self._goal_hold = 0

        # 外环: Configurator 观测 + 调制
        if self.configurator is not None:
            self.configurator.observe(perf, self.t, died)
            self._apply_config(self.configurator.get_config())

        event = Event(t=self.t, s=s, a=a, s_next=s_next,
                      e1=e1, e2=e2, perf=perf, died=died)
        self.t += 1
        return event

    def _apply_config(self, cfg: ComponentConfig) -> None:
        """把 Configurator 输出的参数写入各组件 (有 set_param 的)."""
        for kind, val in cfg.params.items():
            if hasattr(self.value, "set_param"):
                self.value.set_param(kind, val)
            if hasattr(self.goal, "set_param"):
                self.goal.set_param(kind, val)
            if kind == ParamKind.LEARNING_RATE and hasattr(self.world_model, "set_lr"):
                self.world_model.set_lr(val)
            if kind == ParamKind.EXPLORE_TEMP and hasattr(self.planner, "set_temperature"):
                self.planner.set_temperature(val)

    # ── 便捷: 连续运行 ────────────────────────────────────
    def run(self, env, n_ticks: int) -> list[Event]:
        """env 需提供: reset() → obs; step(a) → (obs_next, reward, died, done)"""
        obs = env.reset()
        events = []
        for _ in range(n_ticks):
            a, _ = self.decide(obs)
            obs_next, reward, died, done = env.step(a)
            ev = self.learn(obs, a, obs_next, reward, died)
            events.append(ev)
            obs = obs_next
            if done:
                break
        return events


def build_default_system(seed: int = 0) -> JepaAgent:
    """默认实现组装 (Dummy/轻量实现, 组装自检用)"""
    perception = DummyPerception(seed=seed)
    world_model = DummyWorldModel(seed=seed)
    energy = CuriosityEnergy()
    value = HomeostaticValue()
    memory = AdaptiveMemory(seed=seed)
    goal = ValueAlignedGoal()
    planner = GreedyPlanner(seed=seed)
    planner.set_world_value(world_model, value)
    symbol = BehavioralSymbol(seed=seed)
    configurator = PerfConfigurator()
    return JepaAgent(perception, world_model, energy, value, memory,
                     goal, planner, symbol, configurator)


class DemoEnv:
    """自检用最小环境: 动态 6×6 网格 + 1 个漂移物品 (jpi1 简化)"""
    def __init__(self, size: int = 6, seed: int = 0):
        self.size = size
        self.rng = np.random.RandomState(seed)
        self.pos = np.array([3, 3], dtype=np.float32)
        self.t = 0

    def reset(self) -> Obs:
        self.pos = np.array([3, 3], dtype=np.float32)
        self.t = 0
        return self._obs()

    def _obs(self) -> Obs:
        item = np.array([1.0 + 1.5 * np.sin(self.t * 0.01),
                         2.0 + 1.5 * np.cos(self.t * 0.013)], dtype=np.float32)
        return np.concatenate([self.pos, item, [0.5]]).astype(np.float32)

    def step(self, a: int):
        dxy = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)][a]
        self.pos = np.clip(self.pos + np.array(dxy, dtype=np.float32), 0, self.size - 1)
        self.t += 1
        obs_next = self._obs()
        dist = float(np.linalg.norm(self.pos - obs_next[2:4]))
        reward = float(np.exp(-dist / 1.2))          # 锚值 (靠近物品)
        died = False
        return obs_next, reward, died, False
