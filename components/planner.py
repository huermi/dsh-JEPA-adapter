"""
C7 Planner — 规划器 (中环)
==========================
契约: 潜空间离线思维 — CEM 搜索动作序列, MPC 滚动只执行第一步.

核心原则 (实证):
  - 前瞻锚: 行动选择 = 最大化预测未来状态价值 (Friston 期望自由能)
  - 饥饿门控导航场 (jpi12): 饥饿时被资源吸引 (nav_gain),
    饱足时不被吸引 — 决策与内稳态需求耦合
  - 探索-利用: 温度调制 (Configurator)

接口:
  plan(s, goal) → action_seq          (CEM 潜空间搜索)
  select_action(s, goal) → int        (MPC: 只执行第一步)
  set_temperature(temp)

验收: stable-worldmodel CEM; jpi8 蒸馏预算对比; p2_check.py
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .core import D, Embed, ParamKind


class Planner(ABC):
    """C7 规划器接口"""

    @abstractmethod
    def plan(self, s: Embed, goal: Embed, horizon: int = 5) -> list[int]:
        """潜空间 CEM 搜索: 返回动作序列."""

    @abstractmethod
    def select_action(self, s: Embed, goal: Embed) -> int:
        """MPC 滚动: 只执行第一步."""

    @abstractmethod
    def set_temperature(self, temp: float) -> None:
        """探索温度 (ParamKind.EXPLORE_TEMP)."""


class GreedyPlanner(Planner):
    """真实实现 v2: 1 步前瞻 (世界模型) + 目标趋近 + 饥饿门控导航 + 内稳态锚
    评分 (jpi12 决策动力学):
      score = -w_goal·dist(s_next, goal) + w_val·anchor(s_next)
              + nav_gain(s_next) - move_pen + 探索噪声
    """
    def __init__(self, n_actions: int = 5, temp: float = 0.05, seed: int = 0,
                 w_goal: float = 0.5, w_val: float = 0.6, nav_sigma: float = 3.0,
                 feed_boost: float = 0.5, hunger_gate: float = 0.3):
        self.n_actions = n_actions
        self.temp = temp
        self.w_goal = w_goal
        self.w_val = w_val
        self.nav_sigma = nav_sigma
        self.feed_boost = feed_boost
        self.hunger_gate = hunger_gate
        self.rng = np.random.RandomState(seed)
        self.world_model = None      # 组装时注入
        self.value = None

    def set_world_value(self, world_model, value) -> None:
        self.world_model = world_model
        self.value = value

    def set_temperature(self, temp: float) -> None:
        self.temp = temp

    def _hunger(self) -> float:
        return getattr(self.value, "hunger", 0.0)

    def _resource_protos(self) -> list[Embed]:
        return getattr(self.value, "resource_protos", [])

    def _score(self, s: Embed, a: int, goal: Embed) -> float:
        d = self.world_model.predict_delta(s, a) if self.world_model else np.zeros(D, np.float32)
        s_next = s + d
        # 目标趋近
        dist_goal = float(np.linalg.norm(s_next - goal))
        # 内稳态锚 (前瞻价值)
        val = self.value.anchor(s_next) if self.value else 0.0
        # 饥饿门控导航场: 饥饿时被资源吸引 (jpi12)
        hungry = max(0.0, self._hunger() - self.hunger_gate) * 2.0
        nav = 0.0
        if hungry > 0 and self._resource_protos():
            d_nav = min(np.linalg.norm(np.asarray(s_next, np.float32) - p)
                        for p in self._resource_protos())
            nav = self.feed_boost * float(np.exp(-d_nav / self.nav_sigma)) * hungry
        # 移动成本 (代谢)
        move_pen = 0.02 if a != self.n_actions - 1 else 0.0
        return -self.w_goal * dist_goal + self.w_val * val + nav \
            - move_pen + self.rng.rand() * self.temp

    def plan(self, s: Embed, goal: Embed, horizon: int = 5) -> list[int]:
        scores = [self._score(s, a, goal) for a in range(self.n_actions)]
        return [int(max(range(self.n_actions), key=lambda i: scores[i]))] * horizon

    def select_action(self, s: Embed, goal: Embed) -> int:
        scores = [self._score(s, a, goal) for a in range(self.n_actions)]
        return int(max(range(self.n_actions), key=lambda i: scores[i]))
