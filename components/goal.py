"""
C6 GoalGenerator — 目标生成 (中环)
===================================
契约: 目标 = 统一空间中的可塑向量, 从记忆/原型/插值/随机生成.

核心原则 (实证裁决):
  - 目标与价值对齐: 从记忆采样偏向高 E2 条目 (追"值得的"而非"任意的")
  - 坚持性预算: 切换 = (能量持续不降 > T) AND (探索预算耗尽)
    ("能量趋势决定切换"会逃避困难目标 — 前额叶对折扣抑制的类比)

接口:
  next_goal(memory, s_current) → Embed
  should_switch(e1_trend, t_hold) → bool

验收: jpi1 (坚持性预算稳定目标切换, 无逃避); p2_check.py
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .core import D, Embed, ParamKind


class GoalGenerator(ABC):
    """C6 目标生成接口"""

    @abstractmethod
    def next_goal(self, memory, s_current: Embed) -> Embed:
        """从记忆/原型生成新目标 (E2 价值加权)."""

    @abstractmethod
    def should_switch(self, e1_trend: float, t_hold: int) -> bool:
        """坚持性预算: 能量长期不降 + 超过坚持阈值才切换."""


class ValueAlignedGoal(GoalGenerator):
    """真实实现: E2 价值加权采样 + 坚持性预算 (能量趋势判断)"""

    def __init__(self, persistence: int = 80, min_progress: float = 1e-5):
        self.persistence = persistence            # ParamKind.PERSISTENCE
        self.min_progress = min_progress

    def set_param(self, kind: ParamKind, value: float) -> None:
        if kind == ParamKind.PERSISTENCE:
            self.persistence = int(value)

    def next_goal(self, memory, s_current: Embed) -> Embed:
        return memory.sample_goal(by_value=True)

    def should_switch(self, e1_trend: float, t_hold: int) -> bool:
        """切换条件: 超过坚持阈值 AND 能量不降 (e1_trend >= 0 表示误差未降).
        e1_trend > 0 = 世界越来越难预测 → 当前目标不可达 → 换.
        e1_trend < 0 = 还在学习 → 坚持."""
        return t_hold >= self.persistence and e1_trend >= -self.min_progress
