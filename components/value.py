"""
C4 ValueSystem — 价值系统 (中环)
=================================
契约: 预测未来价值 (Critic) + 提供行动理由 (锚三重结构).

锚三重 (实证裁决 jpi1):
  1. 前瞻: 价值作用于预测的未来状态 value(s_next) (Friston 期望自由能)
  2. 内稳态: 消耗-补充循环 (饥饿-进食), 非静态价值场 (静态锚 = 固着 99.7%)
  3. 外环调制: 内稳态速率由 Configurator 调制 (太快固着/太慢失效)

Critic (LeCun Cost 模块完整形态):
  可训练, 从记忆 (状态, 实际成本) 对学习预测未来成本
  (稀疏奖励下记忆回放 > 即时训练, jpi10: +16.7pp)

验收: jpi1 锚形态对照; jpi2 可训练 Critic; p2_check.py 中环闭环
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .core import D, Embed, ParamKind


class ValueSystem(ABC):
    """C4 价值系统接口"""

    @abstractmethod
    def tick(self) -> bool:
        """内稳态需求推进 (每 tick). 返回是否死亡 (危急信号)."""

    @abstractmethod
    def predict_cost(self, s_next: Embed) -> float:
        """预测未来成本 (Critic)."""

    @abstractmethod
    def anchor(self, s: Embed, demand: float | None = None) -> float:
        """内稳态锚: 需求 × 满足度 (前瞻价值)."""

    @abstractmethod
    def try_feed(self, s: Embed) -> bool:
        """在资源旁进食 (消耗-补充). 返回是否进食."""

    @abstractmethod
    def register_resource(self, proto: Embed) -> None:
        """注册资源原型 (感知/环境注入)."""

    @abstractmethod
    def train_from_memory(self, states: list[Embed], costs: list[float]) -> None:
        """从记忆 (状态, 实际成本) 对学习 Critic."""


class HomeostaticValue(ValueSystem):
    """真实实现: 内稳态锚 v2 (jpi11/12 迁移)
    hunger 消耗-补充循环 + 资源原型 closeness + 线性 Critic (记忆学习)
    """
    def __init__(self, sigma: float = 2.2, feed_thresh: float = 0.55,
                 feed_boost: float = 0.5, hunger_rate: float = 0.001,
                 death_delay: int = 300, w_pain: float = 0.3,
                 critic_lr: float = 0.01, seed: int = 0):
        self.sigma = sigma
        self.feed_thresh = feed_thresh
        self.feed_boost = feed_boost
        self.hunger_rate = hunger_rate        # ParamKind.HOMEOSTASIS_RATE
        self.death_delay = death_delay
        self.w_pain = w_pain
        self.critic_lr = critic_lr
        self.hunger = 0.0
        self.death_counter = 0
        self.resource_protos: list[Embed] = []
        self.critic_w = np.zeros(D, dtype=np.float32)
        self.n_feeds = 0
        self.n_deaths = 0
        self.rng = np.random.RandomState(seed)

    # ── 内稳态动力学 ─────────────────────────────────────
    def tick(self) -> bool:
        """hunger 增长 + 死亡检测"""
        self.hunger = min(1.0, self.hunger + self.hunger_rate)
        if self.hunger >= 0.99:
            self.death_counter += 1
            if self.death_counter >= self.death_delay:
                self.n_deaths += 1
                self.death_counter = 0
                self.hunger = 0.0
                return True                    # 死亡 → 急刹车信号
        else:
            self.death_counter = 0
        return False

    def set_param(self, kind: ParamKind, value: float) -> None:
        if kind == ParamKind.HOMEOSTASIS_RATE:
            self.hunger_rate = value

    # ── 资源原型 ─────────────────────────────────────────
    def register_resource(self, proto: Embed) -> None:
        """注册资源原型 (同一表征空间的向量)"""
        p = np.asarray(proto, np.float32)
        for q in self.resource_protos:
            if np.linalg.norm(p - q) < 1e-3:
                return
        self.resource_protos.append(p)
        if len(self.resource_protos) > 8:
            self.resource_protos.pop(0)

    def closeness(self, s: Embed) -> float:
        """距最近资源原型的接近度 (exp 衰减)"""
        if not self.resource_protos:
            return 0.0
        d = min(np.linalg.norm(np.asarray(s, np.float32) - p) for p in self.resource_protos)
        return float(np.exp(-d / self.sigma))

    def try_feed(self, s: Embed) -> bool:
        """在资源旁进食: 需求 > 0.3 且接近资源"""
        if self.closeness(s) > self.feed_thresh and self.hunger > 0.3:
            self.hunger = max(0.0, self.hunger - self.feed_boost)
            self.n_feeds += 1
            return True
        return False

    # ── 价值 ─────────────────────────────────────────────
    def anchor(self, s: Embed, demand: float | None = None) -> float:
        """前瞻内稳态价值: 接近资源 → 正; 饿且远离 → 疼痛"""
        demand = self.hunger if demand is None else demand
        c = self.closeness(s)
        return c * demand - self.w_pain * demand * (1.0 - c)

    def predict_cost(self, s_next: Embed) -> float:
        return float(max(0.0, -float(np.asarray(s_next, np.float32) @ self.critic_w)))

    def train_from_memory(self, states: list[Embed], costs: list[float]) -> None:
        for s, c in zip(states[-50:], costs[-50:]):
            pred = float(np.asarray(s, np.float32) @ self.critic_w)
            err = c - pred
            self.critic_w += self.critic_lr * np.clip(np.asarray(s, np.float32) * err, -0.1, 0.1)
