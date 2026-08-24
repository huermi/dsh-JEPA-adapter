"""
C3 EnergySystem — 能量系统 E1/E2 分离 (内环/中环)
==================================================
契约: E1 (学习信号) 与 E2 (行动信号) 分属两环, 不可混乘.

  E1 = ||Δŝ - Δs||²        认识论惊讶 — 唯一学习触发器 (训练/巩固/记忆写入)
  E2 = f(e1_history, anchor_value)  行动理由 = 可学习性 + 锚 (锚来自 E1 之外)

核心裁决 (实证):
  - 纯 E1 驱动会死寂 (jpi1: 静态世界 WAIT 29.1%, E2 终值 0.0001)
  - 锚必须前瞻 (value(s_next)) + 内稳态 (消耗-补充循环) + 外环调制
  - 两级能量分离是 DCA InfoDrives 三因子约化的替代 (三环架构根基)

接口:
  update(e1, anchor_value) → e2
  get_stats() → (e1_mean, e1_trend, e2_mean)   (供外环观测)

验收: jpi1_simulator.py 2×2 对照 (死寂/活性/WAIT)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

from .core import Embed, ComponentConfig


class EnergySystem(ABC):
    """C3 能量系统接口"""

    @abstractmethod
    def update(self, e1: float, anchor_value: float = 0.0) -> float:
        """每 tick: 由 E1 与锚更新 E2. 返回 E2."""

    @abstractmethod
    def get_stats(self) -> tuple[float, float, float]:
        """返回 (e1_mean, e1_trend, e2_mean) 供 Configurator 观测."""


class CuriosityEnergy(EnergySystem):
    """默认实现: Curiosity-Critic 式误差-基线分离 + 锚"""

    def __init__(self, baseline_win: int = 500):
        self.e1_hist = deque(maxlen=2000)
        self.e2_hist = deque(maxlen=2000)
        self.baseline_win = baseline_win
        self.e1_baseline = 0.0

    def update(self, e1: float, anchor_value: float = 0.0) -> float:
        import numpy as np
        self.e1_hist.append(e1)
        if len(self.e1_hist) >= 50:
            self.e1_baseline = float(np.mean(list(self.e1_hist)[-self.baseline_win:]))
        # E2 = 可学习性 (误差相对基线的改进) + 锚
        learnability = max(0.0, self.e1_baseline - e1)
        e2 = learnability + anchor_value
        self.e2_hist.append(e2)
        return e2

    def get_stats(self) -> tuple[float, float, float]:
        import numpy as np
        e1_mean = float(np.mean(self.e1_hist)) if self.e1_hist else 0.0
        e2_mean = float(np.mean(self.e2_hist)) if self.e2_hist else 0.0
        trend = 0.0
        if len(self.e1_hist) >= 200:
            first = np.mean(list(self.e1_hist)[-200:-100])
            last = np.mean(list(self.e1_hist)[-100:])
            trend = last - first
        return e1_mean, trend, e2_mean
