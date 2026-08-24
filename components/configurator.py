"""
C10 Configurator — 元认知外环
==============================
契约: 系统自身参数的异稳态调节器 (AI 的"发烧") — 监测各环健康度 → 调制设定点.

机制 (jpi12 达标实证 v5):
  1. 爬山: 绩效 < band → 梯度定位单峰 (绩效-率曲线单峰, 不能假设方向)
  2. 水平加速: 绩效 > hi_band → 强制上调设定点 (异稳态加码)
  3. 死亡急刹车: died → rate ×= panic_scale 立即回调 (疼痛反射)

达标证据 (jpi12): 慢交替 +2.97 / 中交替 +1.20 (超最优固定, 5/5 显著);
  全漂移 99.8% 最优; 无漂移优于默认 48%; 异稳态机制成立.

物理边界 (诚实标注):
  - 观测噪声: 稀疏采集无法精确定位峰值 (保证优于默认, 不保证精确最优)
  - 时间尺度: 切换快于观测窗口 → 跟踪退化 (解法: 与世界模型预测耦合)

接口:
  observe(perf, t, died) → None        (每 tick 观测, 更新调制参数)
  get_params() → dict[ParamKind, float]  (输出到各组件)

验收: jpi12_final_validation.py (C1-C5 全过)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

from .core import ParamKind, ComponentConfig


class Configurator(ABC):
    """C10 元认知外环接口"""

    @abstractmethod
    def observe(self, perf: float, t: int, died: bool = False) -> None:
        """每 tick: 观测绩效/危急信号, 更新设定点."""

    @abstractmethod
    def get_config(self) -> ComponentConfig:
        """输出调制后的参数配置 (写入各组件)."""


class PerfConfigurator(Configurator):
    """达标实现: 爬山 + 水平加速 + 死亡急刹车 (jpi12 v5)"""

    def __init__(self, base_rate: float = 0.001, win: int = 300,
                 check_every: int = 100, step: float = 0.002,
                 lo: float = 0.0005, hi: float = 0.03,
                 hi_band: float = 9.0, dead_zone: float = 1.5,
                 panic_scale: float = 0.6):
        self.rate = base_rate
        self.win = win
        self.check_every = check_every
        self.step = step
        self.lo, self.hi = lo, hi
        self.hi_band = hi_band
        self.dead_zone = dead_zone
        self.panic_scale = panic_scale
        self.perf_hist: list[float] = []
        self.direction = 1.0
        self.reversals = 0
        self.panics = 0
        self.last_anchor: Optional[float] = None
        self._config = ComponentConfig()
        self._config.set(ParamKind.HOMEOSTASIS_RATE, base_rate)

    def observe(self, perf: float, t: int, died: bool = False) -> None:
        import numpy as np
        # 死亡急刹车: 疼痛反射 (不等窗口)
        if died:
            self.rate = float(np.clip(self.rate * self.panic_scale, self.lo, self.hi))
            self.direction = -1.0
            self.panics += 1
            self._config.set(ParamKind.HOMEOSTASIS_RATE, self.rate)
            return
        self.perf_hist.append(perf)
        if t % self.check_every != 0 or len(self.perf_hist) < self.win:
            return
        cur = float(np.mean(self.perf_hist[-150:])) * 1000.0
        # 水平加速: 绩效高 = 快节奏 → 上调
        if cur > self.hi_band:
            self.direction = 1.0
        else:
            # 爬山: 梯度定位单峰 (带死区)
            if self.last_anchor is not None:
                delta = cur - self.last_anchor
                if delta < -self.dead_zone:
                    self.direction = -self.direction
                    self.reversals += 1
        self.last_anchor = cur
        self.rate = float(np.clip(self.rate + self.direction * self.step, self.lo, self.hi))
        self._config.set(ParamKind.HOMEOSTASIS_RATE, self.rate)

    def get_config(self) -> ComponentConfig:
        return self._config
