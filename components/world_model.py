"""
C2 WorldModel — 世界模型预测器 (内环)
=====================================
契约: 预测潜空间状态变化 P(s, a, δ) → Δŝ; 提供能量 E1.

接口:
  predict_delta(s:[D], a:int, delta:int=1) → Δŝ:[D]
  energy(s:[D], a:int, s_next:[D]) → e1:float   (预测误差)
  step(s, a, s_next) → e1                        (AdaJEPA 1 步梯度在线更新)
  set_lr(lr) / adapt_from_buffer(events)         (分层 lr / recent-N 回放)

实现来源:
  - jepa_base.JEPA (predictor) / DCA predictor.py (AdaJEPA 配方)
  - LeWorldModel 两 loss (预测 MSE + SIGReg 高斯正则)
  - 时间分层: 多尺度 δ (Micro-World 骨架, P2 接入)

验收标准:
  - E1 收敛 (jpi1: 静态世界 E1 降 83%)
  - 分布偏移检测 (jpi2: E1 尖峰 +50~178%)
  - 防坍缩 (cosine 不饱和, core.cosine 检测)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .core import D, Embed, Event, ComponentConfig


class WorldModel(ABC):
    """C2 世界模型接口"""

    @abstractmethod
    def predict_delta(self, s: Embed, a: int, delta: int = 1) -> Embed:
        """预测执行动作 a 后 δ 步的状态变化 (残差预测)."""

    @abstractmethod
    def energy(self, s: Embed, a: int, s_next: Embed) -> float:
        """E1 = ||Δŝ - Δs||² (潜空间 L2)."""

    @abstractmethod
    def step(self, s: Embed, a: int, s_next: Embed) -> float:
        """在线学习: 1 步梯度 (AdaJEPA 配方: 最后层优先, 分层 lr). 返回 E1."""

    @abstractmethod
    def adapt_from_buffer(self, events: list[Event]) -> None:
        """从 recent-N 回放缓冲适应 (分布偏移时校准)."""

    @abstractmethod
    def set_lr(self, lr: float) -> None:
        """Configurator 调制学习率 (ParamKind.LEARNING_RATE)."""


class ResidualWorldModel(WorldModel):
    """真实实现: AdaJEPA 配方残差预测器 (DCA predictor / jpi1 迁移)
    结构: [s, a_onehot] → MLP(2 层 ReLU) → Δŝ (残差, 不预测绝对状态)
    在线: 1 步梯度, 分层 lr (预测器高 / 稳定), recent-N 回放适应
    """
    def __init__(self, s_dim: int = D, h_dim: int = 256, n_actions: int = 5,
                 lr_pred: float = 0.05, lr_enc: float = 0.001, seed: int = 0):
        import numpy as np
        self.s_dim, self.h_dim, self.n_actions = s_dim, h_dim, n_actions
        rng = np.random.RandomState(seed)
        # He 初始化: 按 fan-in 缩放, 防高维输入下 x@W 范数爆炸 (AdaJEPA 配方)
        scale = np.sqrt(2.0 / (s_dim + n_actions))
        self.W1 = rng.randn(s_dim + n_actions, h_dim).astype(np.float32) * scale
        self.W2 = np.zeros((h_dim, s_dim), dtype=np.float32)
        self.lr_pred, self.lr_enc = lr_pred, lr_enc

    def _x(self, s: Embed, a: int) -> np.ndarray:
        import numpy as np
        a_oh = np.eye(self.n_actions)[a].astype(np.float32)
        return np.concatenate([np.asarray(s, np.float32), a_oh])

    def predict_delta(self, s: Embed, a: int, delta: int = 1) -> Embed:
        import numpy as np
        x = self._x(s, a)
        h = np.maximum(0, x @ self.W1)
        d = h @ self.W2
        n = np.linalg.norm(d)
        return d if n < 5 else d * 5 / n

    def energy(self, s: Embed, a: int, s_next: Embed) -> float:
        import numpy as np
        d = self.predict_delta(s, a)
        return float(np.mean((d - (np.asarray(s_next, np.float32) - np.asarray(s, np.float32))) ** 2))

    def step(self, s: Embed, a: int, s_next: Embed) -> float:
        import numpy as np
        x = self._x(s, a)
        h = np.maximum(0, x @ self.W1)
        pred = h @ self.W2
        target = np.asarray(s_next, np.float32) - np.asarray(s, np.float32)
        err = pred - target
        e1 = float(np.mean(err ** 2))
        # 归一化梯度下降: 除以激活范数², 消除高维输入下有效步长失控
        # (朴素 SGD 的 lr × ||h||² ≈ 4 导致过冲振荡; 归一化后步长与激活尺度无关)
        h2 = float(np.dot(h, h)) + 1e-6
        x2 = float(np.dot(x, x)) + 1e-6
        self.W2 -= self.lr_pred * np.outer(h, err) / h2
        dH = (err @ self.W2.T) * (h > 0)
        self.W1 -= self.lr_enc * np.outer(x, dH) / x2
        return e1

    def adapt_from_buffer(self, events: list[Event]) -> None:
        """recent-N 回放适应 (分布偏移校准, AdaJEPA 配方)"""
        for ev in events[-5:]:
            self.step(ev.s, ev.a, ev.s_next)

    def set_lr(self, lr: float) -> None:
        self.lr_pred = lr


class DummyWorldModel(WorldModel):
    """占位实现: 线性残差预测 (组装自检用)"""

    def __init__(self, seed: int = 0):
        import numpy as np
        self.rng = np.random.RandomState(seed)
        self.W = self.rng.randn(D, D).astype(np.float32) * np.sqrt(2.0 / D)
        self.lr = 0.05

    def predict_delta(self, s: Embed, a: int, delta: int = 1) -> Embed:
        import numpy as np
        return np.tanh(s @ self.W) * 0.1

    def energy(self, s: Embed, a: int, s_next: Embed) -> float:
        import numpy as np
        d = self.predict_delta(s, a)
        return float(np.mean((d - (s_next - s)) ** 2))

    def step(self, s: Embed, a: int, s_next: Embed) -> float:
        import numpy as np
        e1 = self.energy(s, a, s_next)
        target = s_next - s
        pred = self.predict_delta(s, a)
        err = pred - target
        self.W -= self.lr * np.clip(np.outer(s, err), -0.5, 0.5)
        return e1

    def adapt_from_buffer(self, events: list[Event]) -> None:
        for ev in events[-5:]:
            self.step(ev.s, ev.a, ev.s_next)

    def set_lr(self, lr: float) -> None:
        self.lr = lr
