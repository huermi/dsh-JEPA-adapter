"""
C9 Distiller — 蒸馏器 (中环)
============================
契约: 编译 System-2 进 System-1 — 慢规划 (大预算) → 快策略 (小预算).

核心原则 (实证 jpi8):
  - 软标签蒸馏 + 恰好够的样本量 (相关性 ≈0.99 时最优)
  - 样本量非单调: 太少学不到 (20 → -11.9pp), 太多过拟合 (简单任务退化)
  - 蒸馏不能创造慢模型没有的知识, 只能压缩传递 (n=5 直接已最优则无增益)

验收: distill_check.py (BB 场景, 慢 80 → 快 15, +59.4pp 量级)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class Distiller(ABC):
    """C9 蒸馏器接口"""

    @abstractmethod
    def distill(self, slow_model: Any, data: Any, n_samples: int, **kw) -> Any:
        """慢模型经验 → 快模型 (软标签, 适量样本)."""

    @abstractmethod
    def evaluate(self, fast_model: Any, test_pool: Any, **kw) -> float:
        """效率评估 (引导命中率等)."""


class SoftLabelDistiller(Distiller):
    """真实实现: 软标签蒸馏 (jpi8 迁移)
    通用化: 通过 student_factory / step_fn / predict_fn 注入, 默认适配
    jpi4 风格 SymbolPredictor (predict / step 接口).
    """

    def __init__(self):
        self.reports: list[dict] = []

    def distill(self, slow_model: Any, data: Any, n_samples: int,
                seed: int = 0,
                student_factory: Optional[Callable[[], Any]] = None,
                step_fn: Optional[Callable[[Any, Any, float], None]] = None,
                predict_fn: Optional[Callable[[Any, Any], float]] = None,
                **kw) -> Any:
        """软标签蒸馏:
        slow_model: 慢模型 (Mode-2), 有 predict 接口
        data      : 特征集 [N, d] (与慢模型输入同空间)
        n_samples : 蒸馏样本数 (实证: 需要"恰好够", 20/60/100 中按任务选)
        返回快模型 (Mode-1)."""
        import numpy as np
        rng = np.random.RandomState(seed)
        fast = student_factory() if student_factory else kw.get("student")
        if fast is None:
            raise ValueError("需要 student_factory 或 student")
        _pred = predict_fn or (lambda m, x: m.predict(x))
        _step = step_fn or (lambda m, x, t: m.step(x, t))
        idx = rng.choice(len(data), n_samples, replace=False)
        for i in idx:
            soft = _pred(slow_model, data[i])
            _step(fast, data[i], soft)
        self.reports.append({"n_samples": n_samples, "method": "soft_label",
                             "n_total": len(data)})
        return fast

    def evaluate(self, fast_model: Any, test_pool: Any, **kw) -> float:
        """评估快模型引导效率. test_pool: (feats, steps_log, budget, seed)."""
        feats, steps_log = test_pool[0], test_pool[1]
        budget = kw.get("budget", 15)
        seed = kw.get("seed", 2)
        eval_fn = kw.get("eval_fn")
        if eval_fn is not None:
            return float(eval_fn(fast_model, feats, steps_log, budget, seed))
        return 0.0
