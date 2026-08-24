"""
C5 MemorySystem — 记忆系统 (桥梁)
=================================
契约: 记忆生命周期 — 写 (E1 惊讶门控) → 存 (LTM) → 读 (E2 价值加权)
      → 巩固 (重放) → 遗忘 (显著性衰减).

核心原则 (实证):
  - 写入靠 E1 (记下"不懂的"), 读取靠 E2 (要"值得的") — 价值权重对齐两端
  - 手工阈值全部失效 (0.3/0.7/0.9 三次教训) → 自适应分位数门控
  - 稀疏奖励下回放 > 即时训练 (jpi10: +16.7pp)

接口:
  write(s, e1, e2)                 (surprise 门控写入)
  sample_goal(by_value=True) → Embed   (E2 价值加权采样, GoalGen 用)
  replay(n) → list[(s, a, s_next)]    (最近回放, 在线适应用)
  consolidate()                    (重放巩固)
  get_prototypes() → list[Embed]   (自适应原型)

验收: jpi1/jpi2 (门控失效→自适应修复); jpi10 (回放对比)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

from .core import D, Embed, ParamKind


class MemorySystem(ABC):
    """C5 记忆系统接口"""

    @abstractmethod
    def write(self, s: Embed, e1: float, e2: float) -> bool:
        """E1 惊讶门控写入. 返回是否写入."""

    @abstractmethod
    def sample_goal(self, by_value: bool = True) -> Embed:
        """E2 价值加权采样目标 (记忆空则随机)."""

    @abstractmethod
    def replay(self, n: int) -> list[tuple[Embed, int, Embed]]:
        """最近 n 条 (s, a, s_next) 回放."""

    @abstractmethod
    def get_prototypes(self) -> list[Embed]:
        """自适应原型 (距离分位门控)."""


class AdaptiveMemory(MemorySystem):
    """默认实现: 自适应分位数门控 + 价值加权采样"""

    def __init__(self, cap: int = 200, surprise_q: float = 0.85,
                 proto_q: float = 0.9, seed: int = 0):
        import numpy as np
        self.cap = cap
        self.surprise_q = surprise_q
        self.proto_q = proto_q
        self.items: list[tuple[Embed, float, float]] = []   # (s, e1, e2)
        self.transitions: deque[tuple[Embed, int, Embed, float]] = deque(maxlen=500)
        self.prototypes: list[Embed] = []
        self.e1_hist = deque(maxlen=500)
        self.dist_hist = deque(maxlen=1000)
        self.rng = np.random.RandomState(seed)

    def write(self, s: Embed, e1: float, e2: float) -> bool:
        self.e1_hist.append(e1)
        if len(self.e1_hist) < 50:
            return False
        thresh = float(__import__("numpy").percentile(list(self.e1_hist),
                                                       self.surprise_q * 100))
        if e1 > thresh and len(self.items) < self.cap:
            self.items.append((s.copy(), e1, e2))
            # 自适应原型
            if not self.prototypes:
                self.prototypes.append(s.copy())
            else:
                import numpy as np
                d = min(np.linalg.norm(s - p) for p in self.prototypes)
                self.dist_hist.append(d)
                if len(self.dist_hist) >= 30:
                    t = float(np.percentile(list(self.dist_hist), self.proto_q * 100))
                    if d > t and len(self.prototypes) < 20:
                        self.prototypes.append(s.copy())
            return True
        return False

    def sample_goal(self, by_value: bool = True) -> Embed:
        import numpy as np
        if not self.items:
            return self.rng.randn(D).astype(np.float32) * 0.5
        if by_value:
            # 价值加权: E2 可为负 (疼痛驱动), 权重必须非负 → max(.,0)+0.1
            weights = np.array([max(i[2], 0.0) + 0.1 for i in self.items])
            weights = weights / weights.sum()
            idx = self.rng.choice(len(self.items), p=weights)
        else:
            idx = self.rng.randint(len(self.items))
        return self.items[idx][0].copy()

    def replay(self, n: int) -> list[tuple[Embed, int, Embed]]:
        """最近 n 条 (s, a, s_next) 回放 (兼容旧接口, 去掉 e1)."""
        return [(t[0], t[1], t[2]) for t in list(self.transitions)[-n:]]

    def add_transition(self, s: Embed, a: int, s_next: Embed,
                       e1: float = 1.0) -> None:
        """记录完整转移 (s, a, s_next, 惊讶度 e1).
        e1 = 睡眠巩固的优先级依据 (高惊讶 = 重要经验优先重放)."""
        self.transitions.append((s.copy(), a, s_next.copy(), float(e1)))

    def get_transitions(self) -> list[tuple[Embed, int, Embed, float]]:
        """全部转移 (含 e1), 供睡眠巩固优先级采样."""
        return list(self.transitions)

    def get_prototypes(self) -> list[Embed]:
        return self.prototypes
