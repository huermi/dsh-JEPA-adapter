"""
JEPA AI 组件 — 核心类型系统 (core.py)
========================================
统一契约:
  - 表征空间: D = 768 (JEPA 统一空间; 符号/文本经投影到 Qwen 896 后再投影回)
  - 事件流:   八元组 Event (t, s, a, s_next, e1, e2, perf, died)
  - 参数域:   ParamKind 枚举 (Configurator 可调制的全部参数)
  - 配置:     ComponentConfig (每组件一段可调制配置)

所有组件必须基于本模块的类型定义, 保证可对接.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

import numpy as np

# 统一表征维度 (JEPA 空间)
D = 768
# Qwen 语义空间维度 (文本原型经投影可映射回 JEPA 空间)
QWEN_D = 896

# 观测: 图像 [C,H,W] / 桌面事件 / 文件内容 (统一 numpy 数组)
Obs = np.ndarray
# 表征: 768d 向量
Embed = np.ndarray


@dataclass
class Event:
    """八元组事件流 — 三环共享的单一数据契约.

    t       : tick 序号
    s       : 观测表征 [D]
    a       : 动作 id (int, 由 Planner 选择)
    s_next  : 执行后的下一状态表征 [D]
    e1      : 学习信号 (预测误差 / 惊讶度) — 内环
    e2      : 行动信号 (价值) — 中环
    perf    : 绩效 (采集/奖励/锚值) — 外环观测
    died    : 危急信号 (饥饿致死/危险结局) — 触发急刹车
    """
    t: int
    s: Embed
    a: int
    s_next: Embed
    e1: float
    e2: float
    perf: float
    died: bool = False

    @classmethod
    def blank(cls, t: int = 0) -> "Event":
        return cls(t=t, s=np.zeros(D, np.float32), a=0,
                   s_next=np.zeros(D, np.float32), e1=0.0, e2=0.0, perf=0.0)


class ParamKind(Enum):
    """Configurator (C10) 可调制的参数域 — 每个都有实证依据"""
    HOMEOSTASIS_RATE = auto()   # 内稳态速率 (饥饿/消耗): jpi11c 扫描 (太快固着/太慢失效)
    PERSISTENCE = auto()        # 目标坚持度: jpi1 (能量趋势切换有逃避困难目标风险)
    SURPRISE_THRESH = auto()    # 记忆写入/原型距离阈值: 三次手工阈值失效教训
    LEARNING_RATE = auto()      # 世界模型学习率: AdaJEPA 分层 lr
    EXPLORE_TEMP = auto()       # 探索-利用温度: 非平稳最优漂移 (jpi9)
    VALUE_BAND = auto()         # 价值水平带 (水平加速触发): jpi12 达标实证
    DEATH_PENALTY = auto()      # 死亡惩罚系数: jpi12 (一次死亡=5 采集当量)


@dataclass
class ComponentConfig:
    """组件可调制配置 — Configurator 通过 set_param 写入"""
    params: dict[ParamKind, float] = field(default_factory=dict)

    def get(self, kind: ParamKind, default: float = 0.0) -> float:
        return self.params.get(kind, default)

    def set(self, kind: ParamKind, value: float) -> None:
        self.params[kind] = value


# 组件间共享的"健康度统计" (外环观测源)
@dataclass
class HealthStats:
    """各环健康度 — Configurator 的输入信号集"""
    e1_mean: float = 0.0        # 内环: 平均预测误差
    e1_trend: float = 0.0       # 内环: 误差趋势 (下降=学习)
    e2_mean: float = 0.0        # 中环: 平均价值活性 (0 = 死寂)
    perf_mean: float = 0.0      # 外环: 平均绩效 (采集率/锚值)
    perf_trend: float = 0.0     # 外环: 绩效趋势 (爬山方向)
    death_count: int = 0        # 外环: 死亡次数 (急刹车计数)
    coverage: float = 0.0       # 中环: 探索覆盖率 (停滞检测)
    collapse: float = 0.0       # 内环: 表征坍缩度 (cosine 饱和检测)


def cosine(a: Embed, b: Embed) -> float:
    """表征相似度 (坍缩检测用: 全局平均 cosine 接近 1 = 坍缩)"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
