"""JEPA AI 组件包 — 三环架构可组装组件.

组件清单:
  C1  PerceptionEncoder  感知编码器     (内环)
  C2  WorldModel         世界模型预测器   (内环)
  C3  EnergySystem       能量系统 E1/E2  (内环/中环)
  C4  ValueSystem        价值系统+锚三重  (中环)
  C5  MemorySystem       记忆系统 (桥梁)  (中环)
  C6  GoalGenerator      目标生成         (中环)
  C7  Planner            规划器           (中环)
  C8  SymbolLayer        符号层           (中环)
  C9  Distiller          蒸馏器           (中环)
  C10 Configurator       元认知外环       (外环)
  system.JepaAgent       三环组装器

设计原则: 接口薄、契约清晰、默认实现引用已有实证代码、独立可测.
"""
from .core import D, QWEN_D, Event, ParamKind, ComponentConfig, HealthStats, Obs, Embed

__version__ = "0.1.0"
