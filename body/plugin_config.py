"""
插件功能开关矩阵 (body/plugin_config.py)
=========================================
DeepSeek harness 插件的配置层: 把"额外功能"做成可开关选项.

开关清单 (每个开关 = 一个能力域的闸门):
  learning    在线学习 (权重梯度, AdaJEPA) — 任务经验是否改变权重
  memory      记忆写入 (惊讶门控) — 任务经验是否进入记忆
  sleep       睡眠巩固 (记忆回放) — 是否启用重放固化 (默认关, 手动/周期触发)
  tools       工具调用 — decide 是否产生 tool_calls
  respond_mode 响应模式: retrieval=检索式学会说话 (不依赖 LLM)
                         llm=桥接外部 LLM 表达 (保留路径)
  archive     存档 — 任务级快照 save/load

加载优先级: 显式构造 > 环境变量 (JEPA_*) > 默认值
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class PluginConfig:
    learning: bool = True           # 在线权重梯度
    memory: bool = True             # 记忆写入
    sleep: bool = False             # 睡眠巩固 (回放)
    sleep_epochs: int = 3
    sleep_lr_scale: float = 0.3
    sleep_prio_mix: float = 0.5
    tools: bool = True              # 工具调用
    respond_mode: str = "retrieval"  # retrieval | llm
    archive: bool = False           # 存档
    surprise_thresh: float = 0.3    # E1 触发工具调用的阈值
    max_memory: int = 200           # 记忆上限
    seed: int = 0
    # ── 自由学习 (探索) ──────────────────────────────────
    # 检索未命中时, 允许语义探索试错 (预算内, 温度衰减):
    # 探索 → 结果 → 学习 → 下次检索命中. P4c 实证: 记忆路径 33%→72%.
    explore: bool = True            # 未命中检索 → 语义探索尝试
    explore_budget: int = 3         # 每任务最大探索次数
    explore_decay: float = 0.9      # 每次探索后温度衰减 (×)
    # ── 基准测试 (benchmark) ────────────────────────────
    # 测试阶段: 关探索/网络 (模型只能用自己的内化知识 — 防作弊),
    # 纯检索+回忆应答; responder 容量加大 (大量学习准备不挤掉旧经验).
    benchmark_mode: bool = False    # 开启后: 探索关闭, 知识内化保留
    respond_cap: int = 300          # 回应经验上限 (学习准备期可调大)
    respond_min_sim: float = 0.45   # 回应检索阈值 (基准测试可调低 → 弱回忆也答)
    # ── AdaJEPA 软校准 (落点5, 测试时自适应 TTA) ─────────
    # 判定正确后把命中条目表征向查询微调 (小步长 EMA). stop-gradient 语义:
    # 只调条目不调查询 + margin 充足才校准 (防表征空间拉崩) —
    # "当下校准 × 长期沉淀" 合题: 相似问题自动受益 (AdaJEPA PushObj
    # 未见形状规划成功率翻倍的机制 — 软更新让相似输入泛化, 硬写入只精确匹配).
    soft_align: bool = True
    soft_align_alpha: float = 0.1   # 每步校准步长 (小 → 稳定)

    # ── 环境变量加载 (JEPA_LEARNING / JEPA_MEMORY / JEPA_SLEEP /
    #    JEPA_TOOLS / JEPA_RESPOND_MODE / JEPA_ARCHIVE / JEPA_PORT ...) ──
    @classmethod
    def from_env(cls) -> "PluginConfig":
        cfg = cls()
        env = os.environ

        def _bool(name: str, cur: bool) -> bool:
            v = env.get(name)
            if v is None:
                return cur
            return v.strip().lower() in ("1", "true", "yes", "on")

        cfg.explore = _bool("JEPA_EXPLORE", cfg.explore)
        cfg.benchmark_mode = _bool("JEPA_BENCHMARK", cfg.benchmark_mode)
        cfg.soft_align = _bool("JEPA_SOFT_ALIGN", cfg.soft_align)
        if env.get("JEPA_SOFT_ALIGN_ALPHA"):
            cfg.soft_align_alpha = float(env["JEPA_SOFT_ALIGN_ALPHA"])
        if env.get("JEPA_RESPOND_CAP"):
            cfg.respond_cap = int(env["JEPA_RESPOND_CAP"])
        cfg.learning = _bool("JEPA_LEARNING", cfg.learning)
        cfg.memory = _bool("JEPA_MEMORY", cfg.memory)
        cfg.sleep = _bool("JEPA_SLEEP", cfg.sleep)
        cfg.tools = _bool("JEPA_TOOLS", cfg.tools)
        cfg.archive = _bool("JEPA_ARCHIVE", cfg.archive)
        mode = env.get("JEPA_RESPOND_MODE", cfg.respond_mode).strip().lower()
        cfg.respond_mode = mode if mode in ("retrieval", "llm") else "retrieval"
        if env.get("JEPA_SLEEP_EPOCHS"):
            cfg.sleep_epochs = int(env["JEPA_SLEEP_EPOCHS"])
        if env.get("JEPA_SURPRISE"):
            cfg.surprise_thresh = float(env["JEPA_SURPRISE"])
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Optional[str] = None) -> str:
        s = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)
        return s

    @classmethod
    def from_json(cls, path: str) -> "PluginConfig":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

    def describe(self) -> str:
        """人类可读开关状态 (供 /v1/status 与诊断)"""
        return (f"learning={'on' if self.learning else 'off'} | "
                f"memory={'on' if self.memory else 'off'} | "
                f"sleep={'on' if self.sleep else 'off'} | "
                f"tools={'on' if self.tools else 'off'} | "
                f"respond={self.respond_mode} | "
                f"archive={'on' if self.archive else 'off'}")
