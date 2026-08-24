"""
真实身体 — 底层模型内核 (body/kernel.py)
=========================================
JepaBody: 把三环系统 (JepaAgent) 封装成"底层模型"形式,
对外暴露标准模型接口 + 工具调用, 可接入各家 harness.

设计定位:
  - 不是产品, 是内核: 提供 perceive/predict/decide/learn/tool 五类原子能力
  - 接口契约稳定: harness 只依赖本文件的公开接口, 内部实现可替换
  - 工具调用: OpenAI function-calling 风格 (tool_calls → 执行 → 回传结果)
  - 触发信号: E1 惊讶度 — 世界不可预测时模型"知道自己需要外部信息",
    主动发起工具调用 (这正是之前实证的"惊讶→记忆/学习"机制的外化)

接口一览:
  perceive(obs) -> z                    感知编码 (任意观测 → 768d)
  predict(z, a, delta=1) -> z'          世界模型预测
  energy_of(z) -> float                 E1 惊讶度
  decide(obs) -> dict{action, tool_calls}  决策 (高惊讶时带工具调用)
  learn(obs, a, obs_next, perf, died, tool_result=None)  在线学习
  register_tool(name, fn, desc, schema) 工具注册
  call_tool(name, args) -> str          工具执行
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Optional

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.core import D, Embed, Obs, Event
from components.system import JepaAgent, build_default_system
from plugin_config import PluginConfig


# ─── 情境向量空间 (三槽结构化) ────────────────────────────
# 情境 = concat(enc(task), enc(last_call), enc(last_result)), 每槽 384d
# (MiniLM 语义编码; 哈希袋兜底时也用 384d 三槽, 保证维度恒定 1152).
# 教学与运行时共用 _situation_vec 构造 → 检索空间一致.
SIT_SLOT = 384
SIT_D = SIT_SLOT * 3          # 1152: 统一情境维度 (call_mem/responder/工具预测器)
N_TOOLS_MAX = 64              # 工具 action 索引上限 (dsh 25 工具 + 余量)


# ─── 工具注册表 ─────────────────────────────────────────────
class ToolRegistry:
    """工具注册 + OpenAI 格式导出 + 调用调度"""

    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(self, name: str, fn: Callable, description: str,
                 parameters: Optional[dict] = None) -> None:
        """注册工具. parameters = JSON Schema (OpenAI 风格, 可空)"""
        self._tools[name] = {
            "fn": fn, "description": description,
            "parameters": parameters or {"type": "object", "properties": {}},
        }

    def list(self) -> list[dict]:
        """OpenAI tools 格式导出 (供 harness)"""
        return [{
            "type": "function",
            "function": {
                "name": name,
                "description": t["description"],
                "parameters": t["parameters"],
            }
        } for name, t in self._tools.items()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def call(self, name: str, args: dict = None) -> str:
        """执行工具, 返回字符串结果 (供模型/记忆消费)"""
        if name not in self._tools:
            return f"ERROR: unknown tool '{name}'"
        try:
            res = self._tools[name]["fn"](**(args or {}))
            return str(res)
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"


# ─── 文本 → 观测编码 (轻量, 正式版换 Qwen 编码器) ──────────
def text_to_obs(text: str, dim: int = D) -> Obs:
    """消息文本 → 观测向量. 最小实现: 确定性哈希袋 (词哈希累加).
    正式版: Qwen 编码器 (qwen_to_jepa 已验证 zero-shot 66%).
    dim 可调: 情境槽位用 384 (与 MiniLM 三槽拼接对齐), 记忆感知默认 D."""
    vec = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % (2 ** 31)
        rng = np.random.RandomState(h)
        vec += rng.randn(dim).astype(np.float32) * 0.01
    n = np.linalg.norm(vec)
    return (vec / (n + 1e-9)) if n > 1e-9 else vec


# ─── 底层模型内核 ───────────────────────────────────────────
class JepaBody:
    """真实身体内核: 模型接口 + 工具调用 + 学习闭环"""

    def __init__(self, agent: Optional[JepaAgent] = None,
                 surprise_thresh: float = 0.3, seed: int = 0,
                 config: Optional["PluginConfig"] = None):
        self.agent = agent or build_default_system(seed=seed)
        self.tools = ToolRegistry()
        self.config = config or PluginConfig(seed=seed)
        self.surprise_thresh = self.config.surprise_thresh  # E1 高于此 → 触发工具调用
        self.tool_embeds: dict[str, Embed] = {}  # 工具名 -> 描述表征 (统一空间检索)
        self.t = 0
        self.stats = {"tool_calls": 0, "tool_errors": 0, "decisions": 0}
        self._mm = None                          # 多模态感知 (惰性启用)
        # 回应学习器: 让模型学会输出字符 (检索式, 不依赖 LLM)
        from respond_learner import RespondLearner
        self.responder = RespondLearner(
            min_sim=getattr(self.config, "respond_min_sim", 0.45),
            cap=getattr(self.config, "respond_cap", 300), seed=seed,
            soft_align=getattr(self.config, "soft_align", True),
            soft_align_alpha=getattr(self.config, "soft_align_alpha", 0.1))
        # 调用模式记忆: 学会完整工具调用 (情境→工具+参数), 支撑多步任务
        from call_memory import CallMemory
        self.call_mem = CallMemory(seed=seed)
        # 循环防护: 同一工具连续调用 ≥3 次 → 强制收尾 (防哈希袋情境粘滞死循环)
        self._recent_tools: list[str] = []
        # respond 是模型"内部动作" (输出字符), 不发给 harness 执行 →
        # 工具选择时排除, chat_completion 中转为 content
        self._internal_tools: set[str] = {"respond"}
        # ── 语义编码 (MiniLM 惰性接入, 失败回退哈希袋) ──
        self._mini_enc = None
        try:
            from mini_encoder import get_encoder
            self._mini_enc = get_encoder()
        except Exception:
            pass
        self._text_encoder = None            # 语义文本编码器 (Qwen), 惰性接入
        # ── 工具 → 潜在空间行动 (预测式工具选择骨架) ──
        # 每个工具分配一个 action 索引 (工具世界模型 one-hot 列);
        # 工具效果原型 v_t = EMA(Δz) — 隐式学行动编码 (方案 a).
        self._tool_action: dict[str, int] = {}
        self._action_tool: dict[int, str] = {}
        self._next_action = 0
        self._tool_effect: dict[str, Embed] = {}   # 工具 -> 效果原型 (SIT_D)
        self._effect_alpha = 0.3                   # EMA 系数
        self._tool_wm = None                       # 工具世界模型 (惰性构造)
        # ── 自由学习 (探索): 检索未命中 → 语义试错 → 结果学习 ──
        self.rng = np.random.RandomState(getattr(self.config, "seed", 0))
        self._explore_count = 0                    # 当前任务已探索次数
        self._explore_temp = 1.0                   # 探索温度 (衰减)
        self._explore_success = False              # 探索已获有效信息 → 本任务停探
        self._last_block = "none"                  # 最近受阻类型 (落点1 诊断):
        self._tool_use_counter: dict[str, int] = {}  # 工具调用分布 (落点4: 价值萎缩检测)
        self.register_tool(
            "respond", lambda text="": text,
            "respond to the user with text output",
            {"type": "object",
             "properties": {"text": {"type": "string"}},
             "required": ["text"]})

    # ── 多模态感知 (P4a 接入) ────────────────────────────
    def enable_multimodal(self, image_enc=None, text_enc=None) -> None:
        """启用全模态感知 (MultimodalPerception): 图片/文本/音频/视频 → 768d.
        惰性: 默认构造保持轻量 (Dummy), 调用本方法才加载.
        image_enc 提供真实图片编码器 (如 timm JepaPerception);
        text_enc 提供语义文本编码器 (如 QwenPerception)"""
        from multimodal import MultimodalPerception
        self._mm = MultimodalPerception(image_enc, text_enc)
        self.agent.perception = self._mm          # 替换感知层

    @property
    def multimodal(self) -> bool:
        return self._mm is not None

    # ── 核心模型接口 ──────────────────────────────────────
    def set_text_encoder(self, enc) -> None:
        """接入语义文本编码器 (QwenPerception 等): 感知/工具选择统一语义空间.
        哈希袋语义太弱 → 真实 harness 工具集上乱选 (interrupt_agent 事故)."""
        self._text_encoder = enc

    def ensure_semantic(self) -> bool:
        """尝试加载 MiniLM 语义编码 (插件启动时调用; 失败静默回退哈希袋).
        返回是否可用."""
        if self._mini_enc is None:
            return False
        try:
            return self._mini_enc.ensure_loaded()
        except Exception:
            return False

    def _enc_text(self, text: str) -> Optional[Embed]:
        """统一语义编码入口: 外部编码器 (Qwen) 优先, MiniLM 次之.
        模型不可用 → None (调用方哈希袋兜底)."""
        if not text:
            return None
        if self._text_encoder is not None:
            try:
                v = self._text_encoder.encode(text)
                if v is not None:
                    return np.asarray(v, np.float32)
            except Exception:
                pass
        if self._mini_enc is not None:
            try:
                v = self._mini_enc.encode(text)
                if v is not None:
                    return np.asarray(v, np.float32)
            except Exception:
                pass
        return None

    def perceive(self, obs: Obs) -> Embed:
        """任意观测 → 统一表征
        文本: 优先语义编码器 (_text_encoder/MiniLM), 哈希袋兜底"""
        if isinstance(obs, str):
            v = self._enc_text(obs)
            if v is not None:
                return v
            try:
                return np.asarray(self.agent.perception.encode(obs), np.float32)
            except Exception:
                return text_to_obs(obs)
        return self.agent.perception.encode(obs)

    def predict(self, z: Embed, a: int, delta: int = 1) -> Embed:
        """世界模型: 预测 delta 步后的表征 (delta>0 未来)"""
        s = np.asarray(z, np.float32)
        for _ in range(delta):
            s = s + self.agent.world_model.predict_delta(s, a)
        return s

    def energy_of(self, z: Embed) -> float:
        """惊讶度: 观测离已知经验越远越高 (记忆熟悉度反信号).
        无经验时返回 0 (初始谨慎: 记忆空不盲目调工具 —
        否则模型会发空参数工具调用死循环, dsh 会话实证 step 248→251)."""
        items = getattr(self.agent.memory, "items", None)
        if not items:
            return 0.0                      # 无经验 → 不惊讶 → 不盲目行动
        z = np.asarray(z, np.float32)
        zn = z / (np.linalg.norm(z) + 1e-9)
        sims = []
        for m in items[-50:]:
            s = np.asarray(m[0], np.float32)
            sn = s / (np.linalg.norm(s) + 1e-9)
            sims.append(float(np.dot(zn, sn)))
        return float(np.clip(1.0 - max(sims), 0.0, 1.0))

    def decide(self, obs: Obs) -> dict:
        """决策: 返回 {action, tool_calls}
        - 低惊讶: 纯行动 (三环决策)
        - 高惊讶: 行动 + 工具调用 (模型知道自己需要外部信息)
        - 开关: config.tools=False → 不产生 tool_calls (只行动)
        字符串观测交给 perceive 统一处理 (模态编码器/哈希兜底),
        保证 s 与工具语义表征同一空间"""
        s = self.perceive(obs)
        # agent 内部也用统一表征作为观测 (字符串不能直接喂 Dummy 编码器)
        obs_for_agent = s if isinstance(obs, str) else obs
        a, _ = self.agent.decide(obs_for_agent)
        e1 = self.energy_of(s)
        self.stats["decisions"] += 1
        tool_calls = []
        if (self.config.tools and e1 > self.surprise_thresh
                and self.tools.names()):
            # 语义工具选择: 统一空间检索 (失败则不调用 — 不盲选)
            name = self._select_tool(s)
            if name is not None:
                tool_calls.append({"name": name, "arguments": {}})
                self.stats["tool_calls"] += 1
        return {"action": int(a), "energy": round(float(e1), 5),
                "tool_calls": tool_calls}

    def learn(self, obs: Obs, a: int, obs_next: Obs, perf: float,
              died: bool = False, tool_result: Optional[str] = None) -> dict:
        """在线学习 (尊重开关):
        - config.learning=False → 不更新权重 (只算 E1, 不 step 梯度)
        - config.memory=False   → 不写记忆
        - obs 为 None → 纯工具经验写入 (跳过三环, 工具结果学习)"""
        from dataclasses import asdict
        if obs is None:
            if tool_result is not None and self.config.memory:
                z = self.perceive(tool_result[:200])       # 当前编码器空间
                self.agent.memory.items.append((np.asarray(z, np.float32), 1.0, 1.0))
            self.t += 1
            return {"event": None, "tool_result": tool_result}

        s = self.perceive(obs)
        s_next = self.perceive(obs_next)

        # 内环: 世界模型学习 (开关 learning; 关闭则只算 E1 不更新权重)
        if self.config.learning:
            e1 = self.agent.world_model.step(s, a, s_next)
        else:
            e1 = self.agent.world_model.energy(s, a, s_next)

        # 中环: 价值 + 记忆 (开关 memory)
        anchor_v = self.agent.value.anchor(s_next, demand=1.0)
        e2 = self.agent.energy.update(e1, anchor_v)
        if self.config.memory:
            self.agent.memory.write(s, e1, e2)
            self.agent.memory.add_transition(s, a, s_next, e1)

        # 目标坚持
        _, e1_trend, _ = self.agent.energy.get_stats()
        if self.agent.goal.should_switch(e1_trend, self.agent._goal_hold):
            self.agent._cur_goal = self.agent.goal.next_goal(self.agent.memory, s)
            self.agent._goal_hold = 0

        # 外环: Configurator
        if self.agent.configurator is not None:
            self.agent.configurator.observe(perf, self.t, died)
            self.agent._apply_config(self.agent.configurator.get_config())

        ev = Event(t=self.t, s=s, a=a, s_next=s_next,
                   e1=e1, e2=e2, perf=perf, died=died)
        if tool_result is not None and self.config.memory:
            z_tool = self.perceive(tool_result[:200])
            self.agent.memory.write(z_tool, ev.e1, ev.e2)
        self.t += 1
        return {"event": asdict(ev), "tool_result": tool_result}

    # ── 工具接口 ──────────────────────────────────────────
    def register_tool(self, name: str, fn: Callable, description: str,
                      parameters: Optional[dict] = None) -> None:
        self.tools.register(name, fn, description, parameters)
        self._tool_use_counter.setdefault(name, 0)   # 落点4: 调用分布统计
        # 描述 → 统一空间表征 (384d 槽位维度, 与情境 task 槽同空间;
        # 哈希袋也用 384d 投影 — 否则与 1152d 情境点积维度冲突)
        self.tool_embeds[name] = text_to_obs(description, dim=SIT_SLOT)
        if self._text_encoder is not None:
            try:
                self.set_tool_embed(name, self._text_encoder.encode(description))
            except Exception:
                pass
        if self._mini_enc is not None:
            try:
                v = self._mini_enc.encode(description)
                if v is not None:
                    self.set_tool_embed(name, v)
            except Exception:
                pass
        # 工具 → 潜在空间行动索引 (预测式工具选择骨架)
        if name not in self._tool_action and self._next_action < N_TOOLS_MAX:
            self._tool_action[name] = self._next_action
            self._action_tool[self._next_action] = name
            self._next_action += 1

    def set_tool_embed(self, name: str, emb: Embed) -> None:
        """用更强编码器 (如 Qwen/MiniLM) 覆盖工具描述表征"""
        self.tool_embeds[name] = np.asarray(emb, np.float32)

    # ── 工具世界模型 (JEPA 预测器参与工具选择的骨架) ──────
    def _ensure_tool_wm(self):
        """工具世界模型 (惰性构造): ResidualWorldModel(SIT_D, N_TOOLS_MAX).
        预测"调用工具 t 后情境如何变化" — 行动编码用 Δz 隐式学 (方案 a):
        工具 action 列在 W1 中, 随真实 (z, a_t, z_next) 数据学习该工具的效果方向."""
        if self._tool_wm is None:
            from components.world_model import ResidualWorldModel
            self._tool_wm = ResidualWorldModel(
                s_dim=SIT_D, h_dim=256, n_actions=N_TOOLS_MAX,
                seed=getattr(self.config, "seed", 0))
        return self._tool_wm

    def _update_tool_effect(self, tool: str, z: Embed, z_next: Embed) -> None:
        """工具效果原型 v_t = EMA(Δz): 该工具在情境空间中的典型效果方向.
        隐式行动编码 — 不引入外部文本编码, 效果从真实调用轨迹中自举."""
        dz = np.asarray(z_next, np.float32) - np.asarray(z, np.float32)
        if tool in self._tool_effect:
            self._tool_effect[tool] = ((1 - self._effect_alpha)
                                       * self._tool_effect[tool]
                                       + self._effect_alpha * dz)
        else:
            self._tool_effect[tool] = dz

    def _update_tool_embed(self, tool: str, z_ctx: Embed) -> None:
        """工具表征自适应: 成功调用后, 工具描述向量向"被用于的任务"偏移.
        (工具概念从使用中学习 — 描述是静态先验, 经验塑造其语义.
        实测 read 任务与 read 描述仅 0.170, 需靠调用轨迹拉近.)"""
        if tool not in self.tool_embeds:
            return
        task_slot = np.asarray(z_ctx, np.float32)[:SIT_SLOT]
        n = float(np.linalg.norm(task_slot))
        if n < 1e-9:
            return
        task_slot = task_slot / n
        old = self.tool_embeds[tool]
        if old.shape[0] != SIT_SLOT:
            return
        alpha = 0.2
        self.tool_embeds[tool] = ((1 - alpha) * old
                                  + alpha * task_slot).astype(np.float32)

    def _select_planned(self, z: Embed,
                        candidates: list[tuple[str, dict, float]]
                        ) -> Optional[tuple[str, dict]]:
        """预测验证排序: 检索候选 → 世界模型预测各工具后果 Δŝ →
        与工具效果原型 v_t 对齐 (cos) → 混合排序.
        世界模型未训练 (无效果原型) → 退化为纯检索排序 (冷启动安全)."""
        if not candidates:
            return None
        if self._tool_wm is None or not self._tool_effect:
            return (candidates[0][0], candidates[0][1])
        z = np.asarray(z, np.float32)
        scored = []
        for tool, args, sim in candidates:
            a = self._tool_action.get(tool)
            v = self._tool_effect.get(tool)
            if a is None or v is None:
                scored.append((tool, args, sim))
                continue
            d = self._tool_wm.predict_delta(z, a)
            dn = d / (np.linalg.norm(d) + 1e-9)
            vn = v / (np.linalg.norm(v) + 1e-9)
            align = float(np.dot(dn, vn))
            scored.append((tool, args, 0.5 * sim + 0.5 * align))
        scored.sort(key=lambda x: -x[2])
        return (scored[0][0], scored[0][1])

    def _select_tool(self, obs_z: Embed) -> Optional[str]:
        """统一空间语义工具选择: 任务表征 vs 工具描述表征, 最相似者.
        低于阈值 → 不选 (诚实: 不确定就不假装知道该调哪个).
        维度兼容: 感知可能 768d (哈希袋) / 384d (语义), 描述恒 384d →
        感知超长时截取前 SIT_SLOT 维比较."""
        if not self.tool_embeds:
            return None
        z = np.asarray(obs_z, np.float32)
        if z.shape[0] > SIT_SLOT:
            z = z[:SIT_SLOT]
        zn = z / (np.linalg.norm(z) + 1e-9)
        best, best_sim = None, -1.0
        for name, emb in self.tool_embeds.items():
            if name in self._internal_tools:
                continue                    # 内部动作不参与外部工具竞争
            en = np.asarray(emb, np.float32)
            en = en / (np.linalg.norm(en) + 1e-9)
            s = float(np.dot(zn, en))
            if s > best_sim:
                best, best_sim = name, s
        return best if best_sim > 0.05 else None

    def call_tool(self, name: str, args: dict = None) -> str:
        res = self.tools.call(name, args)
        if res.startswith("ERROR"):
            self.stats["tool_errors"] += 1
        return res

    # ── 消息历史 → 情境表征 (多步任务的核心: 每轮基于前序工具结果决策) ──
    def _situation_vec(self, task: str = "", last_call: str = "",
                       last_result: str = "") -> Embed:
        """结构化三槽情境: concat(enc(task), enc(last_call), enc(last_result)).
        - 每槽 384d (MiniLM 语义 / 哈希袋兜底), 拼接归一化 → SIT_D=1152.
        - 时序显式化: "上一步状态" (last_call + last_result) 是独立维度块,
          检索时槽位相似直接决定命中 — "先 A 后 B" 不再靠词频猜.
        - 教学 (learn_call/learn_response) 与运行时 (_context_z) 共用,
          保证检索空间一致. 语义不可用 → 哈希袋 384d 三槽 (维度恒 1152)."""
        def _slot(text: str) -> np.ndarray:
            if not text:
                return np.zeros(SIT_SLOT, dtype=np.float32)
            v = self._enc_text(text[:300])
            if v is None:
                return text_to_obs(text[:300], dim=SIT_SLOT)
            if v.shape[0] != SIT_SLOT:
                # 外部编码器维度不同 (如 Qwen 768d): 截断/补零到 384
                if v.shape[0] > SIT_SLOT:
                    v = v[:SIT_SLOT]
                else:
                    pad = np.zeros(SIT_SLOT - v.shape[0], dtype=np.float32)
                    v = np.concatenate([v, pad])
            n = float(np.linalg.norm(v))
            return v / (n + 1e-9)

        zt = _slot(task)
        zc = _slot(last_call)
        zr = _slot(last_result)
        v = np.concatenate([zt, zc, zr]).astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / (n + 1e-9)

    def _context_z(self, messages: list[dict]) -> Embed:
        """情境表征: 解析消息历史 → 三槽结构化 (任务 / 最近调用 / 最近结果).
        多步任务: 每轮情境 = 任务 + 上一步工具调用 + 上一步结果 →
        检索/预测下一步调用 (时序在槽位里, 不在词频里)."""
        task = ""
        for m in messages:
            if m.get("role") == "user" and m.get("content"):
                task = str(m["content"])[:300]
                break
        last_call, last_result = "", ""
        for m in reversed(messages):
            if not last_result and m.get("role") == "tool" and m.get("content"):
                last_result = str(m["content"])[:150]
            if not last_call and m.get("role") == "assistant":
                tcs = m.get("tool_calls") or []
                if tcs:
                    fn = tcs[-1].get("function", {})
                    last_call = (f"{fn.get('name','')}: "
                                 f"{fn.get('arguments','')}")[:150]
            if last_result and last_call:
                break
        return self._situation_vec(task, last_call, last_result)

    def _tool_loop_guard(self, tool: str) -> bool:
        """循环防护: 同一工具连续 3 次 → True (应停止调用)."""
        self._recent_tools.append(tool)
        if len(self._recent_tools) > 6:
            self._recent_tools = self._recent_tools[-6:]
        return self._recent_tools[-3:] == [tool] * 3

    def reset_task(self) -> None:
        """任务边界: 清空跨任务状态 (使用厌恶轨迹/探索预算等), 任务间隔离."""
        self._recent_tools.clear()
        self._explore_count = 0
        self._explore_temp = 1.0
        self._explore_success = False

    # ── 自由学习: 语义探索 (未命中检索时的试错) ──────────
    # 知识类任务触发词: "what is X" 类问题在 MiniLM 空间与 fetch 描述
    # 余弦仅 0.04-0.19 (具体概念 vs 抽象动作语义距离远) → 语义探索
    # 够不着, 需要领域先验: 知识类问题优先探索检索类工具 (fetch/web/search)
    KNOWLEDGE_TRIGGERS = ("what is", "what are", "what was", "tell me about",
                          "define", "explain", "how does", "how do",
                          "learn about", "research", "look up", "history of",
                          "what percentage", "how many", "which of the following",
                          "what is the value", "calculate", "find the",
                          "compute", "what does", "what was the")
    RETRIEVAL_TOOLS = ("fetch", "web", "search", "http", "web_search",
                       "web_fetch", "lookup", "research")

    def _select_explore(self, z: Embed, task_text: str = "",
                        blocked: str = "") -> Optional[tuple[str, dict]]:
        """探索选择: 检索未命中时, 选任务表征最接近的工具描述试一次.
        - 受阻显式化 (落点1, 命题2): 触发 = 预期失配检测 —
          blocked ∈ {retrieval_miss(无知识), low_confidence(有候选但校准拒绝)}
          都是"受阻" → 允许探索. 知识触发词/统计特征只是受阻后的工具选择先验,
          不是触发条件本身 (对照实验 A: 好奇/触发词漏学"似曾相识但重要"的知识).
        - 探索预算内 + 温度采样 (衰减) — 防无限探索
        - 排除最近用过的工具 (防死循环)
        - 参数: 复用该工具最近一次成功调用的参数 (记忆怎么用)
        执行结果由 tool_result_step 学习 → 下次检索命中 (P4c 33%→72% 路径)."""
        if not getattr(self.config, "explore", True):
            return None
        if getattr(self.config, "benchmark_mode", False):
            return None          # 基准测试: 只用内化知识, 禁外部探索 (防作弊)
        if self._explore_success:
            return None          # 探索已获有效信息 → 评估结果优先 (防探索风暴)
        if self._explore_count >= getattr(self.config, "explore_budget", 3):
            return None
        if self.rng.random() > self._explore_temp:
            return None
        z = np.asarray(z, np.float32)
        # 任务意图与工具描述比较: 用情境的 task 槽 (前 SIT_SLOT 维)
        zt = z[:SIT_SLOT]
        ztn = zt / (np.linalg.norm(zt) + 1e-9)
        # 工具选择先验 (受阻后才生效, 非触发条件) — LeCun 修正:
        # 词表 → 数据驱动. 优先用 call_mem 统计先验 (已验证调用学出
        # "任务实义词 → 工具" 偏好); 统计表无样本时回退 KNOWLEDGE_TRIGGERS
        # 启发式 (渐进替换 — 数据积累后词表自然失效).
        tlow = (task_text or "").lower()
        words = self.call_mem._words_of(tlow)
        prior = self.call_mem.top_tools(words, exclude_tools=self._recent_tools)
        trig_hit = any(trig in tlow for trig in self.KNOWLEDGE_TRIGGERS)
        stat_hit = bool(re.search(r"\d{4}|\d+\s*%|percentage|share of|"
                                  r"growth rate|per capita|how many", tlow))
        if prior:
            # 数据驱动路径: 统计分最高的工具优先试 (已验证调用支持的偏好)
            for name, _score in prior:
                if (name in self.tool_embeds and name not in self._internal_tools
                        and name not in self._recent_tools):
                    self._explore_count += 1
                    self._explore_temp *= getattr(self.config, "explore_decay", 0.9)
                    if name in ("fetch", "web", "search", "http"):
                        return (name, {"query": task_text[:120]})
                    return (name, self._last_ok_args(name))
        elif trig_hit or stat_hit:
            for name in self.RETRIEVAL_TOOLS:
                if (name in self.tool_embeds and name not in self._internal_tools
                        and name not in self._recent_tools):
                    self._explore_count += 1
                    self._explore_temp *= getattr(self.config, "explore_decay", 0.9)
                    # 检索类工具: query=任务文本 (harness 侧拼搜索引擎 URL)
                    if name in ("fetch", "web", "search", "http"):
                        return (name, {"query": task_text[:120]})
                    return (name, self._last_ok_args(name))
        best, best_sim = None, -1.0
        for name, emb in self.tool_embeds.items():
            if name in self._internal_tools or name in self._recent_tools:
                continue
            en = np.asarray(emb, np.float32)
            en = en / (np.linalg.norm(en) + 1e-9)
            s = float(np.dot(ztn, en))
            if s > best_sim:
                best, best_sim = name, s
        if best is None or best_sim <= 0.12:
            return None
        # 参数: 复用该工具最近一次成功调用 (跨情境泛化工具用法)
        self._explore_count += 1
        self._explore_temp *= getattr(self.config, "explore_decay", 0.9)
        return (best, self._last_ok_args(best))

    def _last_ok_args(self, tool: str) -> dict:
        """该工具最近一次成功调用的参数 (perf>=0.3, 跨情境复用)."""
        for cz, t, a, result, perf in reversed(self.call_mem.calls):
            if t == tool and perf >= 0.3:
                return dict(a)
        return {}

    # ── Harness 对接: OpenAI 风格 function-calling ────────
    def chat_completion(self, messages: list[dict],
                        tools: Optional[list[dict]] = None) -> dict:
        """OpenAI 兼容入口 (多步任务支持):
        完整消息历史 → 情境 → 决策顺序:
          1. 调用记忆检索 (完整调用: 工具+参数) — 多步任务的核心
          2. 检索式回应 (学会输出字符, 无工具时)
          3. 默认 content (收尾)"""
        # 任务边界自动重置: 无 assistant/tool 历史 = 新任务 → 清空使用厌恶轨迹
        # (否则跨任务工具被排除, 检索无可选 — dsh 端到端任务2 失败的根因)
        if not any(m.get("role") in ("assistant", "tool") for m in messages):
            self.reset_task()
        # 注入 harness 提供的工具 (描述表征: 有语义编码器则用它, 否则哈希袋)
        for t in (tools or []):
            fn = t.get("function", {})
            name = fn.get("name", "tool")
            self.register_tool(name, lambda **k: k,
                               fn.get("description", ""), fn.get("parameters"))
            if self._text_encoder is not None and fn.get("description"):
                try:
                    self.set_tool_embed(name, self._text_encoder.encode(fn["description"]))
                except Exception:
                    pass

        # 情境 = 完整消息历史 (含工具结果) — 多步决策的依据
        z_ctx = self._context_z(messages)

        # 1) 调用记忆检索 top-K + 预测验证排序: 学过类似情境的完整调用 →
        #    复用 (工具+参数). 世界模型预测后果与工具效果原型对齐者优先.
        #    使用厌恶: 排除最近用过的工具 (防运行时学习自指循环)
        cands = self.call_mem.select_k(z_ctx, k=3,
                                       exclude_tools=tuple(self._recent_tools))
        if cands:
            call = self._select_planned(z_ctx, cands)
            if call is not None:
                tool, args = call
                if tool in self.tools.names() and not self._tool_loop_guard(tool):
                    self.stats["tool_calls"] += 1
                    self._tool_use_counter[tool] = self._tool_use_counter.get(tool, 0) + 1
                    self.t += 1
                    return {
                        "content": "",
                        "tool_calls": [{
                            "id": f"call_{self.t}_0",
                            "type": "function",
                            "function": {"name": tool,
                                         "arguments": json.dumps(args)},
                        }],
                        "via": "call_memory+planned",
                    }
                if self._recent_tools and self._recent_tools[-1] == tool:
                    self._recent_tools.clear()   # 循环已打断, 复位

        # 1.4) 回忆优先于行动: 检索 miss 后, 先查已内化的回应 (知识类任务
        #     fetch 结果绑定到 responder — 学会的答案直接回答, 不再探索).
        if self.config.respond_mode == "retrieval":
            text = self.responder.respond(z_ctx,
                                          task_text=self._task_of(messages))
            if text is not None:
                self.t += 1
                return {"content": text, "tool_calls": [], "via": "retrieval"}
            # 受阻显式化 (落点1, 命题2): 区分"无知识"/"有候选但不可信"/"矛盾"
            # (矛盾=高级受阻: 互斥矛盾未裁决或命中输方 → 弃权 → 查证裁决).
            rs = self.responder
            if getattr(rs, "last_block_reason", "none") == "contradiction":
                self._last_block = "contradiction"
            else:
                self._last_block = ("low_confidence"
                                    if (rs.last_sim >= rs.min_sim and rs.last_sim > 0)
                                    else "retrieval_miss")

        # 1.5) 自由学习探索: 检索未命中 → 语义试错一次 (预算内).
        #     执行结果由 tool_result_step 学习 → 下次检索命中.
        #     (P4c 实证: 记忆路径 33%→72%; 无探索则陌生任务永远学不会)
        if not cands:
            exp = self._select_explore(z_ctx,
                                       task_text=self._task_of(messages),
                                       blocked=self._last_block)
            if exp is not None:
                tool, args = exp
                if tool in self.tools.names() and not self._tool_loop_guard(tool):
                    self.stats["tool_calls"] += 1
                    self.stats["explores"] = self.stats.get("explores", 0) + 1
                    self._tool_use_counter[tool] = self._tool_use_counter.get(tool, 0) + 1
                    self.t += 1
                    return {
                        "content": "",
                        "tool_calls": [{
                            "id": f"call_{self.t}_0",
                            "type": "function",
                            "function": {"name": tool,
                                         "arguments": json.dumps(args)},
                        }],
                        "via": "explore",
                    }

        # 3) 收尾: 无可用经验 → 默认回应 (任务结束由 harness 判定)
        self.t += 1
        return {"content": "Task complete. No more actions needed.",
                "tool_calls": [], "via": "default"}

    def learn_call(self, obs: Obs, tool: str, args: dict,
                   result: str = "", perf: float = 1.0) -> None:
        """显式教学: (情境 → 工具+参数+结果) 存入调用记忆.
        obs 支持三种形式:
          - dict: {"task","last_call","last_result"} — 结构化三槽 (多步教学)
          - str:  情境文本 → (task=文本) 槽
          - ndarray: 已编码情境向量 (直接用)
        同时教工具世界模型 (z, a_t, z_next): 教学不仅教检索, 还教预测 —
        预测器的工具 action 列从教学轨迹学会该工具的效果方向."""
        if isinstance(obs, dict):
            task = str(obs.get("task", ""))
            lc = str(obs.get("last_call", ""))
            lr = str(obs.get("last_result", ""))
            z = self._situation_vec(task, lc, lr)
        elif isinstance(obs, np.ndarray):
            z = np.asarray(obs, np.float32)
            task = ""
        else:
            task = str(obs)
            z = self._situation_vec(task, "", "")
        idx = self.call_mem.learn(z, tool, args, result, perf,
                                  task_text=task)
        # 可证伪性 (落点3, 命题5): 教学 perf 即外部判定 — 成功 → verified,
        # 失败 → falsified (该模式被证伪, 检索不复用, 反例驱动积累)
        self.call_mem.verify(idx, perf >= 0.3)
        # 教工具世界模型: 成功调用 (有结果) 才有监督
        a = self._tool_action.get(tool)
        if (a is not None and result
                and not str(result).startswith("ERROR")):
            z_next = self._situation_vec(
                task, f"{tool}: {json.dumps(args)}", str(result)[:150])
            self._ensure_tool_wm().step(z, a, z_next)
            self._update_tool_effect(tool, z, z_next)

    def learn_response(self, obs: Obs, text: str) -> None:
        """把 (情境, 实际回应) 教给回应学习器 — 让模型学会输出字符.
        obs 支持 dict (三槽) / str / ndarray; 编码与运行时 _context_z 同空间."""
        if isinstance(obs, dict):
            z = self._situation_vec(str(obs.get("task", "")),
                                    str(obs.get("last_call", "")),
                                    str(obs.get("last_result", "")))
        elif isinstance(obs, np.ndarray):
            z = np.asarray(obs, np.float32)
        else:
            z = self._situation_vec(str(obs), "", "")
        self.responder.learn(z, text)

    def learn_response_negative(self, obs: Obs) -> None:
        """标记答错: 该情境的回应不可靠 → 下次不复用旧答案 (重新获取).
        知识混叠纠错: 相似问题不同答案 (上海天气 vs 东京天气 余弦 0.64)."""
        if isinstance(obs, dict):
            z = self._situation_vec(str(obs.get("task", "")),
                                    str(obs.get("last_call", "")),
                                    str(obs.get("last_result", "")))
        elif isinstance(obs, np.ndarray):
            z = np.asarray(obs, np.float32)
        else:
            z = self._situation_vec(str(obs), "", "")
        self.responder.learn_negative(z)

    def responder_feedback(self, correct: bool) -> None:
        """判定器反馈 → 受阻-通过沉淀 (命题1+3): 最近一次 fast 命中的
        回应条目 — 判定正确 → 通过分+1 (可沉淀); 答错 → -2 (污染惩罚).
        harness/测试脚本在每次回答判定后调用 — 沉淀判据从"命中次数"
        升级为"被实践考验且通过" (对照实验 B 实证: 错误固化 1→0)."""
        self.responder.report_outcome(correct)

    @staticmethod
    def _task_of(messages: list[dict]) -> str:
        for m in messages or []:
            if m.get("role") == "user" and m.get("content"):
                return str(m["content"])[:300]
        return ""

    def tool_result_step(self, tool_calls: list[dict],
                         tool_results: list[str],
                         context_messages: Optional[list[dict]] = None,
                         perfs: Optional[list[float]] = None) -> dict:
        """harness 执行工具后回传结果 → 学习完整调用模式 + 记忆 + 世界模型.
        核心: 把 (工具结果前的情境 → 工具+参数 → 结果) 存入调用记忆,
        并用真实 (z, a_t, z_next) 训练工具世界模型 (预测器学会工具效果).
        perfs (可选): 任务级绩效 — 由外部判定器给出 (如测试题答案对错),
        反映"是否推动了任务", 而非"工具是否执行成功". 缺省: 无 ERROR 即成功."""
        if self.config.memory:
            for i, (tc, tr) in enumerate(zip(tool_calls, tool_results)):
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                args = json.loads(fn.get("arguments") or "{}")
                # 调用前情境 = 工具结果回传前的消息历史 (若提供)
                if context_messages:
                    z_ctx = self._context_z(context_messages)
                    task = self._task_of(context_messages)
                else:
                    z_ctx = text_to_obs(f"call {name} with {fn.get('arguments','')}",
                                        dim=SIT_D)
                    task = ""
                # 绩效: 外部判定优先 (测试题答案), 否则工具执行成功
                ok = not str(tr).startswith("ERROR")
                perf = (perfs[i] if perfs and i < len(perfs)
                        else (1.0 if ok else 0.0))
                idx = self.call_mem.learn(z_ctx, name, args, tr, perf=perf,
                                          task_text=task)
                # 可证伪性 (落点3): 外部判定 perf 即时验证该调用模式 —
                # perf≥0.3 → verified (复用); <0.3 → falsified (不复用, 重学)
                self.call_mem.verify(idx, perf >= 0.3)
                if ok and perf >= 0.3:
                    # 成功且绩效达标 → 停止探索 (外部判定答错不算有效信息)
                    self._explore_success = True
                    self._update_tool_embed(name, z_ctx)  # 工具概念从使用中学习
                # 知识类工具结果 → 自动内化为回应知识:
                # (任务情境 → 检索结果) 绑定到回应学习器 — JEPA 用工具
                # 查到的资料自己记住, 下次同问题直接回答 (不再调工具).
                if (ok and name in self.RETRIEVAL_TOOLS
                        and len(str(tr)) > 30):
                    self.responder.learn(z_ctx, str(tr)[:200])
                # 教工具世界模型 (成功调用): 真实轨迹 (z, a_t, z_next)
                a = self._tool_action.get(name)
                if ok and a is not None and task:
                    z_next = self._situation_vec(
                        task, f"{name}: {fn.get('arguments','')}", str(tr)[:150])
                    self._ensure_tool_wm().step(z_ctx, a, z_next)
                    self._update_tool_effect(name, z_ctx, z_next)
                # 工具结果经验入记忆
                z_tool = self.perceive(f"{name}: {tr}"[:200])
                self.agent.memory.items.append((np.asarray(z_tool, np.float32), 1.0, 1.0))
        self.t += 1
        return {"content": f"tool results integrated (t={self.t})",
                "tool_calls": []}

    # ── 睡眠巩固 (记忆回放 → 权重固化) ────────────────────
    def sleep(self, rehearsal: Optional[list] = None) -> dict:
        """睡眠巩固入口: 白天经验重放刻进权重 (受 config.sleep 与参数控制).
        返回巩固前后 E1 等统计 (SleepConsolidation.consolidate)."""
        from components.consolidation import SleepConsolidation
        sc = SleepConsolidation(
            epochs=self.config.sleep_epochs,
            lr_scale=self.config.sleep_lr_scale,
            prio_mix=self.config.sleep_prio_mix,
            seed=self.config.seed)
        return sc.consolidate(self.agent.memory, self.agent.world_model,
                              rehearsal=rehearsal)

    # ── 存档 (任务边界: 快照/恢复) ────────────────────────
    def save(self, path: str) -> None:
        """全量存档: 记忆 + 原型 + 权重 + 回应经验 + 调用记忆 + 工具世界模型.
        恢复后模型回到存档时刻 (任务边界 rollback 的依据)."""
        import pickle
        wm = self.agent.world_model
        twm = self._tool_wm
        state = {
            "t": self.t,
            "stats": self.stats,
            "memory_items": self.agent.memory.items,
            "transitions": list(self.agent.memory.transitions),
            "prototypes": self.agent.memory.prototypes,
            "W1": getattr(wm, "W1", None), "W2": getattr(wm, "W2", None),
            "W_flat": getattr(wm, "W", None),
            "responder_pairs": self.responder.pairs,
            "responder_core": getattr(self.responder, "core_pairs", []),
            "responder_negatives": getattr(self.responder, "negatives", []),
            "responder_calib": getattr(self.responder, "calib", {}),
            "responder_calib2": getattr(self.responder, "calib2", {}),
            "responder_pass": getattr(self.responder, "pass_scores", {}),
            "responder_lam_W": getattr(self.responder, "W", None),
            "responder_lam_av": getattr(self.responder, "answer_vecs", {}),
            "responder_last_hit": getattr(self.responder, "last_hit", {}),
            "responder_hit_counts": getattr(self.responder, "hit_counts", {}),
            "responder_core_scores": getattr(self.responder, "core_scores", {}),
            "responder_archive": getattr(self.responder, "archive", []),
            "responder_lam_G": getattr(self.responder, "G", None),
            "responder_tick": getattr(self.responder, "tick", 0),
            "responder_entry_meta": getattr(self.responder, "entry_meta", {}),
            "responder_contradictions": getattr(self.responder,
                                                "contradictions", []),
            "call_mem": self.call_mem.calls,
            "call_mem_meta": getattr(self.call_mem, "meta", {}),
            "call_mem_prior": getattr(self.call_mem, "tool_prior", {}),
            "tool_wm_W1": twm.W1 if twm else None,
            "tool_wm_W2": twm.W2 if twm else None,
            "tool_effect": self._tool_effect,
            "tool_action": self._tool_action,
            "config": self.config.to_dict(),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """恢复存档 (必须与 save 配对; 恢复覆盖当前状态)."""
        import pickle
        with open(path, "rb") as f:
            st = pickle.load(f)
        self.t = st["t"]
        self.stats = st["stats"]
        self.agent.memory.items = st["memory_items"]
        self.agent.memory.transitions = st["transitions"]
        self.agent.memory.prototypes = st["prototypes"]
        wm = self.agent.world_model
        if st.get("W1") is not None:
            wm.W1, wm.W2 = st["W1"], st["W2"]
        elif st.get("W_flat") is not None and hasattr(wm, "W"):
            wm.W = st["W_flat"]
        self.responder.pairs = st["responder_pairs"]
        self.responder.core_pairs = st.get("responder_core", [])
        self.responder.negatives = st.get("responder_negatives",
                                          getattr(self.responder, "negatives", []))
        self.responder.calib = st.get("responder_calib",
                                      getattr(self.responder, "calib", {}))
        self.responder.calib2 = st.get("responder_calib2",
                                       getattr(self.responder, "calib2", {}))
        self.responder.pass_scores = st.get("responder_pass",
                                            getattr(self.responder, "pass_scores", {}))
        self.responder.W = st.get("responder_lam_W",
                                  getattr(self.responder, "W", None))
        self.responder.answer_vecs = st.get("responder_lam_av",
                                            getattr(self.responder,
                                                    "answer_vecs", {}))
        self.responder.last_hit = st.get("responder_last_hit",
                                         getattr(self.responder, "last_hit", {}))
        self.responder.hit_counts = st.get("responder_hit_counts",
                                           getattr(self.responder,
                                                   "hit_counts", {}))
        self.responder.core_scores = st.get("responder_core_scores",
                                            getattr(self.responder,
                                                    "core_scores", {}))
        self.responder.archive = st.get("responder_archive",
                                        getattr(self.responder, "archive", []))
        self.responder.G = st.get("responder_lam_G",
                                  getattr(self.responder, "G", None))
        self.responder.tick = st.get("responder_tick",
                                     getattr(self.responder, "tick", 0))
        self.responder.entry_meta = st.get("responder_entry_meta",
                                           getattr(self.responder,
                                                   "entry_meta", {}))
        self.responder.contradictions = st.get(
            "responder_contradictions",
            getattr(self.responder, "contradictions", []))
        self.call_mem.calls = st["call_mem"]
        if hasattr(self.call_mem, "meta"):
            self.call_mem.meta = st.get("call_mem_meta", {})
            self.call_mem.tool_prior = st.get("call_mem_prior", {})
            # 旧存档兼容: 无 meta 时从 result/perf 重建状态
            if not self.call_mem.meta and self.call_mem.calls:
                self.call_mem.meta = {
                    i: {"status": self.call_mem._derive_status(r, p),
                        "verified_n": 0, "falsified_n": 0}
                    for i, (_, _, _, r, p) in enumerate(self.call_mem.calls)}
        # 空间兼容: 旧存档 (哈希袋 768d 情境) 与新空间 (1152d) 不兼容 → 丢弃重学
        if (self.call_mem.calls
                and np.asarray(self.call_mem.calls[0][0]).shape[0] != SIT_D):
            self.call_mem.calls = []
        if (self.responder.pairs
                and np.asarray(self.responder.pairs[0][0]).shape[0] != SIT_D):
            self.responder.pairs = []
        if (getattr(self.responder, "core_pairs", [])
                and np.asarray(self.responder.core_pairs[0][0]).shape[0] != SIT_D):
            self.responder.core_pairs = []
        # 工具世界模型 + 效果原型恢复
        if st.get("tool_effect"):
            self._tool_effect = st["tool_effect"]
        if st.get("tool_action"):
            self._tool_action = st["tool_action"]
            self._action_tool = {v: k for k, v in st["tool_action"].items()}
            self._next_action = max(self._action_tool) + 1 if self._action_tool else 0
        if st.get("tool_wm_W1") is not None:
            twm = self._ensure_tool_wm()
            twm.W1 = st["tool_wm_W1"]
            twm.W2 = st["tool_wm_W2"]

    # ── 认知再生产指标 (落点4, 命题4) ────────────────────
    def _cognition_metrics(self) -> dict:
        """健康判据 = 结构是否还在质变 (范式更新率), 不是"记住了多少".
        对齐《智能论批判》认知病理学四型: 表征失真 / 检索退化 / 认知僵化 /
        价值萎缩 的可观测判据 (responder.health_dict 内实现前三项)."""
        health = self.responder.health_dict()
        # 工具使用熵 (价值萎缩检测: 行为单调化 — 多工具只用少数几个)
        counts = list(self._tool_use_counter.values())
        ent = 0.0
        used = sum(1 for c in counts if c > 0)
        if counts and sum(counts) > 0:
            tot = sum(counts)
            probs = [c / tot for c in counts if c > 0]
            ent = -sum(p * np.log(p) for p in probs)
        health["tool_entropy"] = round(ent, 3)
        health["tools_used"] = used
        health["n_tools"] = len(self._tool_use_counter)
        if len(self._tool_use_counter) >= 5 and used <= 2 \
                and sum(counts) >= 10:
            health["warnings"].append(
                f"价值萎缩: {len(self._tool_use_counter)} 个工具只用过 {used} 个 — "
                f"行为单调化 (需环境异质性输入)")
        return health

    # ── 状态导出 (harness 诊断) ───────────────────────────
    def status(self) -> dict:
        return {
            "t": self.t,
            "stats": self.stats,
            "config": self.config.describe(),
            "tools": self.tools.names(),
            "memory": len(self.agent.memory.items),
            "prototypes": len(getattr(self.agent.memory, "prototypes", [])),
            "responder": self.responder.stats_dict(),
            "call_mem": self.call_mem.stats_dict(),
            "last_block": self._last_block,          # 落点1: 最近受阻类型诊断
            "cognition": self._cognition_metrics(),  # 落点4: 认知再生产健康
            "semantic": (self._mini_enc.stats()
                         if self._mini_enc is not None
                         else {"available": False, "failed": True,
                               "cache": 0, "dim": 0}),
            "ctx_dim": SIT_D,
            "tool_wm": self._tool_wm is not None,
            "tool_effects": len(self._tool_effect),
            "rate": getattr(self.agent.configurator, "rate", None) if self.agent.configurator else None,
        }
