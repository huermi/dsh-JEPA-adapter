"""
调用模式记忆 (body/call_memory.py)
===================================
让 JEPA 学会"完整工具调用" — 不只是选工具名, 还记住参数.

核心思想:
  调用不是生成的, 是检索的. 每次成功调用 (情境 → 工具 + 参数 + 结果)
  存入记忆; 新任务来了, 找到最相似历史情境, 复用其 (工具, 参数).
  这解决上一轮的根因: 模型 arguments 恒空 → 工具执行死循环
  (str_replace_editor step 248→251 事故).

多步任务的支撑:
  每轮工具结果回传后, 消息历史变化 → 情境变化 → 检索到下一步调用 →
  连续多步 (list → read → done) 的每一步都由情境驱动.

接口:
  select(z_ctx) -> Optional[(tool, args)]   检索复用完整调用
  learn(z_ctx, tool, args, result, perf)    存入成功/失败调用
  n() -> int
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from components.core import D, Embed


class CallMemory:
    """调用模式记忆: 情境 → (工具, 参数) 的经验检索.
    范式三元组 (命题5): 每条调用 = (情境, 行动模式, 结果状态) — 结果可被
    外部判定证伪 (verify): verified(判定正确) / falsified(被证伪) /
    failed(执行失败) / unverified(待验证). 被证伪的调用不复用
    ("检索命中的回答要求与结果一致才可复用" — 反例驱动的知识积累)."""

    def __init__(self, min_sim: float = 0.50, cap: int = 300, seed: int = 0):
        # min_sim: 0.50 (MiniLM 语义空间实测 — 任务间余弦 0.49-0.65 普遍
        # 高于任务-工具匹配 0.05-0.57; 0.40 挡不住相似任务互相污染)
        self.min_sim = min_sim
        self.cap = cap
        self.calls: list[tuple[Embed, str, dict, str, float]] = []
        # (z_ctx, tool, args, result, perf)
        # 结果状态 (并行索引, 与 calls 同步): idx -> {status, verified_n, falsified_n}
        self.meta: dict[int, dict] = {}
        self.last_idx: int = -1            # 最近检索命中的索引 (verify 反馈用)
        self.last_fast_ok = False          # 最近一次 select 是否命中 fast 候选
        # 任务→工具统计先验 (LeCun 修正: 词表 → 数据驱动): 实义词 -> {tool: 计数}.
        # 从已验证调用学"什么任务用什么工具" — 替代 kernel 的 KNOWLEDGE_TRIGGERS 手工词表.
        # verify(False) 时对应调用减计数 (被证伪的偏好可撤销, 反例驱动).
        self.tool_prior: dict[str, dict[str, float]] = {}
        self.rng = np.random.RandomState(seed)
        self.stats = {"hits": 0, "misses": 0}

    _STOP_WORDS = frozenset((
        "what", "which", "that", "with", "from", "this", "have", "were",
        "about", "when", "where", "than", "been", "will", "into", "over",
        "more", "most", "would", "there", "their", "the", "and", "for",
        "are", "was", "not", "all", "you", "your", "can", "could", "should",
        "is", "of", "to", "in", "on", "it", "as", "a", "an", "do", "does",
        "did", "has", "had", "how", "why", "list", "find", "get", "tell"))

    @classmethod
    def _words_of(cls, text: str) -> list[str]:
        """任务文本实义词提取 (与 responder 词汇闸同风格)."""
        import re
        return [w for w in re.split(r"[^a-z0-9]+", (text or "").lower())
                if len(w) >= 3 and w not in cls._STOP_WORDS]

    @staticmethod
    def _derive_status(result: str, perf: float) -> str:
        if str(result).startswith("ERROR"):
            return "failed"
        if perf < 0.3:
            return "failed"
        return "unverified"                # 待外部判定 (可证伪通道)

    def select(self, z: Embed,
               exclude_tools: tuple = ()) -> Optional[tuple[str, dict]]:
        """检索最相似情境的调用 (工具+参数). 相似度不足 → None (不盲调).
        exclude_tools: 最近用过的工具 (使用厌恶 — 刚 glob 完不可能再 glob,
        否则运行时学习自指循环: multi_step_check 任务2 死循环根因).
        结果状态门控: falsified (被证伪) 的调用不复用 — 反例驱动的积累."""
        self.last_fast_ok = False
        if not self.calls:
            self.stats["misses"] += 1
            return None
        z = np.asarray(z, np.float32)
        zn = z / (np.linalg.norm(z) + 1e-9)
        best, best_sim, best_idx = None, -1.0, -1
        for i, (cz, tool, args, result, perf) in enumerate(self.calls):
            if perf < 0.3:
                continue                # 失败的调用不复用
            if self.meta.get(i, {}).get("status") == "falsified":
                continue                # 被证伪的模式不复用 (落点3)
            if tool in exclude_tools:
                continue                # 刚用过的工具不立即复用
            czn = np.asarray(cz, np.float32)
            czn = czn / (np.linalg.norm(czn) + 1e-9)
            s = float(np.dot(zn, czn))
            if s > best_sim:
                best, best_sim, best_idx = (tool, args), s, i
        if best_sim >= self.min_sim:
            self.stats["hits"] += 1
            self.last_idx, self.last_fast_ok = best_idx, True
            return best
        self.stats["misses"] += 1
        return None

    def select_k(self, z: Embed, k: int = 3,
                 exclude_tools: tuple = ()) -> list[tuple[str, dict, float]]:
        """Top-K 候选检索 (预测验证用): 返回 [(tool, args, sim)] 降序.
        检索只生成候选, 排序可由世界模型预测验证修正 (kernel._select_planned).
        低于 min_sim 的候选不返回 (诚实底线). 排除 falsified (被证伪)."""
        self.last_fast_ok = False
        if not self.calls:
            self.stats["misses"] += 1
            return []
        z = np.asarray(z, np.float32)
        zn = z / (np.linalg.norm(z) + 1e-9)
        scored = []
        for i, (cz, tool, args, result, perf) in enumerate(self.calls):
            if perf < 0.3:
                continue
            if self.meta.get(i, {}).get("status") == "falsified":
                continue                # 被证伪的模式不进入候选 (落点3)
            if tool in exclude_tools:
                continue
            czn = np.asarray(cz, np.float32)
            czn = czn / (np.linalg.norm(czn) + 1e-9)
            s = float(np.dot(zn, czn))
            if s >= self.min_sim:
                scored.append((tool, args, s, i))
        scored.sort(key=lambda x: -x[2])
        if scored:
            self.stats["hits"] += 1
            self.last_idx, self.last_fast_ok = scored[0][3], True
        else:
            self.stats["misses"] += 1
        return [(t, a, s) for t, a, s, _ in scored[:k]]

    def learn(self, z: Embed, tool: str, args: dict,
              result: str, perf: float = 1.0,
              task_text: str = "") -> int:
        """存入调用经验. perf<0.3 的失败调用也存 (供 select 跳过, 防重蹈).
        结果状态从 result/perf 推导 (failed/unverified). 返回条目索引 (供验证).
        task_text: 任务文本 → 统计先验 (实义词→工具计数, 仅成功调用计入)."""
        self.calls.append((np.asarray(z, np.float32).copy(), tool,
                           dict(args), str(result)[:200], float(perf)))
        idx = len(self.calls) - 1
        words = self._words_of(task_text)
        self.meta[idx] = {"status": self._derive_status(result, perf),
                          "verified_n": 0, "falsified_n": 0,
                          "words": words}
        if perf >= 0.3:
            for w in words:
                self.tool_prior.setdefault(w, {})
                self.tool_prior[w][tool] = self.tool_prior[w].get(tool, 0.0) + 1.0
        if len(self.calls) > self.cap:
            self.calls.pop(0)
            # meta 索引左移 (淘汰最旧)
            self.meta = {i - 1: v for i, v in self.meta.items() if i > 0}
            idx -= 1
        return idx

    def verify(self, idx: int, correct: bool) -> None:
        """外部判定反馈 (可证伪通道, 命题5): 判定正确 → verified;
        判定错误 → falsified (该模式被证伪, 后续不复用 — 反例驱动积累).
        同时撤销被证伪调用的统计先验计数 (词表统计化: 偏好可被证伪修正).
        "检索命中的回答要求与结果一致才可复用"."""
        if idx < 0 or idx >= len(self.calls):
            return
        m = self.meta.setdefault(idx, {"status": "unverified",
                                       "verified_n": 0, "falsified_n": 0})
        if correct:
            m["verified_n"] += 1
            m["status"] = "verified"
        else:
            m["falsified_n"] += 1
            m["status"] = "falsified"
            # 被证伪 → 该调用贡献的词-工具计数撤销 (防负)
            _, tool, _, _, _ = self.calls[idx]
            for w in m.get("words", []):
                tbl = self.tool_prior.get(w)
                if tbl and tool in tbl:
                    tbl[tool] = max(tbl[tool] - 1.0, 0.0)
                    if tbl[tool] <= 0:
                        del tbl[tool]

    def top_tools(self, words: list[str],
                  exclude_tools: tuple = ()) -> list[tuple[str, float]]:
        """统计先验查询: 任务实义词 → 工具偏好分 (已验证调用累计).
        返回 [(tool, score)] 降序. 空统计 (无样本) → [] (调用方回退启发式)."""
        if not self.tool_prior or not words:
            return []
        scores: dict[str, float] = {}
        for w in words:
            tbl = self.tool_prior.get(w)
            if not tbl:
                continue
            for tool, cnt in tbl.items():
                if tool in exclude_tools:
                    continue
                scores[tool] = scores.get(tool, 0.0) + cnt
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:5]

    def n(self) -> int:
        return len(self.calls)

    def stats_dict(self) -> dict:
        st = {**self.stats, "n": len(self.calls),
              "hit_rate": (self.stats["hits"] /
                           max(self.stats["hits"] + self.stats["misses"], 1))}
        st["verified"] = sum(1 for m in self.meta.values()
                             if m["status"] == "verified")
        st["falsified"] = sum(1 for m in self.meta.values()
                              if m["status"] == "falsified")
        st["failed"] = sum(1 for m in self.meta.values()
                           if m["status"] == "failed")
        return st
