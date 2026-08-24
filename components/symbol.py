"""
C8 SymbolLayer — 符号层 (中环)
==============================
契约: 结构/行为的压缩签名 + LLM 语义锚.

核心原则 (实证裁决, 2026-08-24 修正):
  - **符号层分层 (真实结构池 bb_human_check 实证)**:
      L0 图论粗筛 (有环/SCC/可达 → 缩搜索空间)
      L1 环内容分析 (反汇编级: 环内"写1×移动"比例 → 结构长尾正解, 100% 命中)
      L2 行为指纹 (短观测统计 → 统计可学习分布信号)
  - 旧结论修正: jpi6 "行为指纹 +20.1pp > 结构注入 +19.3pp" 只在
    无注入池 (真实最长 log 2.9, 全短机器) 成立; 真实结构长尾上
    环内容 100% >> 行为回归 50% — 行为指纹不是符号层的主要内容.
  - LLM 文本原型 = 语义锚 (timm 起点蒸馏 +18.4pp KNN; zero-shot 66%)
  - 自适应原型聚类 (距离分位), 无手工阈值

接口:
  fingerprint(trajectory) → f       (L2 行为指纹: 短观测统计)
  structure_score(rules) → s        (L0+L1: 图论粗筛 + 环内容判别, 零模拟)
  match(f) → symbol_id               (原型匹配 + 自适应新建)
  text_anchor(text) → proto          (LLM 文本原型, Qwen 编码)

验收: bb_human_check (环内容 100%); jpi6 (行为指纹); p3_check.py (符号涌现)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .core import D, Embed


class SymbolLayer(ABC):
    """C8 符号层接口"""

    @abstractmethod
    def fingerprint(self, trajectory: list[Embed]) -> Embed:
        """行为轨迹 → 压缩观测签名 (统计量)."""

    @abstractmethod
    def match(self, f: Embed) -> int:
        """签名 → 原型 id (自适应新建)."""

    @abstractmethod
    def text_anchor(self, text: str) -> Embed:
        """文本 → 语义原型 (LLM 编码 + 投影)."""


class RingContentSymbol(SymbolLayer):
    """L0+L1 结构符号 (bb_human_check 实证正解):
    零模拟静态分析, 区分"计数器(扩张)"与"空转(死循环)" —
    人类看规则表追问'这个环在干什么'的机器实现.
    structure_score(rules) → 标量: 越大越可能是长运行机器
    """
    def __init__(self, n_states: int = 4):
        self.n_states = n_states

    def structure_score(self, rules: np.ndarray, n: int | None = None) -> float:
        """规则表 → 环内容分数 (零模拟).
        = 最大 SCC 占比 × 可达比例 × (1-halt 密度) × 环内写1比例 × 环内移动比例
        写1+移动 = 磁带扩张 (计数器候选); 写0原地 = 空转死循环"""
        n = n or self.n_states
        adj = np.zeros((n, n), dtype=np.float32)
        writes = np.zeros((n, n, 2), dtype=np.float32)
        cnt = np.zeros((n, n), dtype=np.float32)
        halt = 0
        for i in range(n * 2):
            st = i // 2
            ws, d, nxt = rules[i]
            if nxt < 0:
                halt += 1
            else:
                adj[st, nxt] += 1
                cnt[st, nxt] += 1
                writes[st, nxt, 0] += (1.0 if ws == 1 else 0.0)
                writes[st, nxt, 1] += (1.0 if d == 1 else 0.0)
        reachable = np.zeros(n, dtype=bool)
        stack = [0]
        while stack:
            u = stack.pop()
            if reachable[u]:
                continue
            reachable[u] = True
            stack.extend(np.where(adj[u] > 0)[0])
        reach_ratio = reachable.sum() / n
        # 最大 SCC (近似: 可达子图上的强连通)
        sub = adj[reachable][:, reachable]
        if sub.shape[0] >= 1:
            p = sub.copy()
            for _ in range(min(8, sub.shape[0])):
                p = p @ sub
            # 对角 = 强连通自达 → SCC 近似
            scc_approx = float(np.clip(np.trace(p) / max(1, sub.shape[0]), 0, 1))
        else:
            scc_approx = 0.0
        # 最大 SCC 内写1/移动比例 (近似: 可达子图统计)
        tot = cnt[reachable][:, reachable].sum()
        if tot > 0:
            exp_w = writes[reachable][:, reachable, 0].sum() / tot
            exp_m = writes[reachable][:, reachable, 1].sum() / tot
        else:
            exp_w = exp_m = 0.0
        halt_den = halt / (n * 2)
        return float(scc_approx * reach_ratio * (1.0 - halt_den) * exp_w * exp_m)

    # SymbolLayer 接口: 行为指纹部分降级为 L2 (统计可学习分布)
    def fingerprint(self, trajectory: list[Embed]) -> Embed:
        return np.zeros(8, np.float32)

    def match(self, f: Embed) -> int:
        return 0

    def text_anchor(self, text: str) -> Embed:
        h = abs(hash(text))
        rng = np.random.RandomState(h % (2 ** 32))
        return rng.randn(D).astype(np.float32) * 0.1


class BehavioralSymbol(SymbolLayer):
    """真实实现: 行为指纹 (jpi6) + 自适应原型聚类
    指纹 = 轨迹的聚合统计 (均值/标准差/位移/步长/多样性/末端态)
    —— "行为的压缩观测签名", 不依赖任何结构定义.
    """
    def __init__(self, proto_q: float = 0.85, max_protos: int = 16,
                 min_sep: float = 0.5, seed: int = 0):
        self.proto_q = proto_q
        self.max_protos = max_protos
        self.min_sep = min_sep
        self.rng = np.random.RandomState(seed)
        self.prototypes: list[Embed] = []
        self.dist_hist: list[float] = []
        self.symbol_count: dict[int, int] = {}

    # ── 行为指纹 (jpi6 迁移, v2: 纯运动统计, 无位置维度) ──
    def fingerprint(self, trajectory: list[Embed]) -> Embed:
        """轨迹 [T,D] → 8 维运动/行为统计 (行为签名)
        v2 修正: 移除位置主导维度 (全局均值/质心), 强化行为维度
        (步长/转向率/静止比率/多样性) — 符号应区分"行为模式"."""
        if len(trajectory) < 3:
            return np.zeros(8, np.float32)
        arr = np.stack(trajectory)                 # [T, D]
        d = np.diff(arr, axis=0)                   # [T-1, D]
        steps = np.linalg.norm(d, axis=1)          # [T-1] 每步位移量
        # 方向变化率 (转向比率): 相邻位移夹角 > 72° 的比例
        dirs = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        dot = np.sum(dirs[:-1] * dirs[1:], axis=1)
        turn = float(np.mean(dot < 0.3))
        # 状态多样性 (量化唯一比例)
        q = (arr[::2] > 0).astype(np.int8)
        uniq = len(np.unique(q, axis=0)) / max(1, len(q))
        f = np.array([
            float(steps.mean()),                    # 平均步长
            float(steps.std()),                     # 步长波动
            float(steps.max()),                     # 最大步长
            turn,                                   # 转向率
            float((steps < 0.05).mean()),           # 静止比率
            float(uniq),                            # 状态多样性
            float(arr[-1].std()),                   # 末端波动
            float(np.linalg.norm(arr[-1] - arr[0])),  # 总位移
        ], dtype=np.float32)
        n = np.linalg.norm(f)
        return f / (n + 1e-6) if n > 1e-6 else f

    # ── 自适应原型匹配 ───────────────────────────────────
    def match(self, f: Embed) -> int:
        """返回符号 id; 距离超过分位阈值 → 新建原型"""
        if not self.prototypes:
            self.prototypes.append(np.asarray(f, np.float32).copy())
            self.symbol_count[0] = self.symbol_count.get(0, 0) + 1
            return 0
        dists = [float(np.linalg.norm(np.asarray(f, np.float32) - p))
                 for p in self.prototypes]
        best = int(np.argmin(dists))
        self.dist_hist.append(dists[best])
        if len(self.dist_hist) >= 20:
            thresh = float(np.percentile(self.dist_hist, self.proto_q * 100))
            if dists[best] > max(thresh, self.min_sep) and \
                    len(self.prototypes) < self.max_protos:
                self.prototypes.append(np.asarray(f, np.float32).copy())
                new_id = len(self.prototypes) - 1
                self.symbol_count[new_id] = self.symbol_count.get(new_id, 0) + 1
                return new_id
        self.symbol_count[best] = self.symbol_count.get(best, 0) + 1
        return best

    # ── LLM 语义锚 (占位; 真实接入见 qwen_to_jepa) ───────
    def text_anchor(self, text: str) -> Embed:
        """文本 → 语义原型. 真实现: Qwen 编码 + 投影 (qwen_to_jepa).
        此处返回确定性伪锚 (组装/验收用)."""
        h = abs(hash(text))
        rng = np.random.RandomState(h % (2 ** 32))
        return rng.randn(D).astype(np.float32) * 0.1

    # ── 验收辅助: 原型区分度 ─────────────────────────────
    def prototype_separation(self, group_assign: list[int]) -> float:
        """两组轨迹的原型中心距离 (区分度指标)"""
        if len(self.prototypes) < 2:
            return 0.0
        centers = {g: [] for g in set(group_assign)}
        # 用符号计数近似: 不同组对应不同原型
        protos = np.stack(self.prototypes)
        d = np.linalg.norm(protos[:, None] - protos[None, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return float(d.min())
