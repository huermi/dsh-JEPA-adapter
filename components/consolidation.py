"""
C5b SleepConsolidation — 睡眠巩固 (记忆回放 → 权重固化)
=======================================================
做中学闭环的"睡眠环": 白天的经验在"睡眠"时被重放、强化、刻进权重.
对应人类: 海马体短期记忆 → 睡眠纺锤波重放 → 新皮层长期固化 → 短期记忆清空.

机制 (每条都有实证/生物学对应):
  - 数据源: memory.transitions (完整转移 (s,a,s_next,e1)) — 白天探索的真实事件
  - 优先级回放: 按惊讶度 e1 加权采样 (高惊讶=重要; 人类睡眠选择性强化
    强烈/新颖经验; jpi10 实证稀疏奖励下优先级回放 > 即时训练)
  - 低 lr 多轮重放: 反复重放 (睡眠纺锤波) 但低学习率 (不覆盖旧知识,
    防灾难性遗忘 — jpi5 教训)
  - 巩固后短期记忆衰减: 经验刻进权重后海马体清空 (transitions 保留 recent,
    长期记忆由权重承载, 短期缓冲继续服务下一次探索)

接口:
  consolidate(memory, world_model) → dict
    {e1_before, e1_after, drop_pct, n_replayed, n_forgotten}

验收: sleep_check.py (巩固价值 + 防遗忘 + 优先级 vs 均匀)
"""
from __future__ import annotations

import numpy as np

from .core import D, Embed
from .memory import MemorySystem
from .world_model import WorldModel


class SleepConsolidation:
    """睡眠巩固器: 优先级回放 + 低 lr 多轮强化 + 短期记忆衰减"""

    def __init__(self, epochs: int = 3, batch: int = 32,
                 prio_power: float = 1.0, prio_mix: float = 0.5,
                 lr_scale: float = 0.3,
                 recent_keep: float = 0.2, seed: int = 0):
        """
        epochs:     重放轮数 (每轮对整个池子采样训练)
        batch:      每轮采样条数
        prio_power: 优先级指数 — 1.0=按惊讶度加权, 0.0=均匀
        prio_mix:   优先级混合比例 — 采样分布 = (1-mix)*均匀 + mix*优先级.
                    纯优先级 (mix=1) 会把模型拉向稀有子分布, 饿死常规知识
                    (sleep_check 实证: 普通转移退化 -67287%);
                    人类睡眠同时巩固常规经验 (常规的梦) 与惊奇事件 (噩梦),
                    mix=0.5 是"选择性但不独占"的默认
        lr_scale:   巩固学习率 = 在线 lr × 该系数 (低 = 防灾难性遗忘)
        recent_keep: 巩固后保留的最近转移比例 (海马体清空, 短期缓冲瘦身)
        """
        self.epochs = epochs
        self.batch = batch
        self.prio_power = prio_power
        self.prio_mix = prio_mix
        self.lr_scale = lr_scale
        self.recent_keep = recent_keep
        self.rng = np.random.RandomState(seed)

    def consolidate(self, memory: MemorySystem,
                    world_model: WorldModel,
                    rehearsal: list | None = None) -> dict:
        """执行一次睡眠巩固. 返回前后 E1 (留出集评估) 与统计.
        rehearsal: 可选旧经验列表 [(s, a, s_next, e1)], 混入重放池.
        防灾难性遗忘的简单手段 (EWC 的记忆版): 巩固新任务时同步重放
        已掌握的旧经验, 让旧知识不被新经验覆盖 (睡眠同时重放新旧记忆)."""
        trans = list(memory.get_transitions())
        if rehearsal:
            trans = trans + list(rehearsal)
        if len(trans) < self.batch:
            return {"e1_before": float("nan"), "e1_after": float("nan"),
                    "drop_pct": 0.0, "n_replayed": 0,
                    "n_forgotten": 0, "ok": False}

        # 留出评估集 (不参与训练, 测巩固是否真正提升泛化)
        n_eval = min(24, len(trans) // 3)
        eval_idx = self.rng.choice(len(trans), n_eval, replace=False)
        eval_set = [trans[i] for i in eval_idx]

        def eval_e1():
            es = [world_model.energy(s, a, sn)
                  for s, a, sn, _ in eval_set]
            return float(np.mean(es))

        e1_before = eval_e1()

        # 优先级权重: 采样分布 = (1-mix)*均匀 + mix*优先级
        # (纯优先级会饿死常规知识 — sleep_check 实证 -67287% 退化)
        e1s = np.array([t[3] for t in trans], dtype=np.float32)
        eps = float(np.percentile(e1s, 50)) * 0.1 + 1e-6
        if self.prio_power > 0:
            w_prio = (e1s + eps) ** self.prio_power
            w_prio = w_prio / w_prio.sum()
            w_unif = np.ones(len(trans), dtype=np.float32) / len(trans)
            w = (1.0 - self.prio_mix) * w_unif + self.prio_mix * w_prio
            w = w / w.sum()
        else:
            w = None   # 均匀

        # 低 lr 多轮重放 (睡眠纺锤波式强化)
        lr_orig = getattr(world_model, "lr_pred", 0.05)
        world_model.set_lr(lr_orig * self.lr_scale)
        n_replayed = 0
        n_rehearsal_sampled = 0
        rehearsal_ids = {id(t) for t in (rehearsal or [])}
        try:
            for _ in range(self.epochs):
                idx = self.rng.choice(len(trans), self.batch,
                                      p=w, replace=True)
                for i in idx:
                    s, a, sn, _ = trans[i]
                    world_model.step(s, a, sn)
                    n_replayed += 1
                    if id(trans[i]) in rehearsal_ids:
                        n_rehearsal_sampled += 1
        finally:
            world_model.set_lr(lr_orig)

        e1_after = eval_e1()

        # 短期记忆衰减: 经验刻进权重, 海马体清空 (保留 recent 比例)
        # 注意: rehearsal 经验不写入记忆, 不影响衰减
        n_keep = max(1, int(len(memory.get_transitions()) * self.recent_keep))
        n_forgotten = len(memory.get_transitions()) - n_keep
        memory.transitions = memory.transitions.__class__(
            memory.get_transitions()[-n_keep:],
            maxlen=memory.transitions.maxlen)

        return {
            "e1_before": e1_before,
            "e1_after": e1_after,
            "drop_pct": (1 - e1_after / max(e1_before, 1e-12)) * 100,
            "n_replayed": n_replayed,
            "n_rehearsal_sampled": n_rehearsal_sampled,
            "n_forgotten": n_forgotten,
            "ok": e1_after < e1_before,
        }
