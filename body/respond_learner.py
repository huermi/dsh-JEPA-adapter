"""
回应学习器 (body/respond_learner.py)
=====================================
让模型"学会输出字符" — 回应 = 情境→文本 的经验检索, 不依赖 LLM 表达.

核心思想:
  回应不是生成的, 是检索的. 每个 (情境表征 z, 响应文本 text) 是一次
  经验; 新请求来了, 找到最相似的历史情境, 复用它的回应. 这正是
  "回忆 = 输出" 的扩展 (body_real_check R2 实证 50%) —
  情境相似 → 回应复用, 就是"学会说话"的最简形态.

接口:
  respond(z) -> Optional[str]   检索最相似情境的回应 (相似度 < min_sim 返回 None)
  learn(z, text)                存入 (情境, 回应) 经验
  n() -> int                    经验条数

与工具调用的关系 (用户设计意图):
  回应被实现为一种特殊工具调用 — 插件把 respond 注册为工具,
  模型的"回应动作" = 执行 respond 工具 = 从经验检索字符.
  harness 侧看起来就是一次正常的 function-calling 往返.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from components.core import D, Embed


class RespondLearner:
    """回应学习器: 情境→文本 记忆检索 (检索式输出, 无生成器)"""

    def __init__(self, min_sim: float = 0.45, cap: int = 300, seed: int = 0,
                 margin_thresh: float = 0.15, calib_thresh: float = 0.5,
                 soft_align: bool = True, soft_align_alpha: float = 0.1,
                 conflict_gate: bool = True):
        """
        min_sim: 检索复用的相似度阈值 (太低 → 乱套用, 太高 → 总未命中).
                 经验少时建议偏低; 可后续做自适应分位 (surprise 阈值教训).
        cap:     经验上限 (超限淘汰最旧 — 只保留最近说话风格)
        neg_thresh: 负样本阈值 — 情境与"答错标记"相似度高于此 → 拒绝旧答案
                    (知识混叠纠错: 相似问题不同答案, 如上海天气 vs 东京天气)
        """
        self.min_sim = min_sim
        self.cap = cap
        self.neg_thresh = 0.55
        # ── 分层记忆 (矛盾运动的沉淀结构, 对齐 Google HOPE 记忆洋葱) ──
        # 高频层 (pairs): 矛盾的运动 — 工作/交互/近期学习, 快速更新, cap 小
        # 低频层 (core_pairs): 矛盾的扬弃沉淀 — 核心知识, 慢更新, cap 大, 防遗忘
        # 晋升 (受阻-通过, 命题3): 条目被检索命中 且 外部判定正确 → 通过分+1;
        #   判定错误 → -2 (污染惩罚). 通过分 ≥ promote_thresh → 沉淀到低频层.
        #   (对比旧"命中次数"判据: 被命中就计数, 会把高命中错误条目固化 —
        #   对照实验 B 实证: 混叠错误条目沉淀 1→0)
        self.pairs: list[tuple[Embed, str]] = []
        self.core_pairs: list[tuple[Embed, str]] = []
        self.core_cap = max(cap * 4, 2000)     # 沉淀层大容量 (稳定)
        self.promote_thresh = 2                 # 通过分阈值 (判定正确 ≥N 次 → 沉淀)
        self.pass_scores: dict[int, int] = {}   # 高频索引 -> 通过分 (答错 -2, 可负)
        # ── 分层遗忘 (选择性遗忘蓝图落地) ──
        # fast 层: 淘汰 = 质量(pass_score)×时间(未命中时长) 加权; 使用追踪 last_hit.
        # 沉淀合题: 质量(通过分) × 频率(hit_count) — 被频繁需要且判定正确 → 沉淀.
        # core 层: 巩固分(命中判定±) → 跌破 → 归档(archive, 非删除, 可恢复).
        self.last_hit: dict[int, int] = {}      # fast 索引 -> 最近命中 tick (时间维度)
        self.hit_counts: dict[int, int] = {}    # fast 索引 -> 累计命中次数 (频率维度)
        self.tick = 0                           # 内部时钟 (命中/learn 递增)
        self.core_scores: dict[int, int] = {}   # core 索引 -> 巩固分 (答错-2 跌破→归档)
        self.core_demote_thresh = -4            # core 巩固分跌破阈值 → 归档 (遗忘=归档)
        self.archive: list[tuple[Embed, str, str]] = []  # 归档区 (遗忘的缓存, 可恢复)
        self.last_layer = "fast"                # 最近回答来自哪层 (诊断)
        # ── 矛盾处理协议 (认知升级: 矛盾 = 高级受阻 = 需要更多实践) ──
        # 检测双通道 (LeCun 修正: 符号比较是初筛, 预测误差/实践反馈是精确):
        #   ① learn 符号检测 (粗筛, 分类潜在矛盾对)
        #   ② report_outcome 实践检测 (更新验证史, 精确裁决)
        # 裁决 = 验证历史加权 (正确率×验证量饱和), 非偏新 ±1.
        # 升级: 无法裁决的矛盾 → pending → respond 命中弃权 (矛盾受阻) → 触发查证.
        self.entry_meta: dict[int, dict] = {}   # fast 索引 -> {learned_at, correct_n, wrong_n}
        self.contradictions: list[dict] = []    # 矛盾记录 {a, b, sim_ctx, sim_ans, type, status, ...}
        self.last_block_reason = "none"         # respond 拒绝原因 (含 contradiction)
        self._last_fast_z: Optional[Embed] = None   # 最近 fast 命中的查询 z (判定反馈用)
        self._last_fast_ok = False              # 最近一次 respond 是否 fast 命中
        self._last_fast_idx = -1                # 最近 fast 命中的条目索引 (直接定位)
        self._last_core_ok = False              # 最近一次 respond 是否 core 命中 (巩固分)
        self._last_core_idx = -1                # 最近 core 命中的条目索引
        self.last_margin = 0.0                  # 最近一次回答的 margin (软校准保护用)
        # ── AdaJEPA 软校准 (落点5, TTA) ──
        # 判定正确 → 命中条目表征向查询微调 (EMA 小步长). stop-gradient 语义:
        # 只调条目 (少量"参数"), 不调查询; margin 充足才校准 → 防表征空间拉崩
        # (AdaJEPA: stop-gradient + 只更新最后几层 + 1步梯度).
        self.soft_align_enabled = soft_align
        self.soft_align_alpha = soft_align_alpha
        # ── 混叠互斥 (LeCun 可组合表征的检索式实现) ──
        # 批评: "相似≠因果" — 检索式把相似当相关, 混叠是必然.
        # 改进: learn 时高相似情境(cos 0.75-0.95)但答案不同 → 互加负样本,
        # 强制分离 (实体槽分离: 东京天气/上海天气 共享天气模板, 分开实体).
        self.conflict_gate = conflict_gate
        self.negatives: list[Embed] = []       # 答错标记: 这些情境附近别复用旧答案
        # ── 线性联想记忆 (LeCun 改进: 可微记忆进权重) ──
        # W: 情境 → 答案向量 的线性映射 (功能关系, 非几何距离).
        # delta 规则闭式在线更新 (无反向传播, CPU 微秒级);
        # 检索 miss → W 预测兜底 (模糊变体/未见变体的线性泛化).
        self.W: Optional[np.ndarray] = None    # (emb_dim, z_dim), 惰性初始化
        self.lam_alpha = 0.1                   # delta 学习率
        self.lam_ans_thresh = 0.55             # W 答案匹配置信底线
        # ── W 层遗忘 (弹性权重, EWC 对角线近似) ──
        # 慢衰减: 每次更新 W *= (1-λ) — 过时映射随时间淡出.
        # 验证强度 G: 正确更新时 G += |Δ| — 高 G 方向(被实践验证多)步长小=受保护,
        # 次要方向自由更新. "实践决定哪些信念值得坚守"的权重级实现.
        self.lam_decay = 0.01                  # 慢衰减率 (λ)
        self.G: Optional[np.ndarray] = None    # 验证强度矩阵 (与 W 同形)
        self.lam_g_reg = 0.05                  # G 保护强度 (步长缩放系数)
        self.answer_vecs: dict[int, Embed] = {}  # fast 索引 -> 答案向量 (配对检索)
        self._answer_encoder = None            # MiniLM 编码器 (惰性)
        self.rng = np.random.RandomState(seed)
        self.stats = {"hits": 0, "misses": 0}
        # ── 认知再生产生命周期 (落点4, 命题4: 健康判据 = 结构质变频率) ──
        # 范式更新率 = learned(新增) + covered(覆盖/扬弃) + promoted(沉淀)
        # 遗忘 = forgotten(cap 淘汰, 运动层流动) | 检索区分度 = margins 均值
        # 对应《智能论批判》认知病理学四型的可观测化
        self.lifecycle = {"learned": 0, "covered": 0, "promoted": 0,
                          "forgotten": 0, "margins_sum": 0.0, "margins_n": 0}
        # ── 校准曲线 (元认知: 从经验学"什么时候该回答") ──
        # 不再靠单一人工阈值: 每轮回答记录 (相似度桶, 对错) → 校准表
        # P(对|桶); 决策 = P(对|sim) >= calib_thresh 且 margin(top1-top2)
        # >= margin_thresh. 阈值从经验学出, 人工只留"风险偏好".
        self.calib: dict[int, list[int]] = {}   # 桶索引(0-10) -> [答过次数, 对次数]
        self.calib_thresh = calib_thresh        # 风险偏好: P(对) 底线 (人工旋钮)
        self.margin_thresh = margin_thresh      # 区分度底线 (模糊就不答)
        # ── 二维校准决策面 (LeCun 方法: 决策边界从经验学, 非手工阈值) ──
        # P(对|sim, margin) 从判定史学习 — 检索器输出几何量, "兼容度"
        # 由数据学出的能量函数解释 (对齐 LeCun: 能量/预测误差内化决策).
        # 有样本时数据驱动; 无样本时回退 margin_thresh 启发式 (渐进替换).
        self.calib2: dict[tuple[int, int], list[int]] = {}  # (sim桶,margin桶) -> [n, correct]
        self.min_calib_samples = 5              # 每桶最少样本才信任校准值
        self.default_p = 0.5                    # 无样本桶: 乐观先验 (先答后学, 同轮内自我校准)
        self.last_sim = 0.0                     # 最近一次回答的 sim (反馈用)

    def feedback(self, sim: float, margin: float, correct: bool) -> None:
        """判定器反馈: (sim, margin, 对错) → 更新校准表 (1D) 与二维决策面.
        这是元认知学习 — 模型从回答历史学"什么相似度/区分度可信".
        二维表 (sim桶, margin桶) 是 LeCun 式"从经验学出的决策边界":
        决策不再是手工阈值, 而是数据驱动的兼容度 P(对|sim, margin)."""
        bucket = int(np.clip(sim, 0.0, 1.0) * 10)
        st = self.calib.setdefault(bucket, [0, 0])
        st[0] += 1
        st[1] += int(correct)
        # 二维决策面 (margin 桶: 0-5, 覆盖 -0.5~0.5 范围)
        mb = int(np.clip(margin, -0.5, 0.5) * 5) + 5
        st2 = self.calib2.setdefault((bucket, mb), [0, 0])
        st2[0] += 1
        st2[1] += int(correct)

    def _calib2_p(self, sim: float, margin: float) -> float:
        """二维兼容度 P(对|sim, margin): 有样本 → 数据驱动; 无样本 →
        回退 margin_thresh 启发式 (margin 够 → 保守先验 0.5, 不够 → 0 拒绝).
        渐进替换: 数据积累后, 手工 margin_thresh 完全被学出的决策面取代."""
        b = int(np.clip(sim, 0.0, 1.0) * 10)
        mb = int(np.clip(margin, -0.5, 0.5) * 5) + 5
        st = self.calib2.get((b, mb))
        if st and st[0] >= 3:                 # 该桶有足够样本 → 数据驱动
            return st[1] / st[0]
        # 无样本: 相邻桶 (±1 sim, ±1 margin) 加权, 再没有 → 启发式兜底
        w_n, w_c = 0, 0
        for nb in (b - 1, b, b + 1):
            for nm in (mb - 1, mb, mb + 1):
                s = self.calib2.get((nb, nm))
                if s and s[0] >= 2:
                    w_n += s[0]
                    w_c += s[1]
        if w_n >= 2:
            return w_c / w_n
        # 启发式兜底 (等价旧行为): margin 足够 → 乐观先验; 不足 → 拒绝
        return self.default_p if margin >= self.margin_thresh else 0.0

    def _calib_p(self, sim: float) -> float:
        """P(对 | sim 桶): 校准表插值 + 无样本保守先验."""
        b = int(np.clip(sim, 0.0, 1.0) * 10)
        if b in self.calib and self.calib[b][0] >= self.min_calib_samples:
            n, c = self.calib[b]
            return c / n
        # 无样本: 用相邻桶 (±1) 加权, 再没有 → 保守先验
        w_n, w_c = 0, 0
        for nb in (b - 1, b, b + 1):
            st = self.calib.get(nb)
            if st and st[0] >= 2:
                w_n += st[0]
                w_c += st[1]
        if w_n >= self.min_calib_samples:
            return w_c / w_n
        return self.default_p

    def learn_negative(self, z: Embed) -> None:
        """标记: 该情境的回应答错了 → 下次不复用旧答案 (重新获取)."""
        self.negatives.append(np.asarray(z, np.float32).copy())
        if len(self.negatives) > self.cap:
            self.negatives.pop(0)

    def _lexical_coverage(self, q_text: str, answer: str) -> float:
        """查询实义词在答案中的覆盖率 (检索区分度精炼, 混叠防线):
        同构题的问题形式高相似 (余弦分不出), 但答案的实义词不同 —
        命中候选后验证"查询的关键词是否被答案回应": 覆盖率低 → 混叠
        (seed 条目被 convert 题命中的病根). 0.0-1.0."""
        qw = [w for w in re.split(r"[^a-z0-9]+", q_text.lower())
              if len(w) >= 3 and w not in
              ("what", "which", "that", "with", "from", "this", "have",
               "were", "about", "when", "where", "than", "been", "will",
               "into", "over", "more", "most", "would", "there", "their",
               "as", "of", "the", "and", "in", "on", "to", "for", "is",
               "are", "was", "not", "all", "you", "your")]
        if not qw:
            return 1.0                      # 无实义词 → 不设闸
        # 短答案 (<2 实义词: 数字/专名, 如 "13%"/"Paris") → 无法词汇验证,
        # 不设闸 (靠 margin 区分). 内化教材的选项文本即此类 — 词汇闸误杀
        # 会让短答案条目永远 miss (评估题实义词覆盖率恒 0).
        aw = [w for w in re.split(r"[^a-z0-9]+", answer.lower())
              if len(w) >= 3 and w not in
              ("the", "and", "for", "are", "was", "not", "all", "you",
               "your", "with", "that", "this", "have", "were", "from",
               "about", "than", "into", "over", "more", "most", "their",
               "there", "they", "them", "can", "could", "should", "would")]
        if len(aw) < 2:
            return 1.0                      # 短答案 → 不设闸
        al = answer.lower()
        return sum(1 for w in qw if w in al) / len(qw)

    def respond(self, z: Embed, task_text: str = "") -> Optional[str]:
        """检索最相似情境的回应. 未命中返回 None (触发探索/工具).
        决策三件套: ①负样本拦截 ②校准置信度 P(对|sim) ③区分度 margin.
        阈值从校准表学出, 不再依赖单一人工 min_sim.
        命中 fast 层 → 记录候选 (供 report_outcome 判定后沉淀); 晋升不再
        在 respond 内自动发生 (受阻-通过判据: 外部判定正确才加分)."""
        self._last_fast_ok = False
        self._last_fast_z = None
        if not self.pairs and not self.core_pairs:
            self.stats["misses"] += 1
            return None
        z = np.asarray(z, np.float32)
        zn = z / (np.linalg.norm(z) + 1e-9)
        for nz in self.negatives:
            nzn = np.asarray(nz, np.float32)
            nzn = nzn / (np.linalg.norm(nzn) + 1e-9)
            if float(np.dot(zn, nzn)) > self.neg_thresh:
                self.stats["misses"] += 1
                return None
        best, best_sim = None, -1.0
        second_sim = -1.0
        best_layer = "fast"
        best_idx = -1                     # fast 层内命中索引 (判定反馈定位用)
        for pool, layer in ((self.pairs, "fast"), (self.core_pairs, "core")):
            for i, (pz, text) in enumerate(pool):
                pzn = np.asarray(pz, np.float32)
                pzn = pzn / (np.linalg.norm(pzn) + 1e-9)
                s = float(np.dot(zn, pzn))
                if s > best_sim:
                    second_sim = best_sim
                    best, best_sim, best_layer = text, s, layer
                    best_idx = i
                elif s > second_sim:
                    second_sim = s
        # 决策: 校准置信度 + 区分度 (模糊 → 弃权, 防混叠)
        # 强命中 (top1 ≥ 0.97, 几乎明确) → 信任直接答 (条目自身/近重复).
        # 弱命中 → 要求 margin 区分度 (top1-top2 足够大才答).
        # 修正: 分离存储后 0.95-0.98 相似对 margin 天然 <0.05, 若一律要求
        # margin 会把"已学高相似知识"全拒 (自身查询也答不了).
        margin = best_sim - second_sim
        weak_hit = best_sim < 0.97
        # 矛盾弃权 (矛盾=高级受阻, 认知升级): 命中互斥矛盾对 →
        # 未裁决(双方有验证)或已裁决的输方 → 弃权 (拒绝复用不可靠答案 → 触发查证).
        if best is not None and self._in_pending_conflict(best, best_layer):
            self.last_block_reason = "contradiction"
            self.stats["contradiction_abstain"] = \
                self.stats.get("contradiction_abstain", 0) + 1
            self.stats["misses"] += 1
            return None
        if best is None or best_sim < self.min_sim:
            # 无知识 (retrieval_miss) → W 兜底 (权重记忆线性泛化)
            lam_text = self._lam_fallback(z)
            if lam_text is not None:
                return lam_text
            self.last_sim = best_sim if best is not None else 0.0
            self.stats["misses"] += 1
            return None
        if (weak_hit and self._calib2_p(best_sim, margin) < self.calib_thresh) \
                or self._calib_p(best_sim) < self.calib_thresh:
            # 低置信受阻 (LeCun: 兼容度 P(对|sim,margin) 从经验学, 非手工阈值)
            # → W 兜底 (权重记忆线性泛化).
            # 关键: 内化评估 1% 的根因是同科短答案条目 margin 全拒 (此分支),
            # W 兜底必须覆盖此分支才有意义.
            lam_text = self._lam_fallback(z)
            if lam_text is not None:
                return lam_text
            self.last_sim = best_sim
            self.last_layer = best_layer
            self.stats["misses"] += 1
            return None
        # 检索区分度精炼 (混叠防线): 查询实义词在答案中的覆盖率 —
        # 同构题余弦分不出 (seed 条目被 convert 题命中), 答案关键词能分.
        # 双条件闸: 仅当"覆盖率低 AND margin 低"(双重可疑: 词不相关+区分度差)
        # 才拒绝 — 真知识但措辞不同 (MMLU 题命中教材概念) 时 margin 高 → 放行;
        # 混叠命中 (同构条目互相接近) 时 margin 低 → 拒绝 (走探索).
        # weak_hit 限定: 强命中 (≥0.97) 信任, 词汇闸不误杀.
        if task_text and weak_hit and margin < self.margin_thresh * 2 \
                and self._lexical_coverage(task_text, best) < 0.35:
            self.last_sim = best_sim
            self.last_layer = best_layer
            self.stats["misses"] += 1
            self.stats["refined_reject"] = self.stats.get("refined_reject", 0) + 1
            return None
        self.last_sim = best_sim
        self.last_layer = best_layer
        self.last_margin = margin
        # 检索区分度记录 (落点4): margin 均值 = 记忆可辨识度
        self.lifecycle["margins_sum"] += margin
        self.lifecycle["margins_n"] += 1
        if best_layer == "fast":
            # 记录候选 (判定反馈直接定位索引 — 变体查询也能反馈到条目)
            self._last_fast_ok = True
            self._last_fast_z = zn
            self._last_fast_idx = best_idx
            # 频率追踪 (分层遗忘): 命中次数累积 (需要度) — last_hit 只在
            # 判定正确时更新 (report_outcome) — "近期命中"=近期被实践验证.
            self.tick += 1
            self.hit_counts[best_idx] = self.hit_counts.get(best_idx, 0) + 1
        elif best_layer == "core":
            # core 命中记录 (巩固分: 判定正确+1/错误-2, 跌破→归档)
            self._last_core_ok = True
            self._last_core_idx = best_idx
        self.stats["hits"] += 1
        return best

    def _learn_lam_vec(self, zn: Embed, a_true: Embed) -> None:
        """W 更新 (delta 规则) + W 层遗忘 (弹性权重, EWC 对角线近似):
        ① 慢衰减 W*=(1-λ): 过时映射随时间淡出 (防旧经验永远压过今天).
        ② 验证强度 G 保护: 正确更新时 G += |Δ| — 高 G 方向(被实践验证多)
           步长小=受保护 (实践决定哪些信念值得坚守); 次要方向自由更新."""
        zn = np.asarray(zn, np.float32)
        zn = zn / (np.linalg.norm(zn) + 1e-9)
        a_true = np.asarray(a_true, np.float32)
        if self.W is None:
            self.W = np.zeros((a_true.shape[0], zn.shape[0]), dtype=np.float32)
            self.G = np.zeros_like(self.W)
        self.W *= (1.0 - self.lam_decay)       # 慢衰减 (过时映射淡出)
        pred = self.W @ zn
        err = a_true - pred
        self.G += np.abs(np.outer(err, zn)) * 0.1   # 验证强度累积
        step = self.lam_alpha / (1.0 + self.lam_g_reg * self.G)  # G 保护步长
        self.W += step * np.outer(err, zn)

    def report_outcome(self, correct: bool) -> None:
        """外部判定反馈 (分层遗忘蓝图落地):
        ─ core 命中 → 巩固分 ±, 跌破阈值 → 归档 (遗忘=移入衰退区, 非删除, 可恢复).
        ─ fast 命中 → 受阻-通过沉淀判据合题: 质量(通过分) × 频率(命中次数加权),
          被反复需要且判定正确 → 沉淀 core; 答错 -2 (污染惩罚).
        同步校准表 (元认知): P(对|sim) 也从判定反馈学习.
        语义: "被实践反复考验且通过的经验才沉淀; 被实践反复否定的沉淀撤销"."""
        if getattr(self, "calib_update", True):
            self.feedback(self.last_sim, getattr(self, "last_margin", 0.0),
                          correct)   # 元认知: 1D+2D 校准表 (决策边界学习)
        # ── core 巩固分 (core 持续受审: 沉淀不是终点, 是过程) ──
        if self._last_core_ok:
            i = self._last_core_idx
            self._last_core_ok = False
            if 0 <= i < len(self.core_pairs):
                self.core_scores[i] = self.core_scores.get(i, 0) \
                    + (1 if correct else -2)
                self._update_contradictions(self.core_pairs[i][1], correct)
                if self.core_scores[i] <= self.core_demote_thresh:
                    # 遗忘 = 归档 (移入衰退区, 检索不再命中; 可被未来证据恢复)
                    pz, ptxt = self.core_pairs[i]
                    self.archive.append((pz, ptxt, "core_demoted"))
                    self.core_pairs.pop(i)
                    self.core_scores.pop(i, None)
                    self.lifecycle["forgotten"] += 1
                    self.stats["archived"] = self.stats.get("archived", 0) + 1
            return
        # ── fast 受阻-通过 (沉淀判据合题: 质量 × 频率) ──
        if not self._last_fast_ok or self._last_fast_z is None:
            return
        i = self._last_fast_idx
        if i < 0 or i >= len(self.pairs):
            return
        self.pass_scores[i] = self.pass_scores.get(i, 0) \
            + (1 if correct else -2)
        # 验证史更新 (矛盾裁决的数据源, LeCun: 实践是精确信号)
        meta = self.entry_meta.setdefault(i, {"learned_at": self.tick,
                                              "correct_n": 0, "wrong_n": 0})
        meta["correct_n" if correct else "wrong_n"] += 1
        if correct:
            self.last_hit[i] = self.tick   # 正确命中的时刻 (保护: 近期被验证)
        self._update_contradictions(self.pairs[i][1], correct)
        # 频率加权: 被频繁检索(需要度高) → 更容易沉淀; 质量仍主导 (答错 -2 抵消)
        freq = min(self.hit_counts.get(i, 0), 4) * 0.5
        if self.pass_scores[i] + freq >= self.promote_thresh:
            self.core_pairs.append(self.pairs[i])
            self.core_scores[len(self.core_pairs) - 1] = 0  # 新 core 巩固分起点
            self.pass_scores[i] = 0
            self.lifecycle["promoted"] += 1   # 沉淀 (被实践考验且通过)
        if correct:
            self._soft_align(self._last_fast_z, i)   # AdaJEPA 软校准 (判定正确)
            if i in self.answer_vecs:
                self._learn_lam_vec(self._last_fast_z,
                                    self.answer_vecs[i])  # W 强化 (判定反馈)
        self._last_fast_ok = False   # 只消费一次

    def _soft_align(self, zq: Embed, idx: int) -> None:
        """AdaJEPA 式软校准 (TTA, 落点5): 判定正确后把命中条目表征向查询微调.
        z_i' = normalize(z_i + alpha * (z_q - z_i)) — 条目向验证通过的查询移动
        一小步 → 相似问题自动受益 (泛化), 而非硬写入的精确匹配.
        stop-gradient 语义 (防表征空间拉崩):
          ① 只调命中的条目 (少量"参数")  ② 不调查询  ③ margin 充足才校准
          ④ 小步长 alpha=0.1 (每次重规划 1 步梯度的等价)."""
        if not self.soft_align_enabled:
            return
        if idx < 0 or idx >= len(self.pairs):
            return
        if self.last_margin < self.margin_thresh * 2:
            return                  # 区分度不足 → 校准会挤压 margin, 跳过
        pz, text = self.pairs[idx]
        pzn = np.asarray(pz, np.float32)
        pzn = pzn / (np.linalg.norm(pzn) + 1e-9)
        new_z = pz + self.soft_align_alpha * (np.asarray(zq, np.float32) - pzn)
        self.pairs[idx] = (new_z / (np.linalg.norm(new_z) + 1e-9), text)

    def learn(self, z: Embed, text: str, core: bool = False) -> None:
        """存入 (情境, 回应) 经验. 默认写高频层 (运动); core=True 写低频层 (沉淀).
        知识更新 (扬弃): 同一情境 (余弦>0.95) 学到新回应 → 覆盖旧回应.
        低频层保护: 只有 core=True 的显式学习才允许覆盖低频 (核心知识不被高频污染)."""
        z = np.asarray(z, np.float32).copy()
        zn = z / (np.linalg.norm(z) + 1e-9)
        # 覆盖检查: 先低频 (防高频经验覆盖核心知识, 除非 core=True), 再高频
        pools = [(self.core_pairs, True), (self.pairs, False)] if core else \
                [(self.pairs, False)]
        for pool, is_core in pools:
            for i, (pz, _) in enumerate(pool):
                pzn = np.asarray(pz, np.float32)
                pzn = pzn / (np.linalg.norm(pzn) + 1e-9)
                # 覆盖阈值 0.98 (LeCun 改进): 只有几乎相同的才覆盖.
                # 旧 0.95 会把"高相似但不同答案"的知识抹掉 (东京被上海覆盖) —
                # 实体分离的存储级实现: 相似对独立存储, 靠 margin 区分.
                if float(np.dot(zn, pzn)) > 0.98:
                    pool[i] = (z, text)   # 同情境覆盖: 新知识取代旧知识
                    if pool is self.pairs:
                        self.pass_scores.pop(i, None)   # 覆盖 → 旧通过分作废
                        self.entry_meta[i] = {"learned_at": self.tick,
                                              "correct_n": 0, "wrong_n": 0}
                        self._learn_lam(zn, text)       # W 监督更新 (权重记忆)
                    if pool is self.pairs:
                        self.lifecycle["covered"] += 1  # 扬弃 (旧知识被新知识取代)
                    return
        store = self.core_pairs if core else self.pairs
        # 矛盾检测 (粗筛, LeCun 修正: 符号比较只是初筛, 实践反馈才裁决)
        if self.conflict_gate:
            self._detect_contradictions(zn, text)
        store.append((z, text))
        if not core:
            self._learn_lam(zn, text)       # W 监督更新 (权重记忆)
            self._store_answer_vec(text)    # fast 索引 → 答案向量
            self.entry_meta[len(store) - 1] = {"learned_at": self.tick,
                                               "correct_n": 0, "wrong_n": 0}
        self.lifecycle["learned"] += 1                 # 新增范式
        limit = self.core_cap if core else self.cap
        if len(store) > limit:
            if store is self.pairs:
                # 加权淘汰 (分层遗忘): 质量分低 × 久未命中 → 先滚; 高分/近期命中保护.
                # fast 从"纯容量池"升级为"待沉淀池" — 好的进 core, 坏的先滚.
                evict = self._select_evict()
                store.pop(evict if evict >= 0 else 0)
                self.lifecycle["forgotten"] += 1       # 遗忘 (运动层流动)
                self._shift_fast_indices(evict)
            else:
                store.pop(0)
                self.lifecycle["forgotten"] += 1

    def _select_evict(self) -> int:
        """fast 加权淘汰选择: 质量分(pass_score)最低 × 未命中最久 → 优先淘汰.
        保护: 通过分达沉淀线 或 近期命中 (3 tick 内) → 不淘汰.
        全部受保护 → 返回 -1 (退化为最旧 pop(0))."""
        n = len(self.pairs)
        if n == 0:
            return -1
        now = self.tick
        protected = set(
            i for i in range(n)
            if self.pass_scores.get(i, 0) >= self.promote_thresh
            or (now - self.last_hit.get(i, 0)) < 3)
        cands = [i for i in range(n) if i not in protected]
        if not cands:
            return -1
        def score(i):
            q = self.pass_scores.get(i, 0)        # 质量 (低分优先淘汰)
            age = now - self.last_hit.get(i, 0)   # 时间 (久未命中优先淘汰)
            return q - age * 0.01
        return min(cands, key=score)

    def _shift_fast_indices(self, evict: int) -> None:
        """fast pop(evict) 后所有索引字典统一左移 (分层遗忘配套)."""
        if evict < 0:
            evict = 0
        for d in (self.pass_scores, self.hit_counts, self.last_hit,
                  self.answer_vecs, self.entry_meta):
            new = {}
            for k, v in d.items():
                if k > evict:
                    new[k - 1] = v
                elif k < evict:
                    new[k] = v
            d.clear()
            d.update(new)

    def _detect_contradictions(self, zn: Embed, text: str) -> None:
        """矛盾检测 (粗筛, LeCun 修正: 符号比较只是初筛, 分类/裁决靠实践):
        高相似情境(0.75-0.98) + 答案文本冲突 → 记录矛盾对 (type=conflict).
        redundant (答案同义) 已被 _texts_conflict 过滤 (不构成矛盾).
        互斥/互补的分类不靠符号预判 (反义词 rainy/sunny 词重叠高但语义互斥,
        符号分类不可靠) — 由实践验证史裁决 (report_outcome 的 _update_contradictions)."""
        for pool in (self.pairs, self.core_pairs):
            for pz, ptext in pool:
                pzn = np.asarray(pz, np.float32)
                pzn = pzn / (np.linalg.norm(pzn) + 1e-9)
                s = float(np.dot(zn, pzn))
                if 0.75 < s <= 0.98 and self._texts_conflict(text, ptext):
                    self.stats["conflicts"] = self.stats.get("conflicts", 0) + 1
                    self._log_contradiction(zn, text, pz, ptext, s)
                    break

    def _log_contradiction(self, za, ta, zb, tb, sim_ctx) -> None:
        """记录矛盾对 (去重: 按文本对). 已有 → 更新; 新增 → append."""
        for c in self.contradictions:
            if {c["a"][1], c["b"][1]} == {ta, tb}:
                c["sim_ctx"] = sim_ctx
                c["tick"] = self.tick
                return
        self.contradictions.append({
            "a": (np.asarray(za, np.float32).copy(), ta),
            "b": (np.asarray(zb, np.float32).copy(), tb),
            "sim_ctx": sim_ctx, "type": "conflict",
            "status": "pending", "winner": -1,
            "verified_a": 0, "verified_b": 0, "tick": self.tick,
        })

    @staticmethod
    def _wilson(c: int, n: int, z: float = 1.96) -> tuple[float, float]:
        """Wilson score 比例置信区间 (95%): P(对) 的统计可信范围.
        LeCun 修正: 裁决用贝叶斯置信区间, 不用手工阈值 (0.6/2/0.15)."""
        if n <= 0:
            return 0.0, 0.0
        p = c / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = (z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
        return max(center - half, 0.0), min(center + half, 1.0)

    def _adjudicate(self, ca: int, wa: int, cb: int, wb: int) -> int:
        """Wilson 置信区间裁决 (LeCun 修正: 统计置信, 无手工阈值):
        双方 95% 区间不重叠 → 高置信者胜 (有统计意义的分胜负);
        区间重叠 → -1 (证据不足以裁决 → 待求证, 矛盾=高级受阻)."""
        na, nb = ca + wa, cb + wb
        if na == 0 and nb == 0:
            return -1
        la, ua = self._wilson(ca, na)
        lb, ub = self._wilson(cb, nb)
        if ua < lb:      # a 区间完全低于 b → b 胜
            return 1
        if ub < la:      # b 区间完全低于 a → a 胜
            return 0
        return -1        # 区间重叠 → 无法裁决

    def _update_contradictions(self, text: str, correct: bool) -> None:
        """实践反馈裁决矛盾 (LeCun 修正: 宽容等待防误判, 错误≥2 才判负):
          both-verified: 双方都有正验证 → 正确率裁决:
            双方 ≥60% → complementary (条件/时间差异, 宽容并存, 持续受审)
            否则 → 验证历史加权裁决 (resolved/pending)
          error≥2: 一方错误 ≥2 (实践强烈否决) → 该方判负 (另一方有正验证则胜)
          else: pending (宽容等待 — 不弃权, 双方都有机会验证; 防"先验证者判死后验证者")"""
        for c in self.contradictions:
            if c["a"][1] == text:
                c["verified_a"] += 1 if correct else -1
            elif c["b"][1] == text:
                c["verified_b"] += 1 if correct else -1
            else:
                continue
            va, vb = c["verified_a"], c["verified_b"]
            ca, wa = max(va, 0), max(-va, 0)
            cb, wb = max(vb, 0), max(-vb, 0)
            if va > 0 and vb > 0:
                # 双方都被验证: 区间中点 (Wilson center = P(对) 最佳估计) 均 ≥0.5
                # (不差于随机基线) 且区间重叠 (无统计显著差异) → 互补并存;
                # 否则区间裁决 (不重叠 → 分胜负; 重叠 → 待求证)
                la, ua = self._wilson(ca, ca + wa)
                lb, ub = self._wilson(cb, cb + wb)
                center_a = (la + ua) / 2
                center_b = (lb + ub) / 2
                overlap = (ua >= lb) and (ub >= la)
                if center_a >= 0.5 and center_b >= 0.5 and overlap:
                    c["type"] = "complementary"
                    c["status"] = "resolved"
                    c["winner"] = -1          # 无输方 (并存合法)
                else:
                    w = self._adjudicate(ca, wa, cb, wb)
                    c["status"] = "resolved" if w >= 0 else "pending"
                    c["winner"] = w
            elif wa >= 2 or wb >= 2:
                # 一方错误 ≥2 (实践强烈否决) → 判负 (另一方有正验证则胜)
                if va > 0:
                    c["status"] = "resolved"; c["winner"] = 0
                elif vb > 0:
                    c["status"] = "resolved"; c["winner"] = 1
                else:
                    c["status"] = "pending"
            else:
                # 宽容等待: 无错误或错误<2 → 不裁决不弃权 (双方都有机会验证)
                c["status"] = "pending"

    def _in_pending_conflict(self, text: str, layer: str) -> bool:
        """命中条目是否在"不可复用"的矛盾状态:
        resolved 且 winner>=0 (互斥裁决) 且本条目是输方 → 弃权 (输方不被复用).
        complementary (双方都验证通过, 条件差异) / pending (宽容等待中) →
        不弃权 — 并存合法, 不冤枉未裁决方."""
        for c in self.contradictions:
            if c["status"] == "resolved" and c["winner"] >= 0:
                loser = c["a"][1] if c["winner"] == 1 else c["b"][1]
                if text == loser:
                    return True
        return False

    def n(self) -> int:
        return len(self.pairs) + len(self.core_pairs)

    @staticmethod
    def _texts_conflict(a: str, b: str) -> bool:
        """答案是否实质不同 (矛盾检测): 双方都有"独有实义词" (答案实质不同)
        且共享比例 < 0.85 (非几乎同义).
        反义词对 (rainy/sunny 共享 tokyo/weather 67%) → 冲突 ✓;
        同义超集 ("...rainy today" vs "...rainy") → 不冲突 (一方无独有词)."""
        stop = {"the", "and", "for", "are", "was", "not", "all", "you",
                "your", "with", "that", "this", "have", "were", "from",
                "about", "than", "into", "over", "more", "most", "their",
                "there", "they", "them", "can", "could", "should", "would",
                "is", "has", "had", "its", "it", "in", "on", "of", "to"}
        wa = set(w for w in re.split(r"[^a-z0-9]+", a.lower())
                 if len(w) >= 3 and w not in stop)
        wb = set(w for w in re.split(r"[^a-z0-9]+", b.lower())
                 if len(w) >= 3 and w not in stop)
        if not wa or not wb:
            return False
        if not (wa - wb) or not (wb - wa):
            return False                      # 一方是另一方子集 → 同义
        inter = wa & wb
        return len(inter) / max(len(wa), len(wb)) < 0.85

    # ── 线性联想记忆 (LAM): 可微记忆进权重 ──────────────
    def _get_answer_encoder(self):
        """答案编码器 (惰性): MiniLM 单例 (kernel.ensure_semantic 已初始化)."""
        if self._answer_encoder is None:
            try:
                from mini_encoder import get_encoder
                self._answer_encoder = get_encoder()
            except Exception:
                self._answer_encoder = None
        return self._answer_encoder

    def _vec_answer(self, text: str) -> Optional[Embed]:
        """答案文本 → 语义向量 (编码失败返回 None → 跳过 LAM)."""
        enc = self._get_answer_encoder()
        if enc is None or not enc.available:
            return None
        try:
            v = enc.encode(str(text)[:200])
            if v is None:
                return None
            return np.asarray(v, np.float32)
        except Exception:
            return None

    def _store_answer_vec(self, text: str) -> None:
        """存答案向量: fast 索引 -> 答案向量 (W 兜底时匹配用)."""
        a = self._vec_answer(text)
        if a is not None and self.pairs:
            self.answer_vecs[len(self.pairs) - 1] = a

    def _learn_lam(self, zn: Embed, text: str) -> None:
        """W 监督更新 (delta 规则, 闭式无反向传播):
        W += alpha * (a_true - W·z) ⊗ z   — 情境特征组合 → 答案的线性映射.
        学习信号: learn 时的 ground truth 答案 (监督)."""
        a_true = self._vec_answer(text)
        if a_true is None:
            return
        self._learn_lam_vec(zn, a_true)

    def predict_lam(self, z: Embed) -> Optional[Embed]:
        """W 预测: a_pred = W·z (O(1) 矩阵乘). W 未学 → None."""
        if self.W is None:
            return None
        zn = np.asarray(z, np.float32)
        zn = zn / (np.linalg.norm(zn) + 1e-9)
        return self.W @ zn

    def best_answer_match(self, a_pred: Embed) -> tuple[int, float, float]:
        """a_pred 与条目答案向量的最近匹配 → (fast 索引, top1 cos, top2 cos)."""
        ap = np.asarray(a_pred, np.float32)
        ap = ap / (np.linalg.norm(ap) + 1e-9)
        best_i, best_c, second_c = -1, -1.0, -1.0
        for i, av in self.answer_vecs.items():
            avn = np.asarray(av, np.float32)
            avn = avn / (np.linalg.norm(avn) + 1e-9)
            c = float(np.dot(ap, avn))
            if c > best_c:
                second_c = best_c
                best_i, best_c = i, c
            elif c > second_c:
                second_c = c
        return best_i, best_c, second_c

    def _lam_fallback(self, z: Embed) -> Optional[str]:
        """检索 miss 后的 W 兜底: 预测答案向量 → 匹配已知答案.
        双条件 (与检索 margin 哲学一致): 预测必须"专属"某个答案 —
          top1 cos ≥ thresh (W 有把握) AND top1-top2 ≥ 0.15 (不模糊).
        无关查询的预测与所有答案低相似且无区分 → 拒绝 (诚实)."""
        a_pred = self.predict_lam(z)
        if a_pred is None or not self.answer_vecs:
            return None
        idx, cos, second = self.best_answer_match(a_pred)
        if idx < 0 or cos < self.lam_ans_thresh or cos - second < 0.15:
            return None
        text = self.pairs[idx][1]
        self.stats["lam_hits"] = self.stats.get("lam_hits", 0) + 1
        self.last_layer = "lam"
        self.last_sim = cos
        return text

    def health_dict(self) -> dict:
        """认知再生产健康诊断 (落点4, 对齐《智能论批判》认知病理学四型):
        范式更新率 / 沉淀率 / 检索区分度 / 校准形状 → 病理警示.
        健康判据不是"记住了多少" (命题4: 是"结构是否还在质变")."""
        lc = self.lifecycle
        flux = lc["learned"] + lc["covered"] + lc["promoted"]
        margin_avg = (lc["margins_sum"] / lc["margins_n"]
                      if lc["margins_n"] else None)
        # 校准形状: 高相似桶 (0.9-1.0) 的 P(对) — 表征失真检测
        # (价值-现实脱节: 内部模型与外部反馈耦合断裂)
        hi_p, hi_n = 0, 0
        for b in (9, 10):
            st = self.calib.get(b)
            if st and st[0] >= 2:
                hi_n += st[0]
                hi_p += st[1]
        calib_shape = []
        for b in range(11):
            st = self.calib.get(b)
            if st and st[0] >= 2:
                calib_shape.append([b, round(st[1] / st[0], 2)])
        dist = {"flux": flux,
                "learned": lc["learned"], "covered": lc["covered"],
                "promoted": lc["promoted"], "forgotten": lc["forgotten"],
                "consolidation_rate": (lc["promoted"] / max(flux, 1)),
                "margin_avg": (round(margin_avg, 3) if margin_avg else None),
                "calib_shape": calib_shape}
        # ── 病理警示 (认知病理学四型的可观测判据) ──
        warns = []
        if margin_avg is not None and margin_avg < 0.10:
            warns.append("检索退化 (叙事碎裂): 区分度 margin 均值 %.3f < 0.10 — "
                         "记忆条目过度相似, 经验碎片化" % margin_avg)
        if hi_n >= 5 and hi_p / hi_n < 0.4:
            warns.append("表征失真 (价值-现实脱节): 高相似桶正确率 "
                         "%.0f%% < 40%% — 内部表征与外部反馈脱节"
                         % (100 * hi_p / hi_n))
        if flux >= 50 and self.stats["hits"] + self.stats["misses"] > 0:
            hr = self.stats["hits"] / (self.stats["hits"] + self.stats["misses"])
            if hr < 0.3:
                warns.append("认知僵化 (误差饱和): 已学 %d 条但命中率 %.0f%% — "
                             "学得多但用不上, 更新停滞" % (flux, 100 * hr))
        if len(self.pairs) >= self.cap and lc["forgotten"] == 0 \
                and lc["learned"] >= self.cap:
            warns.append("遗忘缺失: 高频层已满但零淘汰 — 检查 cap 机制")
        dist["warnings"] = warns
        return dist

    def stats_dict(self) -> dict:
        return {**self.stats, "n": self.n(),
                "fast": len(self.pairs), "core": len(self.core_pairs),
                "promoted": sum(1 for i, s in self.pass_scores.items()
                                if s >= self.promote_thresh - 1),
                "hit_rate": (self.stats["hits"] /
                             max(self.stats["hits"] + self.stats["misses"], 1)),
                "lam": {"W": None if self.W is None else self.W.shape,
                        "lam_hits": self.stats.get("lam_hits", 0),
                        "answer_vecs": len(self.answer_vecs)},
                "contradictions": {
                    "total": len(self.contradictions),
                    "pending": sum(1 for c in self.contradictions
                                   if c["status"] == "pending"),
                    "resolved": sum(1 for c in self.contradictions
                                    if c["status"] == "resolved"),
                    "abstain": self.stats.get("contradiction_abstain", 0),
                },
                "lifecycle": self.lifecycle}
