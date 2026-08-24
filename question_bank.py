"""
问题库 (question_bank.py) — 双向公式的问题侧
==========================================
"问题本身也是可以被训练的, 这会赋予 JEPA 持续学习的动力."

双向公式:
  正向 (回答训练): 问题 → 查证 → 答案 → 内化 (已有: learner_loop)
  反向 (提问训练): 答案 → 关联问题 → 入问题库 → 成为新学习目标 (本模块)

问题条目可训练:
  - 学会一个问题 → 价值上升 + 从答案文本繁殖关联问题 (持续学习动力引擎)
  - 多次学不会 → 价值衰减 (哪些问题不值得问 — 问题选择也是学出来的)
  - 掌握的问题 → 归档, 释放到新问题

这使 JEPA 不依赖外部任务: 学完一个 → 繁殖新问题 → 继续学 (内在动机).
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

STOPWORDS = {"what", "which", "that", "with", "from", "this", "have",
             "were", "about", "when", "where", "than", "been", "will",
             "into", "over", "more", "most", "would", "there", "their",
             "the", "and", "for", "are", "was", "has", "had", "its",
             "you", "your", "not", "but", "they", "them", "can", "could",
             "should", "would", "is", "it", "in", "on", "of", "to"}


class QuestionBank:
    """问题库: JEPA 想知道什么 (问题条目可训练)"""

    def __init__(self, seed: int = 0):
        self.questions: dict[str, dict] = {}   # question -> state
        self.rng = np.random.RandomState(seed)

    # ── 问题条目管理 ──────────────────────────────────────
    def add(self, question: str, value: float = 0.5) -> None:
        """加入问题 (默认价值 0.5). 已存在则价值取 max (不降级)."""
        q = question.strip()
        if not q:
            return
        if q not in self.questions:
            self.questions[q] = {"value": value, "attempts": 0,
                                 "mastered": False}
        else:
            self.questions[q]["value"] = max(self.questions[q]["value"],
                                             value)

    def pick(self) -> Optional[str]:
        """选下一个要学的: 未掌握中价值最高 (带探索: 随机扰动)."""
        cands = [(q, st["value"] * (1.0 + self.rng.uniform(-0.2, 0.2)))
                 for q, st in self.questions.items()
                 if not st["mastered"] and st["attempts"] < 5]
        if not cands:
            return None
        cands.sort(key=lambda x: -x[1])
        return cands[0][0]

    def feedback(self, question: str, learned: bool,
                 answer_text: str = "") -> Optional[str]:
        """学习反馈 (问题可训练的核心):
        学会 → 价值升 + 繁殖关联问题 (返回新问题); 失败 → 降权/保留."""
        st = self.questions.get(question)
        if st is None:
            return None
        st["attempts"] += 1
        if learned:
            st["mastered"] = True
            st["value"] = min(1.0, st["value"] + 0.3)
            return self._spawn(question, answer_text)
        st["value"] *= 0.6          # 没学会 → 降权 (但保留, 可再试)
        return None

    # ── 问题生成 (双向公式的反向: 答案 → 问题) ────────────
    def _spawn(self, question: str, answer_text: str) -> Optional[str]:
        """从答案文本提取新实体 → 构造关联问题 (繁殖).
        学会一个 → 想知道更多: 答案里出现的新名词成为新问题."""
        if not answer_text:
            return None
        q_words = set(re.split(r"[^a-z0-9]+", question.lower()))
        a_words = re.split(r"[^a-z0-9]+", answer_text.lower())
        new_entities = []
        for w in a_words:
            if (len(w) >= 5 and w not in STOPWORDS
                    and w not in q_words and w not in new_entities):
                new_entities.append(w)
        if not new_entities:
            return None
        target = new_entities[0]
        # 新问题模板: 从答案语境推断 (数字/百分比 → 统计; 概念词 → 定义)
        if re.search(r"\d|%|percent|rate|share", answer_text.lower()):
            new_q = f"what is the {target} statistic or rate"
        else:
            new_q = f"what does {target} mean"
        self.add(new_q, value=0.6)   # 繁殖的问题价值略高 (好奇心)
        return new_q

    # ── 诊断 ──────────────────────────────────────────────
    def stats(self) -> dict:
        n = len(self.questions)
        mastered = sum(1 for s in self.questions.values() if s["mastered"])
        return {"n": n, "mastered": mastered,
                "open": n - mastered,
                "value_sum": round(sum(s["value"]
                                       for s in self.questions.values()), 2)}
