"""分层记忆验证 (memory_layers_check.py) — 矛盾运动沉淀结构
验证: 持续学习学下去 = 学新不毁旧 (晋升到低频层的核心知识在高压学习下仍保留)."""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import sys
import numpy as np

sys.path.insert(0, os.path.join(REPO_ROOT, "body"))
from respond_learner import RespondLearner
from mini_encoder import get_encoder

enc = get_encoder()
enc.ensure_loaded()


def z(t):
    return enc.encode(t)


def main():
    print("=" * 60)
    print("分层记忆验证 (高频=运动 / 低频=沉淀)")
    print("=" * 60)
    r = RespondLearner(min_sim=0.5, cap=8, seed=0)   # 高频 cap 小: 高压环境

    # 阶段 1: 学 8 条多样化知识 (高频满) — 避免同构导致 margin 拒绝
    lessons = [
        ("what is the capital of france", "france: capital Paris"),
        ("what does photosynthesis mean", "photosynthesis: plants use sunlight to make food"),
        ("what is the largest planet", "jupiter is the largest planet"),
        ("how many continents are there", "there are 7 continents"),
        ("what is the currency of japan", "japan: currency yen"),
        ("what is the speed of light", "light speed is 300000 km per second"),
        ("what is the chemical symbol for gold", "gold symbol is Au"),
        ("what is the boiling point of water", "water boils at 100 celsius"),
    ]
    for q, a in lessons:
        r.learn(z(q), a)
    print(f"阶段1: 学 8 条 → fast={len(r.pairs)} core={len(r.core_pairs)}")

    # 阶段 2: 反复命中前 3 条 + 判定正确 (受阻-通过判据 → 晋升)
    # (沉淀判据 = 被实践考验且通过: 判定正确 → 通过分+1, ≥2 晋升 core)
    for _ in range(3):
        for q, _ in lessons[:3]:
            r.respond(z(q))
            r.report_outcome(True)   # 外部判定: 回答正确 → 通过分+1
    print(f"阶段2: 命中+判定正确 前3条 ×3 → fast={len(r.pairs)} "
          f"core={len(r.core_pairs)} (晋升 {len(r.core_pairs)} 条)")

    # 阶段 3: 高压学新知识 20 条 (高频淘汰最旧 — 运动层流动)
    for i in range(20):
        r.learn(z(f"what is newtopic{i}"), f"newtopic{i}: info {i}")
    print(f"阶段3: 再学 20 条新知识 → fast={len(r.pairs)} "
          f"core={len(r.core_pairs)}")

    # 验证: 晋升的核心知识是否保留 (学新不毁旧)
    kept = []
    for q, _ in lessons[:3]:
        text = r.respond(z(q))
        kept.append(text is not None)
    print(f"\n核心知识保留: 前3条 → {['✓' if k else '✗' for k in kept]}")
    # 验证: 未晋升的旧知识被自然淘汰 (正常遗忘 — 运动层流动)
    old_lost = r.respond(z("what is the speed of light")) is None
    print(f"未晋升旧知识淘汰: 光速 → {'✓ 淘汰' if old_lost else '✗ 仍存'}")

    ok = all(kept) and old_lost
    print(f"\n{'✅ 分层记忆成立: 学新不毁旧 (核心沉淀 + 运动流动)' if ok else '❌'}")
    print(f"   统计: fast={len(r.pairs)} core={len(r.core_pairs)}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
