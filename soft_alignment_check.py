"""
AdaJEPA 软校准验证 (soft_alignment_check.py) — 落点5
=====================================================
验证: ①泛化增益 — 判定正确后条目表征向实际查询分布移动 → 未见变体命中提升
     (AdaJEPA PushObj 未见形状翻倍的检索式等价)
     ②stop-gradient 保护 — 多轮校准后 margin 不显著退化 (表征空间不拉崩)
     ③错误不校准 — 判定错误的混叠条目表征不动 (防错误固化)
     ④kernel 集成 — soft_align 配置生效
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import sys

import numpy as np

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "body"))


def mkz(seed):
    r = np.random.RandomState(seed)
    v = r.randn(128).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def test_generalization_gain():
    """校准后未见变体命中提升 (核心)."""
    print("=" * 62)
    print("[场景1] 软校准泛化增益 (AdaJEPA TTA 核心)")
    print("=" * 62)
    from respond_learner import RespondLearner
    r = RespondLearner(min_sim=0.45, cap=50, seed=0)
    z_a = mkz(100)
    d = mkz(999)                      # 固定偏移方向 (实际查询分布的偏移)
    r.learn(z_a, "paris")
    # 校准前: 测试"未见变体" (同偏移方向) 的命中 sim
    z_test = z_a + 0.5 * d
    z_test = z_test / (np.linalg.norm(z_test) + 1e-9)
    sim_before = float(np.dot(z_test, z_a))
    # 5 次校准: 用同方向变体查询命中判定正确 → 条目向实际查询分布移动
    for i in range(5):
        zq = z_a + 0.5 * d + 0.05 * mkz(300 + i)
        zq = zq / (np.linalg.norm(zq) + 1e-9)
        if r.respond(zq) is not None:
            r.report_outcome(True)
    # 校准后: 条目已向查询分布移动, 测同一未见变体
    pz_after, _ = r.pairs[0]
    sim_after = float(np.dot(z_test, pz_after))
    print(f"  未见变体命中 sim: {sim_before:.3f} → {sim_after:.3f} "
          f"(提升 {sim_after - sim_before:+.3f})")
    ok = sim_after > sim_before + 0.005
    print(f"  {'✅ 软校准提升未见变体命中 (泛化增益 — 硬写入无此效果)' if ok else '❌'}")
    return ok


def test_margin_protection():
    """多轮校准后 margin 不显著退化 (stop-gradient 保护)."""
    print("\n" + "=" * 62)
    print("[场景2] margin 保护 (表征空间不拉崩)")
    print("=" * 62)
    from respond_learner import RespondLearner
    cal = RespondLearner(min_sim=0.45, cap=50, seed=0, soft_align=True)
    noc = RespondLearner(min_sim=0.45, cap=50, seed=0, soft_align=False)
    for i in range(8):                       # 8 条多样化知识
        cal.learn(mkz(400 + i), f"ans{i}")
        noc.learn(mkz(400 + i), f"ans{i}")
    for _ in range(3):                       # 3 轮使用 + 判定正确 (校准)
        for i in range(8):
            for r_ in (cal, noc):
                if r_.respond(mkz(400 + i)) is not None:
                    r_.report_outcome(True)
    m_cal = cal.lifecycle["margins_sum"] / cal.lifecycle["margins_n"]
    m_noc = noc.lifecycle["margins_sum"] / noc.lifecycle["margins_n"]
    print(f"  margin 均值: 校准版 {m_cal:.3f} vs 无校准 {m_noc:.3f} "
          f"(退化 {m_cal - m_noc:+.3f})")
    ok = m_cal > m_noc - 0.05                # 退化 < 0.05 (容忍)
    print(f"  {'✅ stop-gradient 保护: 校准未显著挤压区分度' if ok else '❌'}")
    return ok


def test_error_no_align():
    """判定错误 → 表征不动 (防错误固化)."""
    print("\n" + "=" * 62)
    print("[场景3] 错误不校准 (混叠条目不因校准固化)")
    print("=" * 62)
    from respond_learner import RespondLearner
    r = RespondLearner(min_sim=0.45, cap=20, seed=0, soft_align=True)
    z5 = mkz(500)
    r.learn(z5, "wrong-answer")              # 混叠错误条目 (东京→上海答案)
    before = np.array(r.pairs[0][0])
    for _ in range(3):                       # 3 轮命中判定错
        if r.respond(z5) is not None:
            r.report_outcome(False)
    after = np.array(r.pairs[0][0])
    moved = float(np.linalg.norm(after - before))
    print(f"  表征移动量: {moved:.6f} (判定错误 → 不应移动)")
    ok = moved < 1e-6
    print(f"  {'✅ 错误条目表征不动 (软校准只在判定正确时触发)' if ok else '❌'}")
    return ok


def test_kernel_config():
    """kernel 集成: soft_align 配置生效 (构造参数透传, 不经网络)."""
    print("\n" + "=" * 62)
    print("[场景4] kernel 配置透传 (构造参数)")
    print("=" * 62)
    from respond_learner import RespondLearner
    from plugin_config import PluginConfig
    cfg = PluginConfig(seed=1, soft_align=True, soft_align_alpha=0.15)
    r_on = RespondLearner(soft_align=cfg.soft_align,
                          soft_align_alpha=cfg.soft_align_alpha)
    cfg_off = PluginConfig(seed=1, soft_align=False)
    r_off = RespondLearner(soft_align=cfg_off.soft_align)
    ok = (r_on.soft_align_enabled is True
          and abs(r_on.soft_align_alpha - 0.15) < 1e-9
          and r_off.soft_align_enabled is False)
    print(f"  开启 alpha=0.15 透传: {r_on.soft_align_enabled} "
          f"| 禁用透传: {r_off.soft_align_enabled}")
    print(f"  {'✅ 配置 → RespondLearner 参数透传 (kernel 同构)' if ok else '❌'}")
    return ok


if __name__ == "__main__":
    results = [test_generalization_gain(), test_margin_protection(),
               test_error_no_align(), test_kernel_config()]
    print("\n" + "=" * 62)
    print(f"总体: {'✅ AdaJEPA 软校准全部验证通过' if all(results) else '❌ 有失败'}")
    print("=" * 62)
    sys.exit(0 if all(results) else 1)
