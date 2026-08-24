"""
认知再生产指标验证 (status_metrics_check.py) — 落点4
=====================================================
验证: ①正常学习流 lifecycle 指标合理 (learned/covered/promoted/forgotten)
     ②病理场景触发警示: 同构条目 → margin 退化 → "检索退化"; 
       margin 0 全命中同构 → "表征失真"; 多工具只用少数 → "价值萎缩"
对应《智能论批判》认知病理学四型的工程化可观测判据.
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import sys

import numpy as np

sys.path.insert(0, os.path.join(REPO_ROOT, "body"))
from respond_learner import RespondLearner


def mkz(seed):
    r = np.random.RandomState(seed)
    v = r.randn(128).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def test_healthy_flow():
    """正常持续学习流: 学 → 判定 → 晋升 → 高压冲击 → 指标合理."""
    print("=" * 62)
    print("[场景1] 正常持续学习流 (健康)"
          "\n  8 条多样化知识 → 判定正确晋升 → 高压 20 条冲击")
    print("=" * 62)
    r = RespondLearner(min_sim=0.45, cap=8, seed=0)
    topics = [mkz(100 + i) for i in range(8)]
    for i, z in enumerate(topics):
        r.learn(z, f"answer{i}")
    # 使用 + 判定正确 ×3 轮
    for _ in range(3):
        for i, z in enumerate(topics):
            t = r.respond(z)
            if t is not None:
                r.report_outcome(t == f"answer{i}")
    # 高压冲击
    for i in range(20):
        r.learn(mkz(2000 + i), f"junk{i}")
    h = r.health_dict()
    lc = r.lifecycle
    print(f"  生命周期: 新增{lc['learned']} 覆盖{lc['covered']} "
          f"沉淀{lc['promoted']} 遗忘{lc['forgotten']}")
    print(f"  范式更新率 flux={h['flux']} | 沉淀率 {h['consolidation_rate']:.0%} "
          f"| margin 均值 {h['margin_avg']}")
    print(f"  校准形状: {h['calib_shape'][:4]}")
    print(f"  警示: {h['warnings'] or '无 (健康)'}")
    ok = (lc["promoted"] >= 3 and lc["forgotten"] >= 5
          and h["flux"] >= 25 and not h["warnings"])
    print(f"  {'✅ 指标合理: 结构在质变 (新增+沉淀), 运动层流动 (遗忘)' if ok else '❌'}")
    return ok


def test_retrieval_degradation():
    """病理: 同构条目堆积 → margin 退化 → 检索退化警示."""
    print("\n" + "=" * 62)
    print("[场景2] 同构条目堆积 (病理 — 检索退化/叙事碎裂)")
    print("  30 条 'what is X' 模板 (彼此余弦 ~0.9) → margin 趋零")
    print("=" * 62)
    r = RespondLearner(min_sim=0.45, cap=100, seed=0, margin_thresh=0.01)
    # 同构模板: 共享前缀 + 个体噪声 → 高相似 (能命中但 margin 极低)
    base = mkz(500)
    entries = []
    for i in range(30):
        noise = mkz(600 + i)
        # cos ≈ 0.94 (<0.95 不触发覆盖, 高相似但独立 — read↔search 类污染)
        zv = base + 0.25 * noise
        zv = zv / (np.linalg.norm(zv) + 1e-9)
        entries.append(zv)
        r.learn(zv, f"def{i}")
    for zv in entries:   # 用条目本身查询 → 命中但 margin ~0.08 (难以区分)
        r.respond(zv)
    h = r.health_dict()
    print(f"  margin 均值: {h['margin_avg']} (健康应 >0.15)")
    print(f"  警示: {h['warnings']}")
    ok = h["margin_avg"] is not None and h["margin_avg"] < 0.10 \
        and any("检索退化" in w for w in h["warnings"])
    print(f"  {'✅ 检索退化警示触发 (病理被监测)' if ok else '❌'}")
    return ok


def test_calib_distortion():
    """病理: 高相似桶判定全错 → 表征失真警示 (价值-现实脱节)."""
    print("\n" + "=" * 62)
    print("[场景3] 表征失真 (病理 — 价值-现实脱节)")
    print("  高相似条目判定全部答错 → 校准表 0.9+ 桶 P(对) 崩塌")
    print("=" * 62)
    r = RespondLearner(min_sim=0.45, cap=20, seed=0)
    z = mkz(1)
    r.learn(z, "wrong")
    for _ in range(8):   # 8 次命中但判定全错 (模拟表征与外部反馈脱节)
        r.respond(z)
        r.report_outcome(False)
    h = r.health_dict()
    print(f"  高相似桶校准: {h['calib_shape']}")
    print(f"  警示: {h['warnings']}")
    ok = any("表征失真" in w for w in h["warnings"])
    print(f"  {'✅ 表征失真警示触发 (病理被监测)' if ok else '❌'}")
    return ok


def test_value_atrophy():
    """病理: 工具调用单调 → 价值萎缩警示."""
    print("\n" + "=" * 62)
    print("[场景4] 工具单调 (病理 — 价值萎缩) — kernel 层集成验证")
    print("=" * 62)
    import os
    sys.path.insert(0, REPO_ROOT)
    os.environ.setdefault("JEPA_SILENT", "1")
    from kernel import JepaBody
    from plugin_config import PluginConfig
    body = JepaBody(seed=0, config=PluginConfig(seed=0))
    for i, name in enumerate(["glob", "read", "grep", "write_file",
                              "calculator", "fetch", "edit", "bash"]):
        body.register_tool(name, lambda **k: k, f"tool {i} does {name}")
    # 只用 2 个工具调用 10 次
    for i in range(10):
        body._tool_use_counter["glob"] = body._tool_use_counter.get("glob", 0) + 1
    body._tool_use_counter["read"] = 3
    m = body._cognition_metrics()
    print(f"  工具熵: {m['tool_entropy']} | 用 {m['tools_used']}/{m['n_tools']}")
    print(f"  警示: {m['warnings']}")
    ok = any("价值萎缩" in w for w in m["warnings"])
    print(f"  {'✅ 价值萎缩警示触发 (病理被监测)' if ok else '❌'}")
    return ok


if __name__ == "__main__":
    results = [test_healthy_flow(), test_retrieval_degradation(),
               test_calib_distortion(), test_value_atrophy()]
    print("\n" + "=" * 62)
    print(f"总体: {'✅ 认知再生产指标全部验证通过' if all(results) else '❌ 有失败'}")
    print("=" * 62)
    sys.exit(0 if all(results) else 1)
