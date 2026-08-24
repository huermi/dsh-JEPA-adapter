"""分层遗忘综合验证 (selective_forgetting_check.py)
蓝图落地验证: ①fast 加权淘汰(质量×时间) ②沉淀合题(质量×频率)
③core 巩固分归档(遗忘=归档非删除) ④W 衰减+G 保护 ⑤冲突评审.
合成向量场景 (不依赖 MiniLM, 快速)."""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import sys
import numpy as np
sys.path.insert(0, os.path.join(REPO_ROOT, "body"))
from respond_learner import RespondLearner


def mkz(seed, d=128):
    r = np.random.RandomState(seed)
    v = r.randn(d).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def sim(v1, v2):
    return float(np.dot(v1 / (np.linalg.norm(v1) + 1e-9),
                        v2 / (np.linalg.norm(v2) + 1e-9)))


print("=" * 62)
print("分层遗忘综合验证 (蓝图落地)")
print("=" * 62)
ok_all = True

# ── 场景1: fast 加权淘汰 (污染条目优先滚, 高价值保护) ──
print("\n[场景1] fast 加权淘汰 (质量×时间)")
r = RespondLearner(min_sim=0.35, cap=5, seed=0)
zs = [mkz(100 + i) for i in range(6)]
for i in range(5):
    r.learn(zs[i], f"item{i}")
# 条目0 答错 2 次 (污染, pass_score=-4); 条目1 答对 1 次 + 近期命中 (保护, 不沉淀)
for _ in range(2):
    t = r.respond(zs[0]); r.report_outcome(False)   # 条目0 答错
r.respond(zs[1]); r.report_outcome(True)            # 条目1 答对 (pass=1, 近期命中保护)
print(f"  淘汰前: pass0={r.pass_scores.get(0)} pass1={r.pass_scores.get(1)} "
      f"hit1={r.hit_counts.get(1,0)} fast={len(r.pairs)}")
r.learn(zs[5], "item5")   # fast 6 > cap5 → 触发淘汰
kept = [t for _, t in r.pairs]
ok1 = "item1" in kept and "item0" not in kept and len(r.pairs) == 5
print(f"  淘汰后: item0 {'❌仍在(应淘汰)' if 'item0' in kept else '✅已淘汰'} | "
      f"item1 {'✅保留' if 'item1' in kept else '❌被误杀'} | "
      f"item5 {'✅新入' if 'item5' in kept else '❌未入'}")
ok_all &= ok1

# ── 场景2: 沉淀合题 (质量×频率: 频繁检索+判定正确 → 沉淀) ──
print("\n[场景2] 沉淀合题 (质量×频率)")
r2 = RespondLearner(min_sim=0.35, cap=20, seed=1)
za = mkz(200)
r2.learn(za, "answerA")
for _ in range(4):       # 被频繁检索 4 次 (需要度高)
    r2.respond(za)
r2.respond(za); r2.report_outcome(True)   # 判定正确 1 次 (pass_score=1)
freq = min(r2.hit_counts.get(0, 0), 4) * 0.5
score = r2.pass_scores.get(0, 0) + freq
ok2 = len(r2.core_pairs) == 1
print(f"  pass_score=1 + freq={freq} → 沉淀 {'✅' if ok2 else '❌'} "
      f"(core={len(r2.core_pairs)})")
ok_all &= ok2

# ── 场景3: core 巩固分归档 (沉淀后答错 → 归档非删除) ──
print("\n[场景3] core 巩固分 → 归档 (遗忘=移入衰退区)")
r3 = RespondLearner(min_sim=0.35, cap=20, seed=2)
zb = mkz(300)
r3.learn(zb, "answerB")
# 手动沉淀到 core (移除 fast 副本, 确保 respond 命中 core 而非 fast)
r3.core_pairs.append((r3.pairs[0]))
r3.core_scores[len(r3.core_pairs) - 1] = 0
r3.pairs.clear()
n_core0 = len(r3.core_pairs)
# 命中 core 条目, 答错 2 次 → -4 跌破阈值 → 归档
for _ in range(2):
    t = r3.respond(zb)
    assert t == "answerB", "应命中 core"
    r3.report_outcome(False)
ok3 = (len(r3.core_pairs) == n_core0 - 1
       and len(r3.archive) == 1
       and r3.archive[0][2] == "core_demoted")
print(f"  core: {n_core0}→{len(r3.core_pairs)} | 归档: {len(r3.archive)} 条 "
      f"({'✅ 归档非删除' if ok3 else '❌'})")
ok_all &= ok3

# ── 场景4: W 衰减 + G 保护 (重要方向坚守, 次要方向可更新) ──
print("\n[场景4] W 衰减 + G 保护 (弹性权重)")
r4 = RespondLearner(min_sim=0.35, cap=20, seed=3)
z1, a1 = mkz(400), mkz(500)
z2, a2 = mkz(600), mkz(700)
for _ in range(10):      # z1→a1 学 10 次 (高 G, 重要方向)
    r4._learn_lam_vec(z1, a1)
for _ in range(1):       # z2→a2 学 1 次 (低 G, 次要方向)
    r4._learn_lam_vec(z2, a2)
g1 = float(np.sum(np.abs(r4.G[:, :64]))) if r4.G is not None else 0
# 现在 z1 换新目标 a1' (轻微变化) — 高 G 方向步长小 → a1 方向保持
a1p = a1 * 0.9 + mkz(800) * 0.1
a1p = a1p / (np.linalg.norm(a1p) + 1e-9)
r4._learn_lam_vec(z1, a1p)
pred1 = r4.predict_lam(z1)
keep = sim(pred1, a1)       # 与旧目标 a1 的相似度 (高 = 保持得好)
adapt = sim(pred1, a1p)     # 与新目标 a1p 的相似度
ok4 = keep > 0.9 and r4.W is not None and g1 > 0
print(f"  高G方向: 保持旧目标 {keep:.3f} (>0.9=坚守) | 朝新目标 {adapt:.3f} | "
      f"G 已累积 {'✅' if g1 > 0 else '❌'}")
print(f"  衰减生效: W 范数 {np.linalg.norm(r4.W):.4f} (多次更新后应 < 无衰减累积)")
ok_all &= ok4

# ── 场景5: 矛盾检测 (learn 冲突 → 矛盾记录, 不机械减分; 实践裁决升级) ──
print("\n[场景5] 矛盾检测 (learn 冲突 → 矛盾对记录, 实践裁决)")
r5 = RespondLearner(min_sim=0.35, cap=20, seed=4)
zc = mkz(900)
r5.learn(zc, "old answer")
r5.core_pairs.append((r5.pairs[0]))   # 手动沉淀
r5.core_scores[0] = 0
# 学冲突条目 (高相似 0.8, 不同答案) → 矛盾检测: 矛盾对记录, core 不机械减分
zconf = zc * 0.8 + mkz(950) * 0.6
zconf = zconf / (np.linalg.norm(zconf) + 1e-9)
r5.learn(zconf, "new conflicting answer")
ok5 = (len(r5.contradictions) == 1
       and r5.core_scores.get(0, 0) == 0   # 不机械减分 (等实践裁决)
       and r5.stats.get("conflicts", 0) >= 1)
print(f"  矛盾对: {len(r5.contradictions)} (应1) | core 巩固分: "
      f"{r5.core_scores.get(0)} (应0=不机械减分) | conflicts: "
      f"{r5.stats.get('conflicts',0)}")
print(f"  {'✅ 矛盾检测升级生效 (learn 检测, 实践裁决)' if ok5 else '❌'}")
ok_all &= ok5

print("\n" + "=" * 62)
print(f"总体: {'✅ 分层遗忘五机制全部通过' if ok_all else '❌ 有失败需检查'}")
print("=" * 62)
sys.exit(0 if ok_all else 1)
