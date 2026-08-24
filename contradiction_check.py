"""矛盾处理协议验证 (contradiction_check.py)
四场景: ①互斥裁决(验证历史加权) ②互补并存(条件差异) ③待求证升级(弃权触发查证)
④冗余同义(不构成矛盾). 合成向量 + 文本, 快速."""
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


def high_sim(seed_base, seed_noise):
    """与 base 高相似 (cos≈0.8) 的变体向量."""
    base = mkz(seed_base)
    v = base * 0.8 + mkz(seed_noise) * 0.6
    return v / (np.linalg.norm(v) + 1e-9)


print("=" * 62)
print("矛盾处理协议验证 (认知升级: 矛盾=高级受阻)")
print("=" * 62)
ok_all = True

# ── 场景1: 互斥裁决 (验证历史加权, 不偏新) ──
print("\n[场景1] 互斥矛盾 → 实践裁决 (输方弃权)")
r1 = RespondLearner(min_sim=0.35, cap=30, seed=0)
z1a = mkz(100)
z1b = high_sim(100, 200)
r1.learn(z1a, "the weather in tokyo is rainy")
r1.learn(z1b, "the weather in tokyo is sunny")
# A 判对 3 次, B 判错 2 次 → A 胜
for _ in range(3):
    t = r1.respond(z1a); r1.report_outcome(True)
for _ in range(2):
    t = r1.respond(z1b); r1.report_outcome(False)
c = r1.contradictions[0]
print(f"  矛盾: {c['type']} | 验证 A={c['verified_a']} B={c['verified_b']} "
      f"| status={c['status']} winner={c['winner']}")
oa = r1.respond(z1a)   # 赢方 → 正常
ob = r1.respond(z1b)   # 输方 → 弃权
# 矛盾对 a=后学(sunny), b=先学(rainy); rainy 3 对 → winner=1 (b 胜)
ok1 = c["status"] == "resolved" and c["winner"] == 1 \
      and oa == "the weather in tokyo is rainy" and ob is None
print(f"  赢方回答: {oa!r} {'✅' if oa else '❌'} | "
      f"输方弃权: {ob} {'✅' if ob is None else '❌'}")
ok_all &= ok1

# ── 场景2: 互补并存 (条件差异: 冬季/夏季 都对) ──
print("\n[场景2] 互补矛盾 → 并存合法 (不弃权)")
r2 = RespondLearner(min_sim=0.35, cap=30, seed=1)
z2a = mkz(300)
z2b = high_sim(300, 400)
r2.learn(z2a, "the weather in tokyo in winter is cold")
r2.learn(z2b, "the weather in tokyo in summer is hot")
# 双方各判对 2 次 → complementary (条件差异)
for _ in range(2):
    t = r2.respond(z2a); r2.report_outcome(True)
for _ in range(2):
    t = r2.respond(z2b); r2.report_outcome(True)
c2 = r2.contradictions[0]
oa = r2.respond(z2a)
ob = r2.respond(z2b)
ok2 = c2["type"] == "complementary" and c2["status"] == "resolved" \
      and oa is not None and ob is not None
print(f"  矛盾: {c2['type']} | 验证 A={c2['verified_a']} B={c2['verified_b']} "
      f"| status={c2['status']}")
print(f"  冬季回答: {oa!r} {'✅' if oa else '❌'} | "
      f"夏季回答: {ob!r} {'✅' if ob else '❌'} (并存合法)")
ok_all &= ok2

# ── 场景3a: 宽容等待 (单方验证 → pending 不弃权, 不冤枉未验证方) ──
print("\n[场景3a] 宽容等待 (单方验证 → pending, 未验证方可正常用)")
r3 = RespondLearner(min_sim=0.35, cap=30, seed=2)
z3a = mkz(500)
z3b = high_sim(500, 600)
r3.learn(z3a, "the capital of country X is cityA")
r3.learn(z3b, "the capital of country X is cityB")
for _ in range(2):
    r3.respond(z3a); r3.report_outcome(True)   # cityA 对 2 次
c3 = r3.contradictions[0]
ob = r3.respond(z3b)   # 未验证方 → 不弃权 (宽容)
ok3a = c3["status"] == "pending" and ob is not None \
       and r3.stats.get("contradiction_abstain", 0) == 0
print(f"  矛盾: {c3['type']} | 验证 A={c3['verified_a']} B={c3['verified_b']} "
      f"| status={c3['status']}")
print(f"  未验证方回答: {ob!r} {'✅ 宽容等待不弃权' if ok3a else '❌ 被冤枉了'}")
ok_all &= ok3a

# ── 场景3b: 错误判负 (一方错≥2 → 判负 → 输方弃权触发查证) ──
print("\n[场景3b] 错误判负 (实践否决 → 输方弃权 → 查证)")
r3b = RespondLearner(min_sim=0.35, cap=30, seed=3)
z4a = mkz(700)
z4b = high_sim(700, 800)
r3b.learn(z4a, "the capital of country Y is cityP")
r3b.learn(z4b, "the capital of country Y is cityQ")
for _ in range(2):
    r3b.respond(z4b); r3b.report_outcome(True)    # cityQ 对 2 次
for _ in range(2):
    r3b.respond(z4a); r3b.report_outcome(False)   # cityP 错 2 次 → 判负
c3b = r3b.contradictions[0]
op = r3b.respond(z4a)   # 输方 → 弃权
# 矛盾对 a=后学(cityQ), b=先学(cityP); cityQ 对2 → va=2, cityP 错2 → wb=2 → winner=0 (a 胜)
ok3b = c3b["status"] == "resolved" and c3b["winner"] == 0 \
       and op is None and r3b.last_block_reason == "contradiction" \
       and r3b.stats.get("contradiction_abstain", 0) >= 1
print(f"  矛盾: {c3b['type']} | 验证 A={c3b['verified_a']} B={c3b['verified_b']} "
      f"| status={c3b['status']} winner={c3b['winner']}")
print(f"  输方弃权: {op} + last_block={r3b.last_block_reason} "
      f"({'✅ 矛盾受阻触发查证' if ok3b else '❌'})")
ok_all &= ok3b

# ── 场景4: 冗余同义 → 不构成矛盾 ──
print("\n[场景4] 冗余同义 (答案几乎相同 → 不构成矛盾)")
r4 = RespondLearner(min_sim=0.35, cap=30, seed=3)
z4a = mkz(700)
z4b = high_sim(700, 800)
r4.learn(z4a, "tokyo weather is rainy today")
r4.learn(z4b, "tokyo weather is rainy")   # 同义 (子集)
ok4 = len(r4.contradictions) == 0
print(f"  矛盾记录数: {len(r4.contradictions)} (应 0=同义不矛盾) "
      f"{'✅' if ok4 else '❌'}")
ok_all &= ok4

print("\n" + "=" * 62)
print(f"总体: {'✅ 矛盾处理四场景全部通过' if ok_all else '❌ 有失败需检查'}")
print("=" * 62)
sys.exit(0 if ok_all else 1)
