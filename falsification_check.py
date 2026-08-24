"""
范式三元组可证伪性验证 (falsification_check.py) — 落点3
========================================================
验证: ①verified 调用被复用 ②falsified 调用被排除 (反例驱动积累, 不盲调)
     ③kernel 集成: tool_result_step 外部判定 perf=0 → 该模式证伪 → 不复用
     ④状态统计 (verified/falsified/failed)
对应命题5: 积累对象 = 可证伪因果范式, "检索命中的回答要求与结果一致才可复用".
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


def test_verified_reuse():
    print("=" * 62)
    print("[场景1] verified 调用复用 (健康)")
    print("=" * 62)
    from call_memory import CallMemory
    cm = CallMemory(min_sim=0.5, cap=20, seed=0)
    z = mkz(1)
    idx = cm.learn(z, "glob", {"pattern": "*.py"}, "found 12 files", 1.0)
    cm.verify(idx, True)                      # 判定正确 → verified
    hit = cm.select(z)
    st = cm.stats_dict()
    print(f"  状态: {cm.meta[idx]['status']} | 复用: {hit} "
          f"| 统计: verified={st['verified']} falsified={st['falsified']}")
    ok = hit == ("glob", {"pattern": "*.py"}) and st["verified"] == 1
    print(f"  {'✅ verified 调用可复用' if ok else '❌'}")
    return ok


def test_falsified_excluded():
    print("\n" + "=" * 62)
    print("[场景2] falsified 调用被排除 (反例驱动, 不盲调)")
    print("=" * 62)
    from call_memory import CallMemory
    cm = CallMemory(min_sim=0.5, cap=20, seed=0)
    z = mkz(2)
    idx = cm.learn(z, "read", {"path": "wrong.txt"}, "ERROR: not found", 0.0)
    cm.verify(idx, False)                     # 判定错误 → falsified
    hit = cm.select(z)
    st = cm.stats_dict()
    print(f"  状态: {cm.meta[idx]['status']} | 复用: {hit} "
          f"| 统计: verified={st['verified']} falsified={st['falsified']}")
    ok = hit is None and st["falsified"] == 1
    print(f"  {'✅ falsified 调用不复用 (盲调被阻断)' if ok else '❌'}")
    return ok


def test_kernel_integration():
    print("\n" + "=" * 62)
    print("[场景3] kernel 集成: tool_result_step 判定证伪 → 不复用")
    print("=" * 62)
    from kernel import JepaBody
    from plugin_config import PluginConfig
    body = JepaBody(seed=3, config=PluginConfig(seed=3, respond_cap=500,
                                                respond_min_sim=0.18))
    body.ensure_semantic()
    # 教学一个调用 (成功, verified)
    body.learn_call({"task": "list python files"}, "glob",
                    {"pattern": "*.py"}, "found 12", 1.0)
    # 教学一个调用 (判定失败, falsified — 反例)
    body.learn_call({"task": "read the config file"}, "read",
                    {"path": "/etc/x"}, "ERROR: no config", 0.0)
    st1 = body.call_mem.stats_dict()
    print(f"  教学后: verified={st1['verified']} falsified={st1['falsified']}")
    # 检索: glob 应命中 (verified), read 应被排除 (falsified)
    r1 = body.chat_completion([{"role": "user",
                                "content": "list python files"}],
                              TOOLS)
    r2 = body.chat_completion([{"role": "user",
                                "content": "read the config file"}],
                              TOOLS)
    t1 = (r1.get("tool_calls") or [{}])[0].get("function", {}).get("name")
    v1 = r1.get("via", "?")
    t2 = (r2.get("tool_calls") or [{}])[0].get("function", {}).get("name")
    v2 = r2.get("via", "?")
    print(f"  list 任务 → {t1} (via={v1}) — 应检索复用 (verified)")
    print(f"  read 任务 → {t2 or '(无工具)'} (via={v2}) "
          f"— 不应是 call_memory 复用 (falsified 被证伪)")
    ok = (t1 == "glob" and v1 == "call_memory+planned"
          and v2 != "call_memory+planned")
    mark = "✅ kernel 集成: verified 检索复用 / falsified 不被检索复用 (探索=新试错)" \
        if ok else "❌"
    print(f"  {mark}")
    return ok


TOOLS = [{"type": "function",
          "function": {"name": "glob",
                       "description": "find files by glob pattern",
                       "parameters": {"type": "object",
                                      "properties": {"pattern": {"type": "string"}}}}},
         {"type": "function",
          "function": {"name": "read",
                       "description": "read a file from disk",
                       "parameters": {"type": "object",
                                      "properties": {"path": {"type": "string"}}}}},
         {"type": "function",
          "function": {"name": "grep",
                       "description": "search text in files",
                       "parameters": {"type": "object",
                                      "properties": {"pattern": {"type": "string"}}}}}]


def test_status_stats():
    print("\n" + "=" * 62)
    print("[场景4] 状态统计 (可证伪性汇总)")
    print("=" * 62)
    from call_memory import CallMemory
    cm = CallMemory(min_sim=0.5, cap=20, seed=0)
    for i in range(4):
        idx = cm.learn(mkz(10 + i), f"tool{i}", {}, f"result{i}", 1.0)
        cm.verify(idx, i % 2 == 0)            # 交替 对/错
    cm.learn(mkz(20), "err", {}, "ERROR: boom", 0.0)   # failed
    st = cm.stats_dict()
    print(f"  verified={st['verified']} falsified={st['falsified']} "
          f"failed={st['failed']} total={st['n']}")
    ok = st["verified"] == 2 and st["falsified"] == 2 and st["failed"] == 1
    print(f"  {'✅ 状态统计正确 (三态可观测)' if ok else '❌'}")
    return ok


if __name__ == "__main__":
    results = [test_verified_reuse(), test_falsified_excluded(),
               test_kernel_integration(), test_status_stats()]
    print("\n" + "=" * 62)
    print(f"总体: {'✅ 范式三元组可证伪性全部验证通过' if all(results) else '❌ 有失败'}")
    print("=" * 62)
    sys.exit(0 if all(results) else 1)
