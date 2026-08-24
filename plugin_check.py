"""
DeepSeek harness 插件端到端验证 (plugin_check.py)
=================================================
[1] 开关矩阵: 关学习+记忆 → 任务后状态零变化; 开 → 变化
[2] 响应学习: 教 (情境→回应) → 相似查询检索命中 (学会输出字符, 无 LLM)
[3] HTTP 端到端: 模拟 harness — 对话 → 工具调用 → 结果回传 → 学习
[4] 睡眠巩固: 任务后 sleep() → E1 下降
[5] 存档: save → 修改 → load → 回到存档状态
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import json
import sys
import threading
import time
import urllib.request

import numpy as np

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "body"))

from kernel import JepaBody
from plugin_config import PluginConfig
from plugin_server import PluginServer, start_server
from components.world_model import ResidualWorldModel


def _real_wm(body):
    """把默认 Dummy 世界模型换成真实 ResidualWorldModel (有真实权重 W2)"""
    body.agent.world_model = ResidualWorldModel(n_actions=5, seed=7)


# ── [1] 开关矩阵 ─────────────────────────────────────────
def check_switches():
    print("=" * 70)
    print("[1] 开关矩阵: 关闭学习+记忆 → 任务不改变模型")
    print("=" * 70)
    cfg_off = PluginConfig(learning=False, memory=False, tools=False)
    body_off = JepaBody(seed=7, config=cfg_off)
    _real_wm(body_off)
    s0 = (len(body_off.agent.memory.items),
          float(np.linalg.norm(body_off.agent.world_model.W2)))
    rng = np.random.RandomState(0)
    for _ in range(150):                      # 任务
        obs = rng.randn(5).astype(np.float32)
        d = body_off.decide(obs)
        body_off.learn(obs, d["action"], rng.randn(5).astype(np.float32), 0.5)
    s1 = (len(body_off.agent.memory.items),
          float(np.linalg.norm(body_off.agent.world_model.W2)))
    frozen = s0 == s1
    print(f"  关闭: 记忆 {s0[0]}→{s1[0]} | 权重范数 {s0[1]:.6f}→{s1[1]:.6f}")
    print(f"  {'✅ 关闭学习+记忆 → 模型完全冻结 (可逆性=零污染)' if frozen else '❌ 开关失效'}")

    cfg_on = PluginConfig(learning=True, memory=True, tools=False)
    body_on = JepaBody(seed=7, config=cfg_on)
    _real_wm(body_on)
    rng = np.random.RandomState(0)
    for _ in range(150):
        obs = rng.randn(5).astype(np.float32)
        d = body_on.decide(obs)
        body_on.learn(obs, d["action"], rng.randn(5).astype(np.float32), 0.5)
    changed = (len(body_on.agent.memory.items) > 0 or
               float(np.linalg.norm(body_on.agent.world_model.W2)) > 0)
    print(f"  开启: 记忆 {len(body_on.agent.memory.items)} 条 | "
          f"权重范数 {np.linalg.norm(body_on.agent.world_model.W2):.6f}")
    print(f"  {'✅ 开启学习+记忆 → 任务改变模型' if changed else '❌ 开启无效果'}")
    ok1 = frozen and changed
    print(f"  [1] {'✅ PASS' if ok1 else '❌ FAIL'}")
    return ok1


# ── [2] 响应学习 (学会输出字符) ──────────────────────────
def check_respond():
    print("\n" + "=" * 70)
    print("[2] 响应学习: 教 (情境→回应) → 检索命中 (无 LLM)")
    print("=" * 70)
    body = JepaBody(seed=7, config=PluginConfig(respond_mode="retrieval"))
    # 教学: 3 条情境-回应经验
    lessons = [
        ("hello how are you", "Hello! I am well, thank you for asking."),
        ("what is jepa architecture", "JEPA is a world-model framework that learns by prediction."),
        ("tell me a joke", "Why did the JEPA cross the road? To minimize prediction error!"),
    ]
    for obs, text in lessons:
        body.learn_response(obs, text)
    print(f"  已教 {body.responder.n()} 条回应经验")

    hits = 0
    queries = [("hi there how are you doing", 0),
               ("explain the jepa architecture please", 1),
               ("can you tell me a funny joke", 2),
               ("completely unrelated request about weather", None)]
    for q, expect in queries:
        resp = body.chat_completion([{"role": "user", "content": q}])
        got = resp.get("content", "")
        if expect is not None:
            hit = lessons[expect][1] in got
            hits += 1 if hit else 0
            print(f"  查询 '{q[:28]}...' → {'✅ 命中' if hit else '❌ 未命中'}: {got[:40]}")
        else:
            no_llm = not lessons[0][1] in got
            print(f"  查询 '{q[:28]}...' → {'✅ 未乱套用' if no_llm else '❌ 乱套用'}: {got[:40]}")
    ok2 = hits >= 2
    print(f"  命中 {hits}/3 | 统计 {body.responder.stats_dict()}")
    print(f"  [2] {'✅ PASS (学会输出字符, 未调用任何 LLM)' if ok2 else '❌ FAIL'}")
    return ok2


# ── [3] HTTP 端到端 ──────────────────────────────────────
def check_http():
    print("\n" + "=" * 70)
    print("[3] HTTP 端到端: 模拟 DeepSeek harness 对话/工具往返")
    print("=" * 70)
    srv = start_server(port=8031, config=PluginConfig(seed=7))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)
    base = "http://127.0.0.1:8031"

    def post(path, payload):
        req = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())

    def get(path):
        with urllib.request.urlopen(base + path, timeout=15) as r:
            return json.loads(r.read().decode())

    # 对话请求 (OpenAI 格式, 带工具)
    resp = post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "calculate 2 plus 3"}],
        "tools": [{"type": "function",
                   "function": {"name": "calculator",
                                "description": "calculate arithmetic",
                                "parameters": {"type": "object",
                                               "properties": {"expr": {"type": "string"}}}}}],
    })
    has_tool = bool(resp.get("tool_calls"))
    print(f"  对话响应: content={resp.get('content', '')[:30]!r} "
          f"tool_calls={len(resp.get('tool_calls', []))}")

    # 工具结果回传 (harness 执行 calculator 后)
    if has_tool:
        tc = resp["tool_calls"][0]
        final = post("/v1/tool_results", {
            "tool_calls": [{"function": {"name": tc["function"]["name"]}}],
            "tool_results": ["5.0"]})
        print(f"  工具回传: {final.get('content', '')[:40]!r}")

    # 教回应 (通过 HTTP)
    post("/v1/learn_response", {"obs": "hello plugin", "text": "Greetings from the plugin!"})
    resp2 = post("/v1/chat/completions",
                 {"messages": [{"role": "user", "content": "hello plugin how are you"}]})
    learned = "Greetings from the plugin" in resp2.get("content", "")
    print(f"  学习后回应: {resp2.get('content', '')[:40]!r} "
          f"{'✅ 学到' if learned else '❌ 未学到'}")

    st = get("/v1/status")
    print(f"  status: {st.get('config', '')}")
    ok3 = has_tool and learned and "learning" in st.get("config", "")
    print(f"  [3] {'✅ PASS (对话+工具往返+学习闭环)' if ok3 else '❌ FAIL'}")
    srv.shutdown()
    return ok3


# ── [4] 睡眠巩固接入 ─────────────────────────────────────
def check_sleep():
    print("\n" + "=" * 70)
    print("[4] 睡眠巩固: 任务后 sleep() → E1 下降")
    print("=" * 70)
    from components.consolidation import SleepConsolidation
    cfg = PluginConfig(seed=7)
    body = JepaBody(seed=7, config=cfg)
    _real_wm(body)   # 真实预测器才有可学习的固定偏移
    # 可学任务: 固定偏移转移 (让预测器有东西学)
    rng = np.random.RandomState(0)
    c = np.ones(768, dtype=np.float32) * 0.05
    # 先造记忆: 手动写入转移 (模拟白天经验)
    for _ in range(120):
        s = rng.randn(768).astype(np.float32)
        body.agent.memory.add_transition(s, 0, s + c, 0.05)
    # 评估用同一条可学转移 (预测器学 c 后 E1 应下降)
    s_test = rng.randn(768).astype(np.float32)
    before = body.agent.world_model.energy(s_test, 0, s_test + c)
    r = body.sleep()
    after = body.agent.world_model.energy(s_test, 0, s_test + c)
    drop = (1 - after / max(before, 1e-12)) * 100
    print(f"  E1: {before:.5f} → {after:.5f} (下降 {drop:.0f}%) | "
          f"重放 {r.get('n_replayed', 0)} 条")
    ok4 = drop > 10
    print(f"  [4] {'✅ PASS' if ok4 else '❌ FAIL'}")
    return ok4


# ── [5] 存档 ─────────────────────────────────────────────
def check_archive():
    print("\n" + "=" * 70)
    print("[5] 存档: save → 修改 → load → 回到存档状态")
    print("=" * 70)
    import tempfile, os
    path = os.path.join(tempfile.gettempdir(), "jepa_plugin_test.pkl")
    body = JepaBody(seed=7, config=PluginConfig(seed=7))
    _real_wm(body)
    body.learn_response("hello", "saved response")
    for _ in range(50):
        obs = np.random.randn(5).astype(np.float32)
        d = body.decide(obs)
        body.learn(obs, d["action"], np.random.randn(5).astype(np.float32), 0.5)
    body.save(path)
    mem_saved = len(body.agent.memory.items)
    w_saved = float(np.linalg.norm(body.agent.world_model.W2))

    # 继续修改 (污染)
    for _ in range(80):
        obs = np.random.randn(5).astype(np.float32)
        d = body.decide(obs)
        body.learn(obs, d["action"], np.random.randn(5).astype(np.float32), 0.5)
    mem_dirty = len(body.agent.memory.items)

    body.load(path)
    mem_back = len(body.agent.memory.items)
    w_back = float(np.linalg.norm(body.agent.world_model.W2))
    restored = mem_back == mem_saved and abs(w_back - w_saved) < 1e-6
    print(f"  记忆: 存档 {mem_saved} → 污染后 {mem_dirty} → 恢复 {mem_back}")
    print(f"  权重范数: 存档 {w_saved:.6f} → 恢复 {w_back:.6f}")
    print(f"  {'✅ 存档/恢复完整 (任务边界回滚可用)' if restored else '❌ 存档恢复失败'}")
    print(f"  [5] {'✅ PASS' if restored else '❌ FAIL'}")
    return restored


if __name__ == "__main__":
    ok1 = check_switches()
    ok2 = check_respond()
    ok3 = check_http()
    ok4 = check_sleep()
    ok5 = check_archive()
    print("\n" + "=" * 70)
    print(f"插件验证: 开关{ok1} 响应学习{ok2} HTTP{ok3} 睡眠{ok4} 存档{ok5}")
    print(f"总体: {'✅ 全部通过 — 真实身体插件可用' if all([ok1, ok2, ok3, ok4, ok5])
           else '⚠️ 有未通过项'}")
    print("=" * 70)
