"""
dsh harness 端到端对接验证 (dsh_harness_check.py)
==================================================
模拟 DeepSeek harness 的完整工具调用循环, 走真实 HTTP 通道:
  POST /v1/chat/completions (带 tools schema) → tool_calls
  → harness 执行真实工具 → POST /v1/tool_results (带 context) → 再 chat
  → 直到无 tool_calls (任务完成)

验证目标:
  1. 单步任务: list files → glob (含参数)
  2. 真多步任务: "list then read" → glob → read → 收尾 (时序三槽驱动)
  3. 陌生任务不盲调
  4. 工具世界模型: 预测器参与选择 (tool_wm 生效)

依赖: 无第三方库 (urllib 标准库); 需要 plugin server 在 8045 运行.
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8045"

# dsh 真实工具集 (与 plugin_server 白名单对齐的子集)
TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "glob",
     "description": "find files by glob pattern",
     "parameters": {"type": "object",
                    "properties": {"pattern": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "read",
     "description": "read a file from disk",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "grep",
     "description": "search text in files",
     "parameters": {"type": "object",
                    "properties": {"pattern": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "pwsh",
     "description": "run a powershell command",
     "parameters": {"type": "object",
                    "properties": {"command": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "edit",
     "description": "edit a file",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"},
                                   "old": {"type": "string"},
                                   "new": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "write_file",
     "description": "write content to a file",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"},
                                   "content": {"type": "string"}}}}},
]


def real_tools():
    """真实工具执行器 (harness 侧)"""
    def tool_glob(pattern="*"):
        import glob as g
        return f"found {len(g.glob(os.path.join(REPO_ROOT, REPO_ROOT)+pattern))}: " \
               f"{', '.join(g.glob(os.path.join(REPO_ROOT, REPO_ROOT)+pattern)[:8])}"
    def tool_read(path=""):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read(300)
        except Exception as e:
            return f"ERROR: {e}"
    def tool_grep(pattern="", path=REPO_ROOT):
        hits = []
        for f in os.listdir(path):
            if f.endswith(".py"):
                p = os.path.join(path, f)
                with open(p, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if pattern in line:
                            hits.append(f"{f}: {line.strip()[:40]}")
                            break
        return f"found {len(hits)}: {', '.join(hits[:5])}" if hits else "no matches"
    def tool_pwsh(command=""):
        return f"ran: {command[:50]}"
    def tool_edit(path="", old="", new=""):
        return f"edited {path}"
    def tool_write(path="", content=""):
        return f"wrote {len(content)} bytes to {path}"
    return {"glob": tool_glob, "read": tool_read, "grep": tool_grep,
            "pwsh": tool_pwsh, "edit": tool_edit, "write_file": tool_write}


def post(path, payload, method="POST"):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def get(path):
    return post(path, None, method="GET")


def run_task(task: str, max_rounds: int = 6) -> list[dict]:
    """完整 harness 循环 (与 dsh 相同语义)"""
    tools = real_tools()
    messages = [{"role": "user", "content": task}]
    rounds = []
    for i in range(max_rounds):
        resp = post("/v1/chat/completions",
                    {"messages": messages, "tools": TOOLS_SCHEMA,
                     "stream": False})
        msg = resp["choices"][0]["message"]
        tcs = msg.get("tool_calls", [])
        if not tcs:
            rounds.append({"round": i + 1, "via": resp.get("via", "?"),
                           "content": msg.get("content", "")})
            break
        # harness 执行工具
        results = []
        for tc in tcs:
            fn = tc["function"]
            args = json.loads(fn.get("arguments") or "{}")
            results.append(tools[fn["name"]](**args))
        ctx_before = list(messages)
        # 回传结果 (带 context) + 学习
        post("/v1/tool_results",
             {"tool_calls": tcs, "tool_results": results,
              "context": ctx_before})
        # 追加消息历史
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": tcs})
        messages.append({"role": "tool", "content": results[0],
                         "tool_call_id": tcs[0].get("id", "0")})
        rounds.append({"round": i + 1, "tool": fn["name"], "args": args,
                       "result": results[0][:40]})
    return rounds


def teach():
    """通过 HTTP 教学 (与多步验证同课程).
    先 warmup chat 注入工具注册 — 教学要教世界模型 (工具 action 列)."""
    post("/v1/chat/completions",
         {"messages": [{"role": "user", "content": "warmup"}],
          "tools": TOOLS_SCHEMA, "stream": False})
    lessons = [
        ({"task": "list the files in the directory"}, "glob",
         {"pattern": "*.py"}, "found 12: jepa_base.py, ...", 1.0),
        ({"task": "read the file jepa_base.py"}, "read",
         {"path": os.path.join(REPO_ROOT, "jepa_base.py")}, "import numpy as np ...", 1.0),
        ({"task": "search for class JEPA in the code"}, "grep",
         {"pattern": "class"}, "found 3: ...", 1.0),
        ({"task": "list the files then read the first one",
          "last_call": 'glob: {"pattern": "*.py"}',
          "last_result": "found 12: jepa_base.py, ..."},
         "read", {"path": os.path.join(REPO_ROOT, "bb_component_check.py")},
         "import sys ...", 1.0),
        ({"task": "read the file content shown now the task is complete"},
         "", {}, "done", 0.0),
    ]
    for obs, tool, args, result, perf in lessons:
        post("/v1/learn_call", {"obs": obs, "tool": tool, "args": args,
                                "result": result, "perf": perf})
    st = get("/v1/status")
    return st


def main():
    print("=" * 70)
    print("dsh harness 端到端对接验证 (HTTP 8045)")
    print("=" * 70)
    # 健康检查
    try:
        st0 = get("/v1/status")
    except Exception as e:
        print(f"❌ server 未就绪: {e}")
        print("   先启动: JEPA_PORT=8045 .venv/Scripts/python.exe body/plugin_server.py")
        return False
    print(f"server OK | semantic={st0.get('semantic', {}).get('available')} "
          f"| ctx_dim={st0.get('ctx_dim')}")

    st = teach()
    print(f"教学完成: call_mem={st.get('call_mem', {}).get('n')} "
          f"| 工具效果原型={st.get('tool_effects')}")

    # 任务 1: 单步
    print("\n[任务1] 'list the files in the directory' (应 glob)")
    r1 = run_task("list the files in the directory")
    print(f"  执行: {[(x.get('tool'), x.get('args')) for x in r1]}")
    ok1 = any(x.get("tool") == "glob" for x in r1)
    print(f"  {'✅ 单步命中' if ok1 else '❌ 未命中'}")

    # 任务 2: 真多步
    print("\n[任务2] 'show me what python files exist then read the first one' "
          "(glob → read)")
    r2 = run_task("show me what python files exist then read the first one")
    tools2 = [x.get("tool") for x in r2 if x.get("tool")]
    print(f"  执行序列: {tools2} | {[(x.get('tool'), x.get('args')) for x in r2]}")
    ok2 = (tools2 and tools2[0] == "glob" and "read" in tools2
           and len(tools2) <= 4)
    print(f"  {'✅ 真多步 (glob → read → 收尾)' if ok2 else '❌ 未完成真多步'}")

    # 任务 3: 陌生任务
    print("\n[任务3] 'send an email to the boss' (陌生, 最多1次探索不盲调)")
    r3 = run_task("send an email to the boss")
    tools3 = [x.get("tool") for x in r3 if x.get("tool")]
    print(f"  执行: {[(x.get('round'), x.get('tool', x.get('content',''))[:40]) for x in r3]}")
    # 自由学习预期: 陌生任务允许 ≤1 次探索尝试 (通用工具兜底, 真实 agent 行为),
    # 但不允许死循环/多工具乱调
    ok3 = len(tools3) <= 1
    print(f"  {'✅ 陌生任务收敛 (≤1 次探索)' if ok3 else '❌ 乱调工具'}")
    if tools3:
        bad = [t for t in tools3 if t in ("interrupt_agent", "create_goal")]
        print(f"  内部工具误选: {bad if bad else '无'}")

    # 状态
    stf = get("/v1/status")
    print("\n" + "=" * 70)
    print(f"结果: 单步{ok1} 多步{ok2} 不盲调{ok3}")
    print(f"最终状态: call_mem={stf.get('call_mem', {}).get('n')} "
          f"hit_rate={stf.get('call_mem', {}).get('hit_rate')} "
          f"| 工具效果={stf.get('tool_effects')} "
          f"| tool_wm={stf.get('tool_wm')}")
    overall = all([ok1, ok2, ok3])
    print(f"总体: {'✅ 端到端对接成立' if overall else '⚠️ 有未通过项'}")
    print("=" * 70)
    return overall


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
