"""
全量工具集规模验证 (tools_scale_check.py)
==========================================
哈希袋时代的痛点: 全量 dsh 工具集 (25+) 上语义太弱 → 乱选
(interrupt_agent/create_goal 等内部工具事故).

本次验证: 注入 plugin_server 白名单全量工具 schema (30 个),
JEPA 在语义三槽情境 + 预测验证下:
  1. list 任务仍选 glob (不被 30 工具稀释)
  2. 多步任务仍 glob → read
  3. 陌生任务不盲调 (不选 interrupt_agent 等内部工具)
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import json
import urllib.request

BASE = "http://127.0.0.1:8045"

FULL_TOOLS = [
    ("glob", "find files by glob pattern", {"pattern": {"type": "string"}}),
    ("read", "read a file from disk", {"path": {"type": "string"}}),
    ("grep", "search text in files", {"pattern": {"type": "string"}}),
    ("pwsh", "run a powershell command", {"command": {"type": "string"}}),
    ("bash", "run a bash command", {"command": {"type": "string"}}),
    ("shell", "run a shell command", {"command": {"type": "string"}}),
    ("execute", "execute a command", {"command": {"type": "string"}}),
    ("run_command", "run a command line", {"command": {"type": "string"}}),
    ("edit", "edit a file with old/new", {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}),
    ("write_file", "write content to a file", {"path": {"type": "string"}, "content": {"type": "string"}}),
    ("edit_file", "edit file content", {"path": {"type": "string"}, "content": {"type": "string"}}),
    ("file_read", "read file content", {"path": {"type": "string"}}),
    ("file_write", "write file content", {"path": {"type": "string"}, "content": {"type": "string"}}),
    ("str_replace_editor", "replace string in a file", {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}),
    ("list_dir", "list directory contents", {"path": {"type": "string"}}),
    ("ls", "list files", {"path": {"type": "string"}}),
    ("search", "search files", {"query": {"type": "string"}}),
    ("find", "find files by name", {"name": {"type": "string"}}),
    ("cat", "show file content", {"path": {"type": "string"}}),
    ("web", "fetch a web page", {"url": {"type": "string"}}),
    ("fetch", "fetch url content", {"url": {"type": "string"}}),
    ("http", "make an http request", {"url": {"type": "string"}}),
    ("calculator", "calculate math", {"expr": {"type": "string"}}),
    ("python", "run python code", {"code": {"type": "string"}}),
    ("run_python", "run python script", {"code": {"type": "string"}}),
    ("task_done", "mark task as done", {"summary": {"type": "string"}}),
    ("finish", "finish the task", {"summary": {"type": "string"}}),
    ("complete", "complete the task", {"summary": {"type": "string"}}),
    ("interrupt_agent", "interrupt the agent", {}),
    ("create_goal", "create an internal goal", {"goal": {"type": "string"}}),
]

SCHEMA = [{"type": "function", "function": {"name": n, "description": d,
            "parameters": {"type": "object", "properties": p}}}
          for n, d, p in FULL_TOOLS]


def post(path, payload, method="POST"):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def run_task(task, max_rounds=6):
    messages = [{"role": "user", "content": task}]
    rounds = []
    for _ in range(max_rounds):
        resp = post("/v1/chat/completions",
                    {"messages": messages, "tools": SCHEMA, "stream": False})
        msg = resp["choices"][0]["message"]
        tcs = msg.get("tool_calls", [])
        if not tcs:
            rounds.append({"tool": None, "content": msg.get("content", "")[:40]})
            break
        results = []
        for tc in tcs:
            fn = tc["function"]
            name = fn["name"]
            # 占位执行 (只 glob/read 有真实现, 其余返回占位)
            if name == "glob":
                import os
                results.append(f"found {len([f for f in os.listdir(REPO_ROOT) if f.endswith('.py')])} py files")
            elif name == "read":
                results.append("file content: import numpy as np ...")
            else:
                results.append(f"{name} executed ok")
        post("/v1/tool_results",
             {"tool_calls": tcs, "tool_results": results, "context": list(messages)})
        messages.append({"role": "assistant", "content": "", "tool_calls": tcs})
        messages.append({"role": "tool", "content": results[0], "tool_call_id": "0"})
        rounds.append({"tool": tcs[0]["function"]["name"],
                       "args": json.loads(tcs[0]["function"].get("arguments") or "{}")})
    return rounds


def teach():
    post("/v1/chat/completions",
         {"messages": [{"role": "user", "content": "warmup"}],
          "tools": SCHEMA, "stream": False})
    lessons = [
        ({"task": "list the files in the directory"}, "glob",
         {"pattern": "*.py"}, "found 12: jepa_base.py, ...", 1.0),
        ({"task": "read the file jepa_base.py"}, "read",
         {"path": os.path.join(REPO_ROOT, "jepa_base.py")}, "import numpy as np ...", 1.0),
        ({"task": "list the files then read the first one",
          "last_call": 'glob: {"pattern": "*.py"}',
          "last_result": "found 12: jepa_base.py, ..."},
         "read", {"path": os.path.join(REPO_ROOT, "bb_component_check.py")}, "import sys ...", 1.0),
    ]
    for obs, tool, args, result, perf in lessons:
        post("/v1/learn_call", {"obs": obs, "tool": tool, "args": args,
                                "result": result, "perf": perf})


def main():
    print("=" * 70)
    print(f"全量工具集规模验证 ({len(FULL_TOOLS)} 工具)")
    print("=" * 70)
    teach()
    print("教学完成 (全量工具集下)")

    r1 = run_task("list the files in the directory")
    t1 = [x["tool"] for x in r1 if x["tool"]]
    print(f"[任务1] 执行: {t1}  {'✅ 仍选 glob' if t1 and t1[0]=='glob' else '❌ 乱选'}")

    r2 = run_task("show me what python files exist then read the first one")
    t2 = [x["tool"] for x in r2 if x["tool"]]
    ok2 = t2 and t2[0] == "glob" and "read" in t2 and len(t2) <= 4
    print(f"[任务2] 执行: {t2}  {'✅ 多步仍成立' if ok2 else '❌ 多步失效'}")

    r4 = run_task("send an email to the boss")
    t4 = [x["tool"] for x in r4 if x["tool"]]
    print(f"[任务3] 陌生执行: {t4 or '(无工具, 诚实收尾)'}  "
          f"{'✅ 不盲调' if not t4 else '❌ 乱调(含内部工具?)'}")
    if t4:
        bad = [t for t in t4 if t in ("interrupt_agent", "create_goal")]
        print(f"  内部工具误选: {bad if bad else '无'}")

    print("=" * 70)
    return (t1 and t1[0] == "glob") and ok2 and not t4


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
