"""
连续多步任务验证 (multi_step_check.py)
======================================
验证 JEPA 学会"完整工具调用"并完成连续多步任务:
  教学: learn_call 教 3 个调用模式 (list→glob, read→read, search→grep)
  任务: "list files then read the first one" 应多步执行:
        轮1: glob(*.py) → 轮2: 基于结果 read(...) → 轮3: 收尾回应

流程模拟真实 harness 循环:
  chat(messages) → tool_calls → harness 执行 → tool_result_step(带 context)
  → 消息历史追加 → 再 chat → ... 直到无 tool_calls (任务完成)
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import json
import sys
import time

import numpy as np

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "body"))

from kernel import JepaBody
from plugin_config import PluginConfig


# ─── harness 侧真实工具 ──────────────────────────────────
import os


def tool_glob(pattern: str = "*") -> str:
    try:
        files = sorted(os.listdir(REPO_ROOT))
        if pattern == "*.py":
            files = [f for f in files if f.endswith(".py")]
        elif pattern == "*.md":
            files = [f for f in files if f.endswith(".md")]
        return f"found {len(files)}: {', '.join(files[:8])}"
    except Exception as e:
        return f"ERROR: {e}"


def tool_read(path: str = "") -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(300)
    except Exception as e:
        return f"ERROR: {e}"


def tool_grep(pattern: str = "", path: str = REPO_ROOT) -> str:
    try:
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
    except Exception as e:
        return f"ERROR: {e}"


TOOLS = {"glob": tool_glob, "read": tool_read, "grep": tool_grep}
TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "glob",
     "description": "find files by glob pattern",
     "parameters": {"type": "object",
                    "properties": {"pattern": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "read",
     "description": "read a file",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "grep",
     "description": "search text in files",
     "parameters": {"type": "object",
                    "properties": {"pattern": {"type": "string"}}}}},
]


class HarnessSim:
    """模拟 harness 的多步 agent 循环"""

    def __init__(self, body: JepaBody):
        self.body = body
        self.messages: list[dict] = []

    def run_task(self, task: str, max_rounds: int = 6) -> list[dict]:
        self.body.reset_task()                # 任务边界: 隔离使用厌恶状态
        self.messages = [{"role": "user", "content": task}]
        rounds = []
        for i in range(max_rounds):
            resp = self.body.chat_completion(self.messages, TOOLS_SCHEMA)
            tcs = resp.get("tool_calls", [])
            via = resp.get("via", "?")
            if not tcs:
                rounds.append({"round": i + 1, "via": via,
                               "content": resp.get("content", "")})
                break
            # harness 执行工具
            results = []
            for tc in tcs:
                fn = tc["function"]
                args = json.loads(fn.get("arguments") or "{}")
                results.append(TOOLS[fn["name"]](**args))
            # 调用前情境 (用于学习)
            ctx_before = list(self.messages)
            # 回传 + 学习 (带 context)
            self.body.tool_result_step(tcs, results, context_messages=ctx_before)
            # 追加消息历史
            self.messages.append({"role": "assistant", "content": "",
                                  "tool_calls": tcs})
            self.messages.append({"role": "tool", "content": results[0],
                                  "tool_call_id": tc.get("id", "0")})
            rounds.append({"round": i + 1, "via": via, "tool": fn["name"],
                           "args": args, "result": results[0][:50]})
        return rounds


def teach_calls(body):
    """教学阶段: 教调用模式 — 结构化三槽情境 (时序显式化).
    单步教学: 只填 task 槽. 多步转场教学: 填 last_call + last_result 槽,
    让"结果后情境"驱动下一步 — 运行时每轮 _context_z 构造同样三槽.
    learn_call 同时教工具世界模型 (z, a_t, z_next) — 教学教检索也教预测."""
    lessons = [
        # 单步调用 (task 槽)
        ({"task": "list the files in the directory"}, "glob",
         {"pattern": "*.py"}, "found 12: jepa_base.py, ...", 1.0),
        ({"task": "read the file jepa_base.py"}, "read",
         {"path": os.path.join(REPO_ROOT, "jepa_base.py")}, "import numpy as np ...", 1.0),
        ({"task": "search for class JEPA in the code"}, "grep",
         {"pattern": "class"}, "found 3: ...", 1.0),
        # 中间情境 (多步转场): 上一步结果后 → 转下一步 (泛化模式)
        ({"task": "list the files then read the first one",
          "last_call": 'glob: {"pattern": "*.py"}',
          "last_result": "found 12: jepa_base.py, ..."},
         "read", {"path": os.path.join(REPO_ROOT, "bb_component_check.py")},
         "import sys ...", 1.0),
        # 结果已知 → 收尾 (空调用, perf 低不误复用)
        ({"task": "read the file content shown now the task is complete"},
         "", {}, "done", 0.0),
    ]
    for obs, tool, args, result, perf in lessons:
        body.learn_call(obs, tool, args, result, perf)
    print(f"  已教 {body.call_mem.n()} 个调用模式 (含中间情境三槽)")
    print(f"  工具世界模型: 效果原型 {len(body._tool_effect)} 个")


def main():
    print("=" * 70)
    print("连续多步任务验证 — JEPA 学会完整工具调用 (语义三槽情境)")
    print("=" * 70)

    body = JepaBody(seed=7, config=PluginConfig(seed=7))
    if body.ensure_semantic():
        print("  ✅ MiniLM 语义编码就绪 (情境 3×384d 三槽)")
    else:
        print("  ⚠️ MiniLM 不可用, 哈希袋 384d 三槽兜底")
    # 先注入工具: 教学要教世界模型 (工具 action 列), 工具须先注册
    for t in TOOLS_SCHEMA:
        fn = t["function"]
        body.register_tool(fn["name"], TOOLS[fn["name"]],
                           fn["description"], fn["parameters"])
    teach_calls(body)

    # ── 任务 1: 单步 (list files) ────────────────────────
    print("\n[任务1] 'list the files in the directory' (应直接调 glob)")
    h1 = HarnessSim(body)
    r1 = h1.run_task("list the files in the directory")
    ok1 = any(x.get("tool") == "glob" for x in r1)
    print(f"  执行: {[(x.get('tool'), x.get('args')) for x in r1]}")
    print(f"  {'✅ 调用记忆命中 (工具+参数)' if ok1 else '❌ 未命中'}")

    # ── 任务 2: 真多步 (list → read, 第二步依赖第一步结果) ──
    print("\n[任务2] 'show me what python files exist then read the first one' "
          "(glob → read, 连续两步)")
    h2 = HarnessSim(body)
    r2 = h2.run_task("show me what python files exist then read the first one")
    tools2 = [x.get("tool") for x in r2 if x.get("tool")]
    print(f"  执行序列: {tools2} | "
          f"{[(x.get('tool'), x.get('args')) for x in r2]}")
    ok2 = (tools2 and tools2[0] == "glob" and "read" in tools2
           and len(tools2) <= 4)
    print(f"  {'✅ 真多步 (glob → read → 收尾)' if ok2 else '❌ 未完成真多步'}")

    # ── 任务 3: 学过的任务直接复用 ───────────────────────
    print("\n[任务3] 'read the file jepa_base.py' (直接复用 read 调用)")
    h3 = HarnessSim(body)
    r3 = h3.run_task("read the file jepa_base.py")
    ok3 = any(x.get("tool") == "read" and x.get("args", {}).get("path")
              for x in r3)
    print(f"  执行: {[(x.get('tool'), x.get('args')) for x in r3]}")
    print(f"  {'✅ 参数完整 (含 path)' if ok3 else '❌ 参数缺失'}")

    # ── 任务 4: 陌生任务不盲调 ───────────────────────────
    print("\n[任务4] 'send an email to boss' (陌生情境, 不应瞎调工具)")
    h4 = HarnessSim(body)
    r4 = h4.run_task("send an email to the boss")
    tools4 = [x.get("tool") for x in r4 if x.get("tool")]
    print(f"  执行: {[(x.get('via'), x.get('tool', x.get('content',''))[:40]) for x in r4]}")
    ok4 = not tools4
    print(f"  {'✅ 陌生任务不盲调 (诚实收尾)' if ok4 else '❌ 乱调工具'}")

    print("\n" + "=" * 70)
    print(f"结果: 任务1单步{ok1} 任务2多步{ok2} 任务3参数{ok3} 任务4不盲调{ok4}")
    print(f"调用记忆统计: {body.call_mem.stats_dict()}")
    overall = all([ok1, ok2, ok3, ok4])
    print(f"总体: {'✅ 连续多步任务能力成立' if overall else '⚠️ 有未通过项'}")
    print("=" * 70)
    return overall


if __name__ == "__main__":
    main()
