"""
自由学习验证 (free_learning_check.py)
======================================
验证 JEPA 在不教"任务答案"的情况下, 靠探索 + 结果学习能否学会做事.

两模式对照:
  A. 纯自由学习: 零教学种子. 观察 JEPA 面对未知任务/工具参数的表现
     (预期: 语义探索能选对工具, 但参数生成是检索式边界 → 缺参失败)
  B. 工具用法种子: 只教"工具怎么调" (参数示例), 不教任务答案.
     任务用变体文本 → 测语义泛化检索.
     (P4c 结论: 先验种子 + 经验接管)

每任务 = 一个"公开测试题"风格的请求. 多 epoch 看学习曲线.
判定: 任务成功 = 执行序列中出现预期工具 (行为正确).
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import json
import os
import sys

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "body"))

from kernel import JepaBody
from plugin_config import PluginConfig


# ─── 真实工具执行器 (harness 侧) ──────────────────────────
def tool_glob(pattern: str = "*") -> str:
    import glob as g
    hits = sorted(g.glob(os.path.join(REPO_ROOT, "") + pattern))[:8]
    return f"found {len(hits)}: {', '.join(hits)}"


def tool_read(path: str = "") -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(300)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def tool_grep(pattern: str = "", path: str = REPO_ROOT) -> str:
    try:
        hits = []
        for f in sorted(os.listdir(path)):
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


def tool_calc(expr: str = "") -> str:
    try:
        if not expr:
            return "ERROR: missing expr"
        expr = expr.replace("x", "*").replace("X", "*").replace("plus", "+").replace("minus", "-")
        return f"{expr} = {eval(expr)}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def tool_write(path: str = "", content: str = "") -> str:
    try:
        if not path:
            return "ERROR: missing path"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"ERROR: {e}"


TOOLS = {"glob": tool_glob, "read": tool_read, "grep": tool_grep,
         "calculator": tool_calc, "write_file": tool_write}
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
    {"type": "function", "function": {"name": "calculator",
     "description": "perform arithmetic calculations",
     "parameters": {"type": "object",
                    "properties": {"expr": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "write_file",
     "description": "write content to a file",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"},
                                   "content": {"type": "string"}}}}},
]

# 任务集 (公开测试题风格): (任务文本, 预期工具)
TASKS = [
    ("show me what files are in the directory", "glob"),
    ("calculate 2 plus 3", "calculator"),
    ("search for the JEPA class in the code", "grep"),
    ("read the file jepa_base.py", "read"),
]

# 工具用法种子 (模式 B): 只教参数怎么调, 不教任务答案.
# 种子任务文本 = 工具的泛化原型 (与目标任务语义区分, 不泄露答案)
USAGE_SEEDS = [
    ({"task": "show what files exist"}, "glob", {"pattern": "*.py"},
     "found 12: jepa_base.py, ...", 1.0),
    ({"task": "add two numbers"}, "calculator", {"expr": "2+3"},
     "2+3 = 5", 1.0),
    ({"task": "find the class definition"}, "grep", {"pattern": "class"},
     "found 3: ...", 1.0),
    ({"task": "display text from a file"}, "read", {"path": os.path.join(REPO_ROOT, "jepa_base.py")},
     "import numpy as np ...", 1.0),
]


# ─── 测试题判定器 (模拟公开测试题的答案判定) ──────────────
# 每轮工具调用后判定: 这轮用的工具是否朝向任务 → perf 反馈.
# 这是"自由学习纠错"的关键信号: 用错工具 → perf=0 → 记忆记下
# "这情境不该这么调" → 下轮探索/检索会跳过 (鸡生蛋问题的破局).
def judge(task: str, tool: str) -> float:
    t = task.lower()
    if "files" in t or "directory" in t or "list" in t:
        return 1.0 if tool == "glob" else 0.0
    if "calculate" in t or "plus" in t or "add" in t or "minus" in t:
        return 1.0 if tool == "calculator" else 0.0
    if "search" in t or "class" in t or "find" in t:
        return 1.0 if tool == "grep" else 0.0
    if "read" in t or "file content" in t:
        return 1.0 if tool == "read" else 0.0
    return 0.5   # 未知任务类型: 中性


class Harness:
    def __init__(self, body: JepaBody, use_judge: bool = True):
        self.body = body
        self.use_judge = use_judge

    def run_task(self, task: str, max_rounds: int = 5) -> list[str]:
        """完整 harness 循环 → 返回工具执行序列.
        use_judge=True: 每轮调用后判定器给 perf 反馈 (测试题场景)."""
        self.body.reset_task()
        messages = [{"role": "user", "content": task}]
        seq = []
        for _ in range(max_rounds):
            resp = self.body.chat_completion(messages, TOOLS_SCHEMA)
            tcs = resp.get("tool_calls", [])
            if not tcs:
                break
            results = []
            perfs = []
            for tc in tcs:
                fn = tc["function"]
                name = fn["name"]
                args = json.loads(fn.get("arguments") or "{}")
                results.append(TOOLS.get(name, lambda **k: f"no {name}")(**args))
                perfs.append(judge(task, name) if self.use_judge else 1.0)
                seq.append(name)
            ctx = list(messages)
            self.body.tool_result_step(tcs, results, context_messages=ctx,
                                       perfs=perfs)
            messages.append({"role": "assistant", "content": "", "tool_calls": tcs})
            messages.append({"role": "tool", "content": results[0],
                             "tool_call_id": tcs[0].get("id", "0")})
        return seq


def run_epochs(body: JepaBody, epochs: int = 6, label: str = "",
               use_judge: bool = True) -> list[float]:
    h = Harness(body, use_judge=use_judge)
    curve = []
    for ep in range(1, epochs + 1):
        ok = 0
        detail = []
        for task, expect in TASKS:
            seq = h.run_task(task)
            # 严格判定: 第一步就调对工具, 且序列聚焦 (≤2: 一次调用+收尾)
            hit = (seq and seq[0] == expect and len(seq) <= 2)
            ok += hit
            flag = "✅" if hit else ("⚠️" if (seq and seq[0] == expect) else "❌")
            detail.append(f"{flag}{task[:26]}:{seq}")
        acc = ok / len(TASKS)
        curve.append(acc)
        print(f"  [epoch{ep}] 聚焦成功率 {acc:.0%} | "
              f"call_mem={body.call_mem.n()} 效果原型={len(body._tool_effect)} "
              f"| {label}")
        for d in detail:
            print(f"      {d}")
    return curve


def main():
    print("=" * 70)
    print("自由学习验证 — 探索 + 结果学习 (不教任务答案)")
    print("=" * 70)

    # ── 模式 A: 纯自由学习 (零种子) ──────────────────────
    print("\n[模式A] 纯自由学习 (零教学种子):")
    bodyA = JepaBody(seed=7, config=PluginConfig(seed=7))
    for t in TOOLS_SCHEMA:
        fn = t["function"]
        bodyA.register_tool(fn["name"], TOOLS[fn["name"]],
                            fn["description"], fn["parameters"])
    bodyA.ensure_semantic()
    curveA = run_epochs(bodyA, epochs=5, label="A:纯自由")
    print(f"  曲线: {['%.0f%%' % (c*100) for c in curveA]}")

    # ── 模式 B: 工具用法种子 (只教怎么调) ────────────────
    print("\n[模式B] 工具用法种子 (参数示例, 不教任务答案):")
    bodyB = JepaBody(seed=7, config=PluginConfig(seed=7))
    for t in TOOLS_SCHEMA:
        fn = t["function"]
        bodyB.register_tool(fn["name"], TOOLS[fn["name"]],
                            fn["description"], fn["parameters"])
    bodyB.ensure_semantic()
    for obs, tool, args, result, perf in USAGE_SEEDS:
        bodyB.learn_call(obs, tool, args, result, perf)
    curveB = run_epochs(bodyB, epochs=5, label="B:种子")
    print(f"  曲线: {['%.0f%%' % (c*100) for c in curveB]}")

    print("\n" + "=" * 70)
    print(f"模式A (纯自由): {['%.0f%%' % (c*100) for c in curveA]}")
    print(f"模式B (种子):   {['%.0f%%' % (c*100) for c in curveB]}")
    print("结论: 探索让纯自由学习学会选工具; 参数生成是检索式边界, "
          "种子教用法后成功率应上升")
    print("=" * 70)


if __name__ == "__main__":
    main()
