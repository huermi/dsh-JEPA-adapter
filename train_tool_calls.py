"""
工具调用可靠性训练 (train_tool_calls.py)
=========================================
持续训练 JEPA 的工具调用, 直到"可靠":
  可靠 = 熟悉任务类型稳定正确 (工具+参数), 陌生任务不乱调.

训练管线 (走 HTTP, 训练进服务器实例, dsh 直接用):
  1. 教学扩充: 每工具多任务变体 (覆盖 read 任务被 search 污染的
     0.634 余弦问题 — 同类记录多, 检索才可能命中正确的)
  2. 训练循环: 多 epoch 跑任务集 + 判定器 perfs 反馈 (答错记 perf=0)
  3. 睡眠巩固: 每 3 epoch 记忆回放固化 (call_mem→世界模型权重)
  4. 泛化评估: 训练未出现的任务变体 (测试集) → 成功率

达标标准: 训练集 ≥95%, 测试集 ≥80%.
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8045"


# ─── 真实工具执行器 ───────────────────────────────────────
def tool_glob(pattern="*"):
    import glob as g
    hits = sorted(g.glob(os.path.join(REPO_ROOT, "") + pattern))[:8]
    return f"found {len(hits)}: {', '.join(hits)}"


def tool_read(path=""):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read(300)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


def tool_grep(pattern="", path=REPO_ROOT):
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


def tool_calc(expr=""):
    try:
        if not expr:
            return "ERROR: missing expr"
        expr = (expr.replace("x", "*").replace("X", "*")
                .replace("plus", "+").replace("minus", "-"))
        return f"{expr} = {eval(expr)}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


TOOLS = {"glob": tool_glob, "read": tool_read, "grep": tool_grep,
         "calculator": tool_calc}
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
]


# ─── 判定器 (测试题答案判定) ──────────────────────────────
def judge(task: str, tool: str) -> float:
    t = task.lower()
    if "files" in t or "directory" in t or "list" in t or "glob" in t:
        return 1.0 if tool == "glob" else 0.0
    if "read" in t or "open" in t or "content" in t or "show the text" in t:
        return 1.0 if tool == "read" else 0.0
    if "search" in t or "find" in t or "class" in t or "look for" in t:
        return 1.0 if tool == "grep" else 0.0
    if "calculate" in t or "plus" in t or "minus" in t or "add" in t or "compute" in t or "times" in t:
        return 1.0 if tool == "calculator" else 0.0
    return 0.5   # 未知: 中性


# ─── HTTP helpers ─────────────────────────────────────────
def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def get(path):
    return post(path, None) if False else _get(path)


def _get(path):
    req = urllib.request.Request(BASE + path, method="GET")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


# ─── 教学扩充 (每工具多变体) ──────────────────────────────
LESSONS = [
    # glob 变体
    ({"task": "list the files in the directory"}, "glob", {"pattern": "*.py"},
     "found 12: jepa_base.py, ...", 1.0),
    ({"task": "show me what files exist"}, "glob", {"pattern": "*.py"},
     "found 12: jepa_base.py, ...", 1.0),
    ({"task": "find python files"}, "glob", {"pattern": "*.py"},
     "found 12: jepa_base.py, ...", 1.0),
    # read 变体 (同类多记录 → 检索对抗 search 污染)
    ({"task": "read the file jepa_base.py"}, "read",
     {"path": os.path.join(REPO_ROOT, "jepa_base.py")}, "import numpy as np ...", 1.0),
    ({"task": "open the file kernel.py"}, "read",
     {"path": os.path.join(REPO_ROOT, "body/kernel.py")}, "from __future__ import ...", 1.0),
    ({"task": "show the content of plugin_config.py"}, "read",
     {"path": os.path.join(REPO_ROOT, "body/plugin_config.py")}, "dataclass ...", 1.0),
    # grep 变体
    ({"task": "search for class JEPA in the code"}, "grep",
     {"pattern": "class"}, "found 3: ...", 1.0),
    ({"task": "find the word energy in system.py"}, "grep",
     {"pattern": "energy"}, "found 5: ...", 1.0),
    # calculator 变体
    ({"task": "calculate 2 plus 3"}, "calculator", {"expr": "2+3"},
     "2+3 = 5", 1.0),
    ({"task": "add 5 and 7"}, "calculator", {"expr": "5+7"},
     "5+7 = 12", 1.0),
    ({"task": "what is 12 minus 4"}, "calculator", {"expr": "12-4"},
     "12-4 = 8", 1.0),
    # 中间转场 (多步: glob 结果后 → read)
    ({"task": "list the files then read the first one",
      "last_call": 'glob: {"pattern": "*.py"}',
      "last_result": "found 12: jepa_base.py, ..."},
     "read", {"path": os.path.join(REPO_ROOT, "jepa_base.py")}, "import numpy as np ...", 1.0),
]

# 训练集 (含教学变体, 判定器反馈)
TRAIN_TASKS = [
    ("list the files in the directory", "glob"),
    ("show me what files exist", "glob"),
    ("read the file jepa_base.py", "read"),
    ("open the file kernel.py", "read"),
    ("search for class JEPA in the code", "grep"),
    ("calculate 2 plus 3", "calculator"),
    ("add 5 and 7", "calculator"),
    ("what is 12 minus 4", "calculator"),
]

# 测试集 (训练未出现的变体 → 泛化)
TEST_TASKS = [
    ("show me the py files", "glob"),
    ("open the file jepa_base.py", "read"),
    ("find the class WorldModel in the code", "grep"),
    ("compute 3 times 4", "calculator"),
    ("list files then read the first one", None),   # 多步: 首步 glob
]


# ─── 完整 harness 循环 (判定器 perfs) ────────────────────
def run_task(task, max_rounds=4):
    messages = [{"role": "user", "content": task}]
    seq, errs = [], 0
    for _ in range(max_rounds):
        resp = post("/v1/chat/completions",
                    {"messages": messages, "tools": TOOLS_SCHEMA,
                     "stream": False})
        msg = resp["choices"][0]["message"]
        tcs = msg.get("tool_calls", [])
        if not tcs:
            break
        results, perfs = [], []
        for tc in tcs:
            fn = tc["function"]
            name = fn["name"]
            args = json.loads(fn.get("arguments") or "{}")
            r = TOOLS[name](**args)
            results.append(r)
            perfs.append(judge(task, name))
            seq.append(name)
            if r.startswith("ERROR"):
                errs += 1
        post("/v1/tool_results",
             {"tool_calls": tcs, "tool_results": results,
              "context": list(messages), "perfs": perfs})
        messages.append({"role": "assistant", "content": "", "tool_calls": tcs})
        messages.append({"role": "tool", "content": results[0],
                         "tool_call_id": tcs[0].get("id", "0")})
    return seq, errs


def success(task, expect, seq, errs):
    """严格判定: 第一步调对工具, 序列聚焦, 无参数错误."""
    if expect is None:   # 多步任务: 首步 glob
        expect = "glob"
    return (seq and seq[0] == expect and len(seq) <= 2 and errs == 0)


def main():
    print("=" * 70)
    print("工具调用可靠性训练 (HTTP 8045)")
    print("=" * 70)
    try:
        st0 = get("/v1/status")
        print(f"server OK | call_mem={st0['call_mem']['n']} "
              f"effects={st0['tool_effects']}")
    except Exception as e:
        print(f"❌ server 未就绪: {e}")
        return

    # 1) 教学扩充
    n_teach = 0
    for obs, tool, args, result, perf in LESSONS:
        post("/v1/learn_call", {"obs": obs, "tool": tool, "args": args,
                                "result": result, "perf": perf})
        n_teach += 1
    print(f"① 教学扩充: +{n_teach} 条调用模式")

    # 2) 训练循环
    print("\n② 训练循环 (判定器反馈 + 睡眠巩固):")
    EPOCHS = 6
    for ep in range(1, EPOCHS + 1):
        ok = 0
        for task, expect in TRAIN_TASKS:
            seq, errs = run_task(task)
            if success(task, expect, seq, errs):
                ok += 1
        acc = ok / len(TRAIN_TASKS)
        st = get("/v1/status")
        print(f"  [epoch{ep}] 训练集 {acc:.0%} | "
              f"call_mem={st['call_mem']['n']} effects={st['tool_effects']}")
        if ep % 3 == 0:
            r = post("/v1/sleep", {})
            print(f"    └ 睡眠巩固: {r.get('note', 'ok')}")

    # 3) 测试集泛化
    print("\n③ 测试集泛化 (训练未出现的变体):")
    ok_t, details = 0, []
    for task, expect in TEST_TASKS:
        seq, errs = run_task(task)
        hit = success(task, expect, seq, errs)
        ok_t += hit
        details.append(f"  {'✅' if hit else '❌'} {task[:40]:<42} "
                       f"{'→'.join(seq) if seq else '(收尾)'}")
    acc_t = ok_t / len(TEST_TASKS)
    for d in details:
        print(d)
    print(f"  测试集泛化率: {acc_t:.0%}")

    # 4) 达标判定
    stf = get("/v1/status")
    print("\n" + "=" * 70)
    print(f"最终: call_mem={stf['call_mem']['n']} "
          f"effects={stf['tool_effects']} | 测试集 {acc_t:.0%}")
    if acc_t >= 0.8:
        print("✅ 工具调用可靠 (测试集 ≥80%)")
    else:
        print("⚠️ 未达标, 需继续训练/调参")
    print("=" * 70)
    return acc_t >= 0.8


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
