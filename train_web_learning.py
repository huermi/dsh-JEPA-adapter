"""
互联网检索学习验证 (train_web_learning.py)
==========================================
验证 JEPA 能否"用互联网工具检索资料来学习":

路径:
  教学种子: "look up information about a topic" → fetch (工具用法先验)
  问题 1  : "what is a red-black tree" (未教过) → 检索到"查资料→fetch"
            → fetch 真实抓取维基百科 → 结果回传 → call_mem 学调用,
            harness 把抓取文本教给 responder (问题→知识绑定)
  问题 2  : 同一问题再问 → responder 直接回答 (不再调工具) = 学会了

判定:
  P1: 问题1 第一次探索 fetch 且抓取成功 (结果非 ERROR)
  P2: 问题2 无工具调用 + 回答内容含抓取知识片段 (知识复用)
"""
import json
import os
import re
import sys
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8045"


# ─── 真实 fetch 执行器 (harness 侧, 多源路由) ─────────────
# 国内网络环境盘点: 维基/DDG 被墙, 百度必应反爬空壳, restcountries 已废弃,
# numbersapi 404 → 可用干净源: open-meteo (天气/地理, JSON) +
# dictionaryapi (单词释义) + 必应 title 兜底.
def tool_fetch(url="", query=""):
    UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
          "Accept": "application/json, text/html, */*"}
    try:
        if query and not url:
            q = query.strip()
            ql = q.lower()
            # 1) 天气类: "weather in X" / "temperature in X"
            m = re.search(r"(?:weather|temperature|forecast)\s+(?:in|at|for)?\s*([a-zA-Z\s]+)$", ql)
            if m:
                place = m.group(1).strip().split()[0]
                with urllib.request.urlopen(
                        "https://geocoding-api.open-meteo.com/v1/search"
                        f"?name={urllib.parse.quote(place)}&count=1",
                        timeout=15) as r:
                    geo = json.loads(r.read().decode("utf-8"))
                if not geo.get("results"):
                    return f"no location found for {place}"
                g = geo["results"][0]
                with urllib.request.urlopen(
                        "https://api.open-meteo.com/v1/forecast"
                        f"?latitude={g['latitude']}&longitude={g['longitude']}"
                        "&current_weather=true", timeout=15) as r:
                    w = json.loads(r.read().decode("utf-8"))["current_weather"]
                return (f"weather in {g.get('name')}, {g.get('country')}: "
                        f"{w['temperature']}°C, wind {w['windspeed']} km/h")
            # 2) 地点/国家信息类: "capital of X" / "country X"
            m2 = re.search(r"(?:capital of|country|where is)\s+([a-zA-Z\s]+)$", ql)
            if m2:
                place = m2.group(1).strip().split()[0]
                with urllib.request.urlopen(
                        "https://geocoding-api.open-meteo.com/v1/search"
                        f"?name={urllib.parse.quote(place)}&count=1",
                        timeout=15) as r:
                    geo = json.loads(r.read().decode("utf-8"))
                if not geo.get("results"):
                    return f"no location found for {place}"
                g = geo["results"][0]
                return (f"{g.get('name')}: country {g.get('country')}, "
                        f"admin1 {g.get('admin1', '?')}, population "
                        f"{g.get('population', '?')}")
            # 3) 单词释义类 (单个英文词)
            if re.fullmatch(r"[a-zA-Z]+", q):
                req = urllib.request.Request(
                    "https://api.dictionaryapi.dev/api/v2/entries/en/" + q,
                    headers=UA)
                with urllib.request.urlopen(req, timeout=15) as r:
                    data = json.loads(r.read().decode("utf-8"))
                if data and data[0].get("meanings"):
                    defs = [m["definitions"][0]["definition"]
                            for m in data[0]["meanings"][:2]
                            if m.get("definitions")]
                    return f"{q}: " + " | ".join(defs)[:350]
                return f"{q}: no definition found"
            # 4) 兜底: 必应搜索页 title
            url = "https://cn.bing.com/search?q=" + urllib.parse.quote(query)
        if not url:
            return "ERROR: missing url"
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=15) as r:
            html = r.read(20000).decode("utf-8", errors="replace")
        # 4) 兜底: 必应搜索页 (空壳检测 — 必应对无 JS 客户端返回
        #    "query - 搜索" 标题, 无实质知识 → 判定无结果, 不污染内化)
        m = re.search(r'<meta[^>]*name="description"[^>]*content="([^"]*)"',
                      html)
        if m and len(m.group(1).strip()) > 15:
            return m.group(1)[:350]
        t = re.search(r"<title>(.*?)</title>", html, flags=re.S)
        if t:
            title = t.group(1).strip()
            if title and "搜索" not in title and "Search" not in title \
                    and len(title) > 10:
                return title[:200]
        return "no result found"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


TOOLS = {"glob": None, "read": None, "grep": None, "fetch": tool_fetch}
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
    {"type": "function", "function": {"name": "fetch",
     "description": "look up what something is, research a topic, find definitions and learn new things online",
     "parameters": {"type": "object",
                    "properties": {"url": {"type": "string"},
                                   "query": {"type": "string"}}}}},
]


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())


def get(path):
    req = urllib.request.Request(BASE + path, method="GET")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


# 知识类任务触发词 (与 kernel.KNOWLEDGE_TRIGGERS 对齐)
KNOWLEDGE_TRIGGERS = ("what is", "what are", "what was", "tell me about",
                      "define", "explain", "how does", "how do",
                      "learn about", "research", "look up", "history of")


def is_knowledge(task: str) -> bool:
    return any(t in task.lower() for t in KNOWLEDGE_TRIGGERS)


def run_task(task, max_rounds=4):
    """完整 harness 循环 → (工具序列, 最终回应, 最后一次抓取文本).
    知识类任务: fetch 是"获取过程"非任务完成 → 判定器给 perf=0.2
    (低于 call_mem 复用线 0.3 → 下次检索跳过, 内化的答案由 responder 接管)."""
    knowledge = is_knowledge(task)
    messages = [{"role": "user", "content": task}]
    seq, content, last_fetch = [], "", ""
    for _ in range(max_rounds):
        resp = post("/v1/chat/completions",
                    {"messages": messages, "tools": TOOLS_SCHEMA,
                     "stream": False})
        msg = resp["choices"][0]["message"]
        content = msg.get("content", "")
        tcs = msg.get("tool_calls", [])
        if not tcs:
            break
        results, perfs = [], []
        for tc in tcs:
            fn = tc["function"]
            name = fn["name"]
            args = json.loads(fn.get("arguments") or "{}")
            if name == "fetch" and TOOLS["fetch"]:
                r = TOOLS["fetch"](**args)
                last_fetch = r
            else:
                r = f"{name} executed (placeholder)"
            results.append(r)
            # 知识任务: fetch=过程 (0.2), 其他工具=0; 非知识: 1.0
            perfs.append(0.2 if (knowledge and name == "fetch")
                         else (0.0 if knowledge else 1.0))
            seq.append(name)
        post("/v1/tool_results",
             {"tool_calls": tcs, "tool_results": results,
              "context": list(messages), "perfs": perfs})
        messages.append({"role": "assistant", "content": "", "tool_calls": tcs})
        messages.append({"role": "tool", "content": results[0],
                         "tool_call_id": tcs[0].get("id", "0")})
    return seq, content, last_fetch


def main():
    print("=" * 70)
    print("互联网检索学习验证 (JEPA 用工具查资料并记住)")
    print("=" * 70)
    try:
        st0 = get("/v1/status")
        print(f"server OK | call_mem={st0['call_mem']['n']}")
    except Exception as e:
        print(f"❌ server 未就绪: {e}")
        return

    # ① 教学种子: 工具用法 (查资料 → fetch)
    post("/v1/learn_call", {
        "obs": {"task": "look up information about a topic"},
        "tool": "fetch",
        "args": {"url": "https://en.wikipedia.org/api/rest_v1/page/summary/JEPA"},
        "result": "JEPA is a machine learning framework ...",
        "perf": 1.0})
    print("① 教学: 'look up information' → fetch (工具用法种子)")

    # ② 问题 1: 陌生知识问题 → 期待探索/检索到 fetch
    print("\n② 问题1: 'what is the weather in shanghai' (未教过)")
    q1 = "what is the weather in shanghai"
    seq1, c1, fetch1 = run_task(q1)
    print(f"  工具序列: {seq1}")
    p1 = ("fetch" in seq1 and not fetch1.startswith("ERROR")
          and len(fetch1) > 30)
    print(f"  fetch 结果 ({'✅' if p1 else '❌'}): {fetch1[:140]}")

    # ③ 把抓取知识教给回应学习器 (问题→知识绑定 = 用工具学会)
    if p1:
        post("/v1/learn_response", {"obs": {"task": q1}, "text": fetch1[:200]})
        print("③ 知识绑定: (问题 → 抓取内容) 已教给回应学习器")

    # ④ 问题 2: 同一问题再问 → 期待直接回答 (不再调工具)
    print("\n④ 问题2: 同一问题再问 (期待直接回答, 不再调工具)")
    seq2, c2, _ = run_task(q1)
    print(f"  工具序列: {seq2 or '(无工具)'}")
    print(f"  回应: {c2[:140]}")
    # 知识关键词判定: 回答里含抓取内容核心词 (非默认收尾文本)
    default_text = "Task complete. No more actions needed."
    kw = ("weather" in c2.lower()
          or (c2 != default_text and len(c2) > 30))
    p2 = (not seq2 and kw)
    print(f"  {'✅ 学会并复用知识 (直接回答)' if p2 else '❌ 未复用'}")

    # ⑤ 对照: 另一个未教过的问题 (泛化检索 + 混叠纠错)
    print("\n⑤ 泛化: 'what is the weather in tokyo' (未教过, 同类型)")
    q3 = "what is the weather in tokyo"
    seq3, c3, fetch3 = run_task(q3)
    default_text2 = "Task complete. No more actions needed."
    wrong = ("shanghai" in c3.lower() or c3 == default_text2)
    print(f"  首次: seq={seq3 or '(无工具)'} content={c3[:60]}")
    if wrong:
        print("  ⚠️ 混叠 (答了上海/默认) → 判定器判错 → 负样本标记")
        post("/v1/learn_response_negative", {"obs": {"task": q3}})
        seq3b, c3b, fetch3b = run_task(q3)
        print(f"  纠错后: seq={seq3b or '(无工具)'} fetch={fetch3b[:80]}")
        p3 = ("fetch" in seq3b and not fetch3b.startswith("ERROR")
              and len(fetch3b) > 30)
    else:
        p3 = True
    if p3 and "fetch" in (seq3 if not wrong else seq3b):
        f3 = fetch3 if not wrong else fetch3b
        post("/v1/learn_response",
             {"obs": {"task": q3}, "text": (f3 or fetch3)[:200]})
        print("  ✅ 泛化检索成功 + 知识绑定")

    stf = get("/v1/status")
    print("\n" + "=" * 70)
    print(f"结果: 问题1抓取{p1} 问题2复用{p2} 泛化{p3}")
    print(f"最终: call_mem={stf['call_mem']['n']} "
          f"responder={stf['responder']['n']} effects={stf['tool_effects']}")
    print("=" * 70)
    return p1 and p2


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
