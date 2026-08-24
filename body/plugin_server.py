"""
DeepSeek harness 插件 (body/plugin_server.py)
==============================================
作为 DeepSeek harness 与 JepaBody 的桥:
  harness (OpenAI 兼容协议) ⇄ 本插件 (HTTP 服务器) ⇄ JepaBody

端点:
  POST /v1/chat/completions  对话 + 工具调用 (OpenAI function-calling 格式)
  GET  /v1/status            模型状态 + 开关矩阵
  GET  /v1/models            模型列表 (JEPA 可选, 支持 OpenAI 兼容 harness)
  GET  /v1/config            读取当前配置 (设置界面数据源)
  POST /v1/config            热更新配置 (校验类型)
  GET  /ui                   设置界面 (HTML, 全部可配置项可视化)
  POST /v1/sleep             触发睡眠巩固 (记忆回放)
  POST /v1/learn_response    显式教回应 (情境→文本, 学会输出字符)
  POST /v1/archive           存档 (save/load)

配置 (环境变量, 见 plugin_config.PluginConfig):
  JEPA_PORT=8031
  JEPA_LEARNING=0|1   JEPA_MEMORY=0|1   JEPA_SLEEP=0|1
  JEPA_TOOLS=0|1      JEPA_RESPOND_MODE=retrieval|llm   JEPA_ARCHIVE=0|1
  JEPA_CONFIG=config.json (可选, JSON 开关配置)

启动:
  python body/plugin_server.py   (读取环境变量)
"""
from __future__ import annotations

import json
import sys
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kernel import JepaBody
from plugin_config import PluginConfig


class PluginServer:
    """插件桥: 持有 JepaBody 单例 + 提供 HTTP 处理逻辑"""

    def __init__(self, body: JepaBody | None = None,
                 config: PluginConfig | None = None):
        self.config = config or PluginConfig.from_env()
        # 从环境变量加载 config.json (若指定)
        if os.environ.get("JEPA_CONFIG"):
            try:
                self.config = PluginConfig.from_json(os.environ["JEPA_CONFIG"])
            except Exception as e:
                print(f"[plugin] config.json 加载失败, 用环境配置: {e}")
        self.body = body or JepaBody(config=self.config)
        self.lock = threading.Lock()     # JepaBody 非线程安全, 串行化
        # MiniLM 语义编码 (惰性加载, 失败回退哈希袋 — 不阻断启动)
        # 情境 = 3×384d 三槽 (task/last_call/last_result), 时序显式化
        try:
            if self.body.ensure_semantic():
                print("[plugin] MiniLM 语义编码已接入 (情境三槽 1152d)")
        except Exception as e:
            print(f"[plugin] MiniLM 接入失败, 哈希袋兜底: {e}")
        # Qwen 语义编码可选 (CPU 太慢, 默认关; 优先级高于 MiniLM)
        # 工具白名单: 只向模型暴露"做事"工具 (dsh 真实工具集
        # = pwsh/read/edit/glob/grep/str_replace_editor 等; 哈希袋语义弱,
        # 全 25 工具上会乱选 create_goal/interrupt_agent 等内部工具)
        self._allowed_tools = {
            "pwsh", "bash", "shell", "execute", "run", "run_command",
            "read", "edit", "write", "write_file", "read_file",
            "edit_file", "file_read", "file_write", "str_replace_editor",
            "list_dir", "ls", "glob", "grep", "search", "find", "cat",
            "web", "fetch", "http", "calculator",
            "python", "run_python", "task_done", "finish", "complete",
        }
        # Qwen 语义编码可选 (CPU 太慢, 默认关; 哈希袋+白名单够小工具集)
        if os.environ.get("JEPA_QWEN") == "1":
            try:
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from p4b_check import load_qwen, QwenPerception
                load_qwen()
                self.body.set_text_encoder(QwenPerception())
                print("[plugin] Qwen 语义编码已接入 (工具选择)")
            except Exception as e:
                print(f"[plugin] Qwen 加载失败, 用哈希袋: {type(e).__name__}: {e}")
        print(f"[plugin] JepaBody ready | {self.config.describe()}")

    # ── 请求处理 ──────────────────────────────────────────
    def handle_chat(self, payload: dict) -> dict:
        messages = payload.get("messages", [])
        tools = payload.get("tools") or payload.get("functions")
        # 诊断: 记录 dsh 实际注入的工具名
        if tools:
            names = [t.get("function", {}).get("name", "?") for t in tools]
            print(f"[plugin] harness 注入 {len(names)} 工具: {names[:20]}", flush=True)
            # 工具白名单过滤: 模型只能看到做事工具
            tools = [t for t in tools
                     if t.get("function", {}).get("name", "")
                     in self._allowed_tools]
            allowed = [t.get("function", {}).get("name") for t in tools]
            print(f"[plugin] 白名单后暴露: {allowed}", flush=True)
        with self.lock:
            resp = self.body.chat_completion(messages, tools)
        return resp
    def handle_tool_results(self, payload: dict) -> dict:
        """harness 执行工具后回传: {tool_calls, tool_results, context?, perfs?}
        perfs: 任务级绩效列表 (可选) — 测试题答案判定器等外部信号."""
        tc = payload.get("tool_calls", [])
        tr = payload.get("tool_results", [])
        ctx = payload.get("context")        # 可选: 工具执行前的消息历史
        perfs = payload.get("perfs")        # 可选: 任务级绩效
        with self.lock:
            return self.body.tool_result_step(tc, tr, context_messages=ctx,
                                              perfs=perfs)

    def handle_learn_call(self, payload: dict) -> dict:
        """显式教调用: {obs, tool, args, result} — 学会完整工具调用"""
        with self.lock:
            self.body.learn_call(payload.get("obs", ""),
                                 payload.get("tool", ""),
                                 payload.get("args", {}),
                                 payload.get("result", ""),
                                 float(payload.get("perf", 1.0)))
        return {"ok": True, "n": self.body.call_mem.n()}

    def handle_learn_response(self, payload: dict) -> dict:
        """显式教回应: {obs: str, text: str} — 让模型学会输出字符"""
        obs = payload.get("obs", "")
        text = payload.get("text", "")
        with self.lock:
            self.body.learn_response(obs, text)
        return {"ok": True, "n": self.body.responder.n()}

    def handle_learn_response_negative(self, payload: dict) -> dict:
        """标记答错: {obs} — 该情境的旧回应不可靠, 下次重新获取.
        知识混叠纠错: 相似问题不同答案 (判定器发现答错时调用)."""
        obs = payload.get("obs", "")
        with self.lock:
            self.body.learn_response_negative(obs)
        return {"ok": True, "negatives": len(self.body.responder.negatives)}

    def handle_sleep(self, payload: dict) -> dict:
        with self.lock:
            r = self.body.sleep()
        return {"ok": True, **r}

    def handle_archive(self, payload: dict) -> dict:
        """{action: save|load, path: str}"""
        action = payload.get("action", "save")
        path = payload.get("path", "jepa_body.pkl")
        with self.lock:
            if action == "save":
                self.body.save(path)
                return {"ok": True, "action": "save", "path": path}
            self.body.load(path)
            return {"ok": True, "action": "load", "path": path}

    def status(self) -> dict:
        with self.lock:
            return self.body.status()

    # ── 配置接口 (设置界面数据源 + 热更新) ────────────────
    def handle_config(self, payload: dict | None = None) -> dict:
        """GET (payload=None): 返回配置全量 + schema (设置界面数据源).
        POST: 更新合法键 (按当前类型转换), 热应用到运行中的 body."""
        cfg = self.config
        if payload is None:
            return {"ok": True, "config": cfg.to_dict(),
                    "schema": CONFIG_SCHEMA}
        updated = {}
        for k, v in payload.items():
            if not hasattr(cfg, k):
                continue
            cur = getattr(cfg, k)
            try:
                if isinstance(cur, bool):
                    nv = bool(v) if isinstance(v, bool) else \
                        str(v).strip().lower() in ("1", "true", "yes", "on")
                elif isinstance(cur, int):
                    nv = int(v)
                elif isinstance(cur, float):
                    nv = float(v)
                elif isinstance(cur, str):
                    nv = str(v)
                else:
                    continue
            except (TypeError, ValueError):
                continue
            if nv != cur:
                setattr(cfg, k, nv)
                updated[k] = nv
        if updated:
            self._apply_config(updated)
        return {"ok": True, "updated": updated}

    def _apply_config(self, updated: dict) -> None:
        """热更新运行中 body 的响应式参数 (responder 侧)."""
        r = self.body.responder
        if "respond_cap" in updated:
            r.cap = updated["respond_cap"]
        if "respond_min_sim" in updated:
            r.min_sim = updated["respond_min_sim"]
        if "soft_align" in updated:
            r.soft_align_enabled = updated["soft_align"]
        if "soft_align_alpha" in updated:
            r.soft_align_alpha = updated["soft_align_alpha"]


# ── 设置界面 schema (字段 → 中文说明, 供 /ui 与 /v1/config) ──
CONFIG_SCHEMA = {
    "learning":      {"type": "bool", "label": "在线学习",
                      "desc": "任务经验是否改变权重 (在线权重梯度)"},
    "memory":        {"type": "bool", "label": "记忆写入",
                      "desc": "任务经验是否进入记忆 (惊讶门控)"},
    "sleep":         {"type": "bool", "label": "睡眠巩固",
                      "desc": "记忆回放固化 (默认关, 手动/周期触发)"},
    "sleep_epochs":  {"type": "int", "label": "睡眠轮数",
                      "desc": "睡眠巩固的重放轮数"},
    "sleep_lr_scale": {"type": "float", "label": "睡眠学习率",
                       "desc": "睡眠回放的权重更新步长缩放"},
    "sleep_prio_mix": {"type": "float", "label": "优先级混合",
                       "desc": "睡眠重放的优先级混合系数"},
    "tools":         {"type": "bool", "label": "工具调用",
                      "desc": "决策是否产生 tool_calls (做事能力)"},
    "respond_mode":  {"type": "str", "label": "响应模式",
                      "desc": "retrieval=检索式学会说话 | llm=桥接外部 LLM"},
    "archive":       {"type": "bool", "label": "存档",
                      "desc": "任务级快照 save/load"},
    "surprise_thresh": {"type": "float", "label": "惊讶阈值",
                        "desc": "E1 高于此触发工具调用"},
    "max_memory":    {"type": "int", "label": "记忆上限",
                      "desc": "短期记忆条目上限"},
    "explore":       {"type": "bool", "label": "自由探索",
                      "desc": "检索未命中 → 语义试错 (预算内)"},
    "explore_budget": {"type": "int", "label": "探索预算",
                       "desc": "每任务最大探索次数"},
    "explore_decay": {"type": "float", "label": "探索温度衰减",
                      "desc": "每次探索后温度衰减系数"},
    "benchmark_mode": {"type": "bool", "label": "基准模式",
                       "desc": "探索关闭, 只用内化知识 (防作弊)"},
    "respond_cap":   {"type": "int", "label": "回应经验上限",
                      "desc": "responder 检索条目容量"},
    "respond_min_sim": {"type": "float", "label": "回应检索阈值",
                        "desc": "检索命中最低相似度 (低 → 弱回忆也答)"},
    "soft_align":    {"type": "bool", "label": "软校准 (AdaJEPA)",
                      "desc": "判定正确后条目表征向查询微调 (TTA)"},
    "soft_align_alpha": {"type": "float", "label": "软校准步长",
                         "desc": "每步校准步长 (小 → 稳定)"},
    "seed":          {"type": "int", "label": "随机种子",
                      "desc": "随机数种子 (复现用)"},
}


class _Handler(BaseHTTPRequestHandler):
    server: PluginServer   # 注入

    def log_message(self, *a):      # 静默请求日志 (保持终端干净)
        pass

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_chat(self, resp: dict, model: str = "jepa-1"):
        """OpenAI 兼容 SSE 流式响应 (dsh/pi-ai 默认流式).
        内容一次发完 (delta 带全部 content/tool_calls), 然后 finish_reason."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")   # 关键: SSE 后必须关闭,
        # 否则 pi-ai 等流 EOF 卡死 (headless timeout 124 的根因)
        self.end_headers()

        content = resp.get("content", "")
        tcs = resp.get("tool_calls", [])
        finish = "tool_calls" if tcs else "stop"

        # 内容 chunk (先发 content, 再发 tool_calls)
        if content:
            chunk = {
                "id": f"chatcmpl-jepa-{self.server.body.t}",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0,
                             "delta": {"role": "assistant", "content": content},
                             "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        for i, tc in enumerate(tcs):
            chunk = {
                "id": f"chatcmpl-jepa-{self.server.body.t}",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{
                        "index": i, "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {"name": tc["function"]["name"],
                                     "arguments": tc["function"].get("arguments", "{}")}}],
                }, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        # 结束 chunk
        end = {
            "id": f"chatcmpl-jepa-{self.server.body.t}",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        self.wfile.write(f"data: {json.dumps(end)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True        # 强制关闭, 让客户端读到 EOF

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw or b"{}")
        except Exception:
            return {}

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/v1/status":
            self._send(200, self.server.status())
        elif path == "/v1/config":
            self._send(200, self.server.handle_config())
        elif path == "/v1/models":
            # dsh/OpenAI 兼容客户端: 拉模型列表 → JEPA 出现在模型选择中
            self._send(200, {"object": "list", "data": [
                {"id": "jepa-1", "object": "model", "created": 0,
                 "owned_by": "jepa",
                 "name": "JEPA DCA-4.0",
                 "description": "DCA 检索式认知体: 检索+权重记忆+矛盾处理"}]})
        elif path == "/ui":
            self._send_html(UI_PAGE)
        else:
            self._send(404, {"error": f"unknown endpoint {path}"})

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        path = urlparse(self.path).path
        p = self._read_json()
        srv = self.server
        try:
            if path == "/v1/chat/completions":
                resp = srv.handle_chat(p)
                # 流式请求 → SSE; 否则普通 JSON
                if p.get("stream"):
                    self._send_sse_chat(resp)
                else:
                    # 非流式: OpenAI 标准结构
                    self._send(200, {
                        "id": f"chatcmpl-jepa-{srv.body.t}",
                        "object": "chat.completion",
                        "model": "jepa-1",
                        "choices": [{"index": 0,
                                     "message": {"role": "assistant",
                                                 "content": resp.get("content", ""),
                                                 "tool_calls": resp.get("tool_calls", [])},
                                     "finish_reason": "tool_calls"
                                     if resp.get("tool_calls") else "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                                  "total_tokens": 0},
                    })
            elif path == "/v1/config":
                self._send(200, srv.handle_config(p))
            elif path == "/v1/tool_results":
                self._send(200, srv.handle_tool_results(p))
            elif path == "/v1/learn_response":
                self._send(200, srv.handle_learn_response(p))
            elif path == "/v1/learn_response_negative":
                self._send(200, srv.handle_learn_response_negative(p))
            elif path == "/v1/learn_call":
                self._send(200, srv.handle_learn_call(p))
            elif path == "/v1/sleep":
                self._send(200, srv.handle_sleep(p))
            elif path == "/v1/archive":
                self._send(200, srv.handle_archive(p))
            else:
                self._send(404, {"error": f"unknown endpoint {path}"})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def start_server(port: int = 8031, host: str = "127.0.0.1",
                 config: PluginConfig | None = None,
                 body: JepaBody | None = None) -> ThreadingHTTPServer:
    """启动插件服务器 (独立进程用; 测试时可在线程中跑)."""
    plugin = PluginServer(body=body, config=config)
    srv = ThreadingHTTPServer((host, port), _Handler)
    srv.body = plugin.body                # SSE chunk id 用
    srv.status = plugin.status
    srv.handle_config = plugin.handle_config
    srv.handle_chat = plugin.handle_chat
    srv.handle_tool_results = plugin.handle_tool_results
    srv.handle_learn_response = plugin.handle_learn_response
    srv.handle_learn_response_negative = plugin.handle_learn_response_negative
    srv.handle_learn_call = plugin.handle_learn_call
    srv.handle_sleep = plugin.handle_sleep
    srv.handle_archive = plugin.handle_archive
    print(f"[plugin] listening on http://{host}:{port}")
    return srv


def _port_in_use(port: int) -> bool:
    """端口占用探测: 绑定失败 = 已被占用 (防多实例堆积 — 8032-8044 事故教训)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return False
        except OSError:
            return True


UI_PAGE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>JEPA DCA-4.0 设置</title>
<style>
  body { background:#111; color:#ddd; font-family:'Segoe UI',sans-serif;
         max-width:760px; margin:0 auto; padding:24px; }
  h1 { color:#7fd; font-size:20px; border-bottom:1px solid #333; padding-bottom:10px; }
  .grp { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:8px;
         padding:14px 18px; margin:12px 0; }
  .grp h2 { color:#9be; font-size:14px; margin:0 0 10px; }
  .item { display:flex; align-items:center; gap:12px; padding:7px 0;
          border-bottom:1px solid #222; }
  .item:last-child { border-bottom:none; }
  .item label { flex:1; font-size:13px; }
  .item .desc { color:#888; font-size:11px; display:block; margin-top:2px; }
  input[type=text],input[type=number],select { background:#222; color:#ddd;
    border:1px solid #444; border-radius:4px; padding:5px 8px; width:120px; }
  input[type=checkbox] { width:18px; height:18px; accent-color:#4af; }
  .bar { display:flex; gap:10px; margin-top:16px; }
  button { background:#25a; color:#fff; border:none; border-radius:6px;
           padding:9px 22px; font-size:14px; cursor:pointer; }
  button:hover { background:#36b; }
  #msg { color:#5d5; font-size:12px; margin-left:12px; }
  .model { background:#132; border:1px solid #264; border-radius:8px;
           padding:12px 16px; font-size:13px; margin-top:16px; }
  .model b { color:#7fd; }
</style></head><body>
<h1>JEPA DCA-4.0 — 认知体设置</h1>
<div class="model">模型 ID: <b>jepa-1</b> · 在支持 OpenAI 兼容接口的 harness
  (DeepSeek / Claude Code / 自定义客户端) 的<b>模型选择</b>中选择
  <b>jepa-1</b> 即可调用本认知体。</div>
<div id="form"></div>
<div class="bar">
  <button onclick="save()">保存配置</button>
  <button onclick="load()">刷新</button>
  <span id="msg"></span>
</div>
<script>
let cfg = {};
async function load() {
  const r = await fetch('/v1/config');
  const d = await r.json();
  cfg = d.config || {};
  window.__schema = d.schema || {};
  const schema = window.__schema;
  const box = document.getElementById('form');
  box.innerHTML = '';
  for (const k in schema) {
    const s = schema[k], v = cfg[k];
    const item = document.createElement('div');
    item.className = 'item';
    let input;
    if (s.type === 'bool') {
      input = `<input type="checkbox" id="f_${k}" ${v ? 'checked' : ''}>`;
    } else if (s.type === 'int' || s.type === 'float') {
      input = `<input type="number" id="f_${k}" step="${s.type==='float'?'0.05':'1'}"
               value="${v}">`;
    } else {
      input = `<select id="f_${k}">
        ${(v==='retrieval')?'<option selected>retrieval</option>':'<option>retrieval</option>'}
        ${(v==='llm')?'<option selected>llm</option>':'<option>llm</option>'}</select>`;
    }
    item.innerHTML = `<label>${s.label}<span class="desc">${s.desc}</span></label>${input}`;
    box.appendChild(item);
  }
}
// 收集表单值
function collect() {
  const p = {};
  const schema = window.__schema || {};
  for (const k in schema) {
    const el = document.getElementById('f_' + k);
    if (!el) continue;
    p[k] = (schema[k].type === 'bool') ? el.checked :
           (schema[k].type === 'int') ? parseInt(el.value, 10) : parseFloat(el.value);
  }
  return p;
}
async function save() {
  const r = await fetch('/v1/config', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(collect())});
  const d = await r.json();
  document.getElementById('msg').textContent =
    '已更新: ' + Object.keys(d.updated||{}).join(', ') + ' ✓';
}
load();
</script></body></html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("JEPA_PORT", "8031"))
    if _port_in_use(port):
        print(f"[plugin] ❌ 端口 {port} 已被占用 — 已有实例在跑? "
              f"不要重复启动 (历史教训: 多实例堆积 8032-8044). "
              f"如需重启: 先杀旧进程再启动.", flush=True)
        sys.exit(1)
    srv = start_server(port=port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[plugin] shutdown")


# ── 设置界面 (HTML, 深色主题, 动态生成全部可配置项) ──
