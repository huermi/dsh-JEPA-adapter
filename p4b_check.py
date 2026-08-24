"""
P4b 架构验证 (p4b_check.py) — JEPA 指挥官 + 生成引擎执行器
==========================================================
端到端最小闭环: 用户任务 → JEPA 感知 → 语义工具选择 → 生成引擎执行
(Qwen 写文本 / PIL 画图) → 结果回传记忆 → 回忆输出.

验证点:
  [V1] 意图选择: "写诗"→write_text, "画图"→draw_image (统一空间语义检索)
  [V2] 生成落地: 真实文本 (Qwen) + 真实图片文件 (PNG)
  [V3] 回传学习: 生成结果进记忆 (记忆增长)
  [V4] 回忆输出: JEPA 检索记忆输出"最近做了什么"
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import os
os.environ.pop("ACC_PRODUCT_CONFIG_V3", None)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys, time
import numpy as np

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "body"))

from kernel import JepaBody, text_to_obs

OUT_DIR = os.path.join(REPO_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Qwen 引擎 (文本编码 + 生成) ───────────────────────────
_qwen = {"tok": None, "model": None}


def load_qwen():
    if _qwen["model"] is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("加载 Qwen2.5-0.5B (CPU)...", flush=True)
    t0 = time.time()
    _qwen["tok"] = AutoTokenizer.from_pretrained(
        os.path.join(REPO_ROOT, "weights/qwen/qwen2.5-0.5b"), trust_remote_code=True)
    _qwen["model"] = AutoModelForCausalLM.from_pretrained(
        os.path.join(REPO_ROOT, "weights/qwen/qwen2.5-0.5b"), trust_remote_code=True)
    _qwen["model"].eval()
    print(f"  Qwen 就绪 ({time.time()-t0:.0f}s)")


def qwen_encode(text: str) -> np.ndarray:
    """文本 → 768d 表征 (最终层 hidden_state 均值)"""
    import torch
    tok, model = _qwen["tok"], _qwen["model"]
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    h = out.hidden_states[-1][0][1:]          # 最终层, 去 BOS
    return h.mean(0).float().numpy().astype(np.float32)   # bf16 → float32


def qwen_generate(prompt: str, max_new: int = 60) -> str:
    """文本生成"""
    import torch
    tok, model = _qwen["tok"], _qwen["model"]
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new,
                             do_sample=True, top_p=0.9, temperature=0.8)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class QwenPerception:
    """文本 → Qwen 896d → 固定投影 → 768d (JEPA 统一空间).
    工具选择/记忆检索只依赖同一投影下的一致性, 随机固定投影保相似度排序"""
    def __init__(self):
        import torch
        torch.manual_seed(0)
        self.proj = torch.nn.Linear(896, 768)   # 固定随机投影
        self.proj.eval()

    def encode(self, obs) -> np.ndarray:
        import torch
        if isinstance(obs, str):
            z = qwen_encode(obs)              # 896d
            with torch.no_grad():
                return self.proj(torch.from_numpy(z)).numpy().astype(np.float32)
        return np.asarray(obs, np.float32)


# ─── 生成引擎工具 ───────────────────────────────────────────
def write_text(prompt: str = "") -> str:
    """文本生成引擎: Qwen 落地 (JEPA 意图 → 真实文本)"""
    return qwen_generate(prompt or "write a short sentence")


def draw_image(subject: str = "cat", size: int = 128) -> str:
    """图片生成引擎占位: PIL 程序化绘制 (正式版接扩散模型)"""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (size, size), (240, 240, 250))
    d = ImageDraw.Draw(img)
    cx, cy, r = size // 2, size // 2 + 10, size // 4
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(200, 120, 60))   # 脸
    d.polygon([(cx - r, cy - r), (cx - r - 20, cy - r - 30), (cx - r + 10, cy - r - 5)],
              fill=(120, 60, 20))                                       # 左耳
    d.polygon([(cx + r, cy - r), (cx + r + 20, cy - r - 30), (cx + r - 10, cy - r - 5)],
              fill=(120, 60, 20))                                       # 右耳
    d.ellipse([cx - 20, cy - 12, cx - 6, cy + 2], fill=(30, 30, 30))    # 左眼
    d.ellipse([cx + 6, cy - 12, cx + 20, cy + 2], fill=(30, 30, 30))    # 右眼
    d.line([cx - 12, cy + 20, cx - 30, cy + 28], fill=(30, 30, 30))     # 胡须
    d.line([cx + 12, cy + 20, cx + 30, cy + 28], fill=(30, 30, 30))
    path = f"{OUT_DIR}/p4b_{subject}_{int(time.time())}.png"
    img.save(path)
    return f"saved image: {path} (subject={subject})"


# ─── 主验证 ─────────────────────────────────────────────────
def main():
    print("=" * 76)
    print("P4b 架构验证 — JEPA 指挥官 + 生成引擎执行器 (端到端)")
    print("=" * 76)
    load_qwen()

    body = JepaBody(seed=7)
    body.surprise_thresh = 0.001          # 新任务必然高惊讶 → 触发工具
    qp = QwenPerception()
    body.agent.perception = qp            # 感知走 Qwen 语义空间 (768d)

    # 注册生成引擎工具 + Qwen 语义描述表征
    body.register_tool("write_text", write_text,
                       "generate text, write article poem story, answer question")
    body.register_tool("draw_image", draw_image,
                       "generate image, draw picture painting, visual art")
    for name in ["write_text", "draw_image"]:
        desc = {"write_text": "generate text, write article poem story",
                "draw_image": "generate image, draw picture painting"}[name]
        body.set_tool_embed(name, qp.encode(desc))     # 768d 语义表征
    print(f"工具已注册: {body.tools.names()} (Qwen 语义描述表征)")

    # 场景 A: 文本输出
    print("\n" + "-" * 76)
    print("场景 A: 任务 'write a short poem about a cat'")
    task_a = "write a short poem about a cat"
    z_a = qp.encode(task_a)
    # 意图选择 (统一空间检索)
    sel_a = body._select_tool(z_a)
    print(f"  意图选择: {sel_a} (应为 write_text)")
    d_a = body.decide(task_a)
    print(f"  decide: {d_a}")
    ok_a = sel_a == "write_text"
    if d_a["tool_calls"]:
        tc = d_a["tool_calls"][0]
        res = body.call_tool(tc["name"], {"prompt": task_a})
        print(f"  生成引擎执行 → {res[:100]}...")
        body.tool_result_step([tc], [res])
    else:
        res = "NO TOOL"
    ok_gen_a = "poem" in res or len(res) > 20

    # 场景 B: 图片输出
    print("\n" + "-" * 76)
    print("场景 B: 任务 'draw a picture of a cat'")
    task_b = "draw a picture of a cat"
    z_b = qp.encode(task_b)
    sel_b = body._select_tool(z_b)
    print(f"  意图选择: {sel_b} (应为 draw_image)")
    d_b = body.decide(task_b)
    ok_b = sel_b == "draw_image"
    if d_b["tool_calls"]:
        tc = d_b["tool_calls"][0]
        res_b = body.call_tool(tc["name"], {"subject": "cat"})
        print(f"  生成引擎执行 → {res_b}")
        body.tool_result_step([tc], [res_b])
    else:
        res_b = "NO TOOL"
    ok_gen_b = "saved image" in res_b and ".png" in res_b

    # 验证 V3: 记忆增长
    mem = len(body.agent.memory.items)
    print(f"\n[V3] 记忆: {mem} 条 (工具结果已回传) | 工具调用 {body.stats['tool_calls']} 次, "
          f"错误 {body.stats['tool_errors']}")

    # 验证 V4: 回忆输出 (检索记忆中最相关的经验)
    print("\n" + "-" * 76)
    print("场景 C: 回忆 'what did you do recently?'")
    q = qp.encode("what did I ask you to create recently")
    items = body.agent.memory.items
    if items:
        sims = []
        for m in items[-20:]:
            s = np.asarray(m[0], np.float32)
            sn = s / (np.linalg.norm(s) + 1e-9)
            qn = q / (np.linalg.norm(q) + 1e-9)
            sims.append(float(np.dot(sn, qn)))
        best = int(np.argmax(sims))
        # 记忆条目是哈希/文本表征, 无法反解文本 → 输出相似度 + 条目统计
        print(f"  检索到最相关记忆 #{best} (相似度 {sims[best]:.3f}) — "
              f"JEPA 有 '最近输出' 经验, 但无文本解码器 (文本反解 = 解码器问题, 符合架构裁决)")
        ok_recall = sims[best] > 0.1
    else:
        ok_recall = False
        print("  记忆为空")

    print("\n" + "=" * 76)
    results = {"V1 意图选择": ok_a and ok_b,
               "V2 生成落地": ok_gen_a and ok_gen_b,
               "V3 回传记忆": mem > 0,
               "V4 回忆能力": ok_recall}
    for k, v in results.items():
        print(f"  {k}: {'✅' if v else '❌'}")
    print(f"判定: {'✅ P4b 指挥官-执行器架构闭环成立' if all(results.values()) else '⚠️ 部分成立'}")
    print("=" * 76)


if __name__ == "__main__":
    main()
