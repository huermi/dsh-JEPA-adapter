"""
Qwen2.5-0.5B 权重 → JEPA 编码器 (qwen_to_jepa.py)
==================================================
两条路径:

  路径1 load_qwen_encoder (朴素直接编码):
    Qwen embed_tokens 矩阵 SVD 主成分 → 初始化 JEPA cls_token/pos_embed 分布
    预期: 模态鸿沟 → 效果≈随机 (作为"直接权重拷贝"的诚实基线)

  路径2 distill_qwen_prototypes (语义原型蒸馏, 主实验):
    Qwen 冻结编码类别文本 → 类原型 (896d) → 投影 768d
    训练 JEPA 编码器 (冻结前 10 层, 训最后 2 层) 使图像表征对齐类原型
    (InfoNCE) → LLM 的语义知识"编码"进 JEPA 可用的权重
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

QWEN_PATH = "weights/qwen/qwen2.5-0.5b"
QWEN_TEXT_PROMPTS = [
    "a photo of an airplane", "a photo of an automobile",
    "a photo of a bird", "a photo of a cat",
    "a photo of a deer", "a photo of a dog",
    "a photo of a frog", "a photo of a horse",
    "a photo of a ship", "a photo of a truck",
]


def load_qwen_embeddings():
    """读取 Qwen embed_tokens 权重 [151936, 896] (不加载整个模型, 省内存)"""
    import os
    for f in os.listdir(QWEN_PATH):
        if f.endswith(".safetensors"):
            sd = load_file(os.path.join(QWEN_PATH, f))
            for k in ("model.embed_tokens.weight", "embed_tokens.weight"):
                if k in sd:
                    return sd[k]
    raise FileNotFoundError("Qwen embed_tokens 未找到")


# ─── 路径1: 朴素直接编码 ──────────────────────────────────
def load_qwen_encoder(enc: nn.Module, rank: int = 768):
    """Qwen embed_tokens 的 SVD 主成分 → 初始化 JEPA cls_token/pos_embed.
    语义: LLM 词嵌入的主成分构成"语义基", 用它初始化位置/类别 token,
    让 JEPA 的 token 从"语义丰富的分布"起步.
    返回统计信息 (诚实报告: 这是弱连接, 预期≈随机)."""
    emb = load_qwen_embeddings().float()          # [151936, 896]
    # 中心化 + SVD
    emb_c = emb - emb.mean(0, keepdim=True)
    U, S, V = torch.linalg.svd(emb_c, full_matrices=False)
    # 前 rank 主成分: V[:rank] 是 896→rank 的正交基
    semantic_base = V[:rank]                       # [768, 896]
    # token 在语义基上的坐标
    coords = emb_c @ semantic_base.T               # [151936, 768]
    # 用坐标分布初始化 pos_embed / cls_token
    enc.pos_embed.data.normal_(coords.mean().item(), coords.std().item())
    enc.cls_token.data.copy_(coords.mean(0, keepdim=True).unsqueeze(0))
    enc.pos_embed.data.mul_(0.1)                   # 保守缩放
    return {
        "method": "svd_semantic_base",
        "embed_tokens": tuple(emb.shape),
        "explained_var_top768": float((S[:rank].sum() / S.sum())),
        "note": "LLM 词嵌入主成分初始化 token, 模态鸿沟下预期≈随机",
    }


# ─── 路径2: 语义原型蒸馏 (主实验) ─────────────────────────
def get_qwen_prototypes(prompts=None):
    """Qwen 冻结编码文本 → 类原型 [n_cls, 896]"""
    from transformers import AutoModel, AutoTokenizer
    prompts = prompts or QWEN_TEXT_PROMPTS
    tok = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(QWEN_PATH, trust_remote_code=True)
    model.eval()
    protos = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt")
            out = model(**ids)
            h = out.last_hidden_state[0].float()     # [L, 896] 转 float32
            # 均值池化 (去掉首 token 的 BOS)
            protos.append(h[1:].mean(0))
    return torch.stack(protos)                     # [10, 896]


class ProtoProjector(nn.Module):
    """768 (JEPA) → 896 (Qwen) 投影: 图像表征映射到 Qwen 语义空间"""
    def __init__(self, in_dim=768, out_dim=896):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


def distill(enc: nn.Module, train_loader, epochs=3, lr=3e-4, n_cls=10,
            finetune_layers=2):
    """原型对齐蒸馏: 图像表征 vs Qwen 类原型 (InfoNCE).
    冻结编码器前 depth-finetune_layers 层, 训练最后 2 层 + 投影头.
    返回 (proto_768, 投影头)"""
    # Qwen 原型
    print("编码 Qwen 类原型...")
    proto_896 = get_qwen_prototypes()
    projector = ProtoProjector()
    with torch.no_grad():
        proto_768 = F.normalize(projector(proto_896), dim=-1)
    print(f"类原型: {proto_768.shape}")

    # 冻结前 10 层
    blocks = enc.blocks
    for i, blk in enumerate(blocks):
        if i < len(blocks) - finetune_layers:
            for p in blk.parameters():
                p.requires_grad_(False)
    params = [p for p in blocks.parameters() if p.requires_grad] + \
             list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    print("开始原型对齐蒸馏...")
    total = 0
    for epoch in range(epochs):
        for x, y in train_loader:
            z = enc(x)                              # [B, 768]
            z = F.normalize(z, dim=-1)
            p = proto_768[y]                        # [B, 768]
            # InfoNCE: 正样本相似度 vs 全类
            logits = (z * p).sum(-1)                # [B]
            # 与所有原型相似度做 softmax
            logits_all = z @ proto_768.T            # [B, 10]
            loss = F.cross_entropy(logits_all, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += 1
            if total % 50 == 0:
                print(f"  epoch{epoch} step{total}: loss {loss.item():.4f}")
    # 解除冻结
    for p in enc.parameters():
        p.requires_grad_(True)
    return proto_768, projector


if __name__ == "__main__":
    # 快速测试: 朴素直接编码 + 原型获取
    torch.manual_seed(0)
    from jepa_base import build_vit_base
    enc = build_vit_base()
    r = load_qwen_encoder(enc)
    print("朴素直接编码:", {k: v if not isinstance(v, tuple) else v for k, v in r.items()})
    p = get_qwen_prototypes(["a photo of a cat", "a photo of a dog"])
    print(f"Qwen 类原型: {p.shape}")
