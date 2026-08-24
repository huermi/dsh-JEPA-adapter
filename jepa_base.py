"""
JEPA 底座 — 权重吸收版 (jepa_base.py)
======================================
架构 (对齐 I-JEPA ViT-B/16, 命名对齐 timm 以便加载外部权重):
  - 编码器: ViT-B/16 (patch_embed + cls_token + pos_embed + 12 层 transformer)
  - 预测器: I-JEPA 风格轻量 transformer (4 层), 预测 masked patch 的潜表征
  - EMA target encoder: 防坍缩 (I-JEPA 标准)
  - 能量: ||z_target - z_pred||^2 (潜空间 L2)

用途 (真实身体第一步):
  1. 加载公开权重 (timm ViT / Sparsh-IJEPA / Qwen 迁移) → 表征提取
  2. 线性探测 / KNN 评估表征质量
  3. 作为后续 "LLM→JEPA 权重编码" 实验的底座
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def trunc_normal_(tensor, mean=0.0, std=0.02):
    with torch.no_grad():
        return tensor.normal_(mean, std)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim=768, num_heads=12, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """图像 → patch token 嵌入 (I-JEPA 风格, 无 cls 无 pos)"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # B, N, D
        return x


class VisionEncoder(nn.Module):
    """I-JEPA 风格 ViT 编码器 (对齐 timm vit_base_patch16_224 命名)"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop)
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, drop, attn_drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_patches=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # 位置编码: 若 patch 数不同则 2D 插值 (支持任意分辨率)
        if x.shape[1] != self.pos_embed.shape[1]:
            pos = self.pos_embed[:, 1:]                    # 1, P, D
            grid = int(round(pos.shape[1] ** 0.5))          # 原 grid (如 14)
            pos = pos.reshape(1, grid, grid, -1).permute(0, 3, 1, 2)
            new_p = x.shape[1] - 1
            new_grid = int(round(new_p ** 0.5))
            pos = F.interpolate(pos, size=(new_grid, new_grid),
                                mode="bilinear", align_corners=False)
            pos = pos.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
            pe = torch.cat([self.pos_embed[:, :1], pos], dim=1)
        else:
            pe = self.pos_embed
        x = self.pos_drop(x + pe)
        x = self.blocks(x)
        x = self.norm(x)
        if return_patches:
            return x  # B, 1+N, D (含 cls)
        return x[:, 0]  # B, D (CLS)


class Predictor(nn.Module):
    """I-JEPA 风格预测器: 从 context 表征预测 masked patch 表征"""
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        return self.norm(self.blocks(x))


class JEPA(nn.Module):
    """JEPA 底座: 编码器 + 预测器 + EMA target + 能量
    mode='infer': 表征提取 (冻结 EMA 编码器)
    mode='pretrain': 掩码潜表征预测训练 (I-JEPA 目标)
    """
    def __init__(self, embed_dim=768, depth=12, num_heads=12,
                 predictor_depth=4, img_size=224, patch_size=16, in_chans=3,
                 momentum=0.999, mask_ratio=0.6):
        super().__init__()
        self.embed_dim = embed_dim
        self.momentum = momentum
        self.mask_ratio = mask_ratio
        self.encoder = VisionEncoder(img_size, patch_size, in_chans, embed_dim, depth, num_heads)
        self.predictor = Predictor(embed_dim, predictor_depth, num_heads)
        # EMA target encoder
        self.target_encoder = VisionEncoder(img_size, patch_size, in_chans, embed_dim, depth, num_heads)
        self.target_encoder.requires_grad_(False)
        self._init_target_weights()
        self.num_patches = self.encoder.patch_embed.num_patches

    def _init_target_weights(self):
        for p_t, p_s in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            p_t.data.copy_(p_s.data)

    @torch.no_grad()
    def _momentum_update(self):
        for p_t, p_s in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            p_t.data.mul_(self.momentum).add_(p_s.data, alpha=1.0 - self.momentum)

    # ── 推理模式: 表征提取 ────────────────────────────────
    def encode(self, x):
        """提取 CLS 表征 (冻结编码器)"""
        with torch.no_grad():
            return self.encoder(x)

    def encode_patches(self, x):
        with torch.no_grad():
            return self.encoder(x, return_patches=True)

    # ── 训练模式: I-JEPA 掩码潜表征预测 ───────────────────
    def energy(self, x, mask_ratio=None):
        """I-JEPA 训练: 掩码部分 patch, 用 context 预测 masked 表征.
        返回 (loss, 能量)"""
        B = x.shape[0]
        N = self.num_patches
        device = x.device
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        n_masked = max(1, int(N * mask_ratio))

        # 每行固定数量掩码 (randperm)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for i in range(B):
            idx = torch.randperm(N, device=device)[:n_masked]
            mask[i, idx] = True

        # context 表征: 用 CLS + 全部 patch 过编码器, 取 CLS 作为全局 context
        # (简化: 完整 I-JEPA 用 context-only 编码 + 可学习 mask token, 此处先近似)
        z_ctx = self.encoder(x)  # B, D

        # target 表征 (EMA 编码器, 全 patch, 取被掩码的)
        with torch.no_grad():
            z_target_all = self.target_encoder(x, return_patches=True)[:, 1:]  # B, N, D
            z_target = z_target_all[mask].view(B, n_masked, self.embed_dim)

        # 预测 masked 表征 (用 context 表征扩展 + 可学习位置偏置简化)
        pred_tokens = self.predictor.blocks(z_ctx.unsqueeze(1).expand(B, n_masked, self.embed_dim))
        z_pred = self.predictor.norm(pred_tokens)

        # 能量 = 潜空间 L2
        loss = F.mse_loss(z_pred, z_target)
        self._momentum_update()
        return loss, loss.item()


def build_vit_base():
    """ViT-B/16 (标准配置)"""
    return VisionEncoder(embed_dim=768, depth=12, num_heads=12)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    enc = build_vit_base()
    print(f"ViT-B/16 编码器参数: {count_params(enc)/1e6:.1f}M")
    x = torch.randn(2, 3, 224, 224)
    z = enc(x)
    print(f"CLS 表征: {z.shape}")

    jepa = JEPA()
    print(f"JEPA 总参数: {count_params(jepa)/1e6:.1f}M")
    loss, e = jepa.energy(x)
    print(f"I-JEPA 能量(首步): {e:.4f}")
