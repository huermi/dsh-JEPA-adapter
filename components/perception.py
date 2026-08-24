"""
C1 PerceptionEncoder — 感知编码器 (内环)
=========================================
契约: 原始观测 (图像/屏幕/文件) → 768d JEPA 表征.

接口:
  encode(obs:[B,C,H,W] | [C,H,W]) → z:[B,D] | z:[D]
  encode_patches(obs) → z_all:[B,1+N,D]   (含 cls, 供 I-JEPA 掩码预测)
  load_weights(source)                     (吸收公开权重)

权重吸收来源:
  - timm ViT-B/16 ImageNet   (weight_loaders.load_timm, 150 层)
  - Sparsh-IJEPA 官方         (weight_loaders.load_sparsh, 148 层)
  - Qwen 语义蒸馏             (qwen_to_jepa 路径2, 投影头参与)

验收标准 (effect_check.py):
  - 随机基线: 线性探测 ~32.7%
  - timm: ~74.3% | timm+Qwen蒸馏: ~80.0% | zero-shot 66.0%
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .core import D, Obs, Embed


class PerceptionEncoder(ABC):
    """C1 感知编码器接口"""

    @abstractmethod
    def encode(self, obs: Obs) -> Embed:
        """观测 → 768d 表征. obs 可为 [C,H,W] 或 [B,C,H,W] (后者返回 [B,D])."""

    @abstractmethod
    def encode_patches(self, obs: Obs) -> Embed:
        """观测 → [B, 1+N, D] patch 表征 (含 cls, I-JEPA 训练用)."""

    @abstractmethod
    def load_weights(self, source: str) -> dict:
        """吸收公开权重. source: 'timm' | 'sparsh' | 权重文件路径.
        返回加载统计 (loaded_layers / skipped)."""

    @abstractmethod
    def forward_loss(self, x: Obs, mask_ratio: float = 0.6) -> tuple[float, float]:
        """I-JEPA 掩码潜表征预测: 返回 (loss, energy). (训练模式)"""


class JepaPerception(PerceptionEncoder):
    """真实实现: jepa_base.VisionEncoder + 权重吸收 (timm/Sparsh)

    验收: effect_check.py (CIFAR-10 表征质量: timm 74.3% / +Qwen蒸馏 80.0%)
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_chans: int = 3, embed_dim: int = D,
                 depth: int = 12, num_heads: int = 12, seed: int = 0):
        """img_size=224 默认: pos_embed (1,197) 与 timm 权重匹配;
        实际输入任意分辨率 (96/224) 由 forward 的 pos 插值处理."""
        from jepa_base import VisionEncoder
        self.enc = VisionEncoder(img_size=img_size, patch_size=patch_size,
                                 in_chans=in_chans, embed_dim=embed_dim,
                                 depth=depth, num_heads=num_heads)
        self.img_size = img_size
        self.load_info: dict = {}
        self._mask_head = None

    def encode(self, obs: Obs) -> Embed:
        import numpy as np
        import torch
        x = np.asarray(obs, dtype=np.float32)
        batched = x.ndim == 4
        if not batched:
            x = x[None]                      # [1,C,H,W]
        with torch.no_grad():
            z = self.enc(torch.from_numpy(x))
        z = z.numpy()
        return z if batched else z[0]

    def encode_patches(self, obs: Obs) -> Embed:
        import numpy as np
        import torch
        x = np.asarray(obs, dtype=np.float32)
        batched = x.ndim == 4
        if not batched:
            x = x[None]
        with torch.no_grad():
            z = self.enc(torch.from_numpy(x), return_patches=True)
        z = z.numpy()
        return z if batched else z[0]

    def load_weights(self, source: str) -> dict:
        from weight_loaders import load_timm, load_sparsh
        if source == "timm":
            self.load_info = load_timm(self.enc)
        elif source == "sparsh":
            self.load_info = load_sparsh(self.enc, "weights/sparsh/ijepa_vitbase.safetensors")
        elif source.startswith("sparsh:"):
            self.load_info = load_sparsh(self.enc, source.split(":", 1)[1])
        else:
            raise ValueError(f"未知权重源: {source} (支持 timm / sparsh)")
        return self.load_info

    def forward_loss(self, x: Obs, mask_ratio: float = 0.6) -> tuple[float, float]:
        """I-JEPA 掩码潜表征预测 (单步, 训练模式)"""
        import numpy as np
        import torch
        t = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if t.ndim == 3:
            t = t[None]
        self.enc.train()
        # 简化 I-JEPA 目标: 掩码 patch, 用编码器输出预测 (完整 JEPA 目标见 jepa_base.JEPA)
        B, C, H, W = t.shape
        n_patches = (H // 16) * (W // 16)
        n_mask = max(1, int(n_patches * mask_ratio))
        mask = torch.zeros(B, n_patches, dtype=torch.bool)
        for i in range(B):
            idx = torch.randperm(n_patches)[:n_mask]
            mask[i, idx] = True
        z_all = self.enc(t, return_patches=True)            # [B, 1+N, D]
        z_patches = z_all[:, 1:]
        # 用 [CLS] 表征预测被掩码 patch (线性头近似, 缓存避免重初始化)
        if self._mask_head is None:
            self._mask_head = torch.nn.Linear(D, D)
        cls = z_all[:, 0]
        pred = self._mask_head(cls).unsqueeze(1).expand(B, n_patches, D)
        loss = torch.nn.functional.mse_loss(pred[mask], z_patches[mask])
        self.enc.eval()
        return float(loss.item()), float(loss.item())


class DummyPerception(PerceptionEncoder):
    """占位实现 (组装自检用): 随机固定投影, lazy 初始化适配任意输入维度"""

    def __init__(self, d_in: int = 3 * 96 * 96, seed: int = 0):
        import numpy as np
        self.rng = np.random.RandomState(seed)
        self.d_in = d_in
        self.W: np.ndarray | None = None
        self._patches = 36

    def _ensure_w(self, d_in: int) -> None:
        import numpy as np
        if self.W is None or self.W.shape[0] != d_in:
            self.W = self.rng.randn(d_in, D).astype(np.float32) / (d_in ** 0.5)

    def encode(self, obs: Obs) -> Embed:
        import numpy as np
        x = np.asarray(obs, dtype=np.float32)
        if x.ndim == 3:                       # [C,H,W] 单图像
            flat = x.reshape(1, -1)
            self._ensure_w(flat.shape[-1])
            return np.tanh(flat @ self.W)[0]
        if x.ndim == 1:                       # [D] 单向量观测
            self._ensure_w(x.shape[-1])
            return np.tanh(x @ self.W)
        flat = x.reshape(x.shape[0], -1)      # [B, ...] batch
        self._ensure_w(flat.shape[-1])
        return np.tanh(flat @ self.W)

    def encode_patches(self, obs: Obs) -> Embed:
        import numpy as np
        z = self.encode(obs)
        if z.ndim == 1:
            return np.tile(z, (1, self._patches + 1, 1))
        return np.tile(z[:, None, :], (1, self._patches + 1, 1))

    def load_weights(self, source: str) -> dict:
        return {"loaded_layers": 0, "skipped": ["dummy"], "note": "占位实现, 无权重"}

    def forward_loss(self, x: Obs, mask_ratio: float = 0.6) -> tuple[float, float]:
        return 1.0, 1.0
