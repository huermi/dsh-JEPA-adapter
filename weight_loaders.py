"""
权重加载器 (weight_loaders.py)
===============================
三路公开权重 → JEPA 底座 VisionEncoder 的统一加载:

  A. timm ViT-B/16 (ImageNet 预训练, 通用视觉基线)
  B. Sparsh-IJEPA (Meta 官方 I-JEPA 训练 ViT-B, 官方 JEPA 权重)
  C. Qwen2.5-0.5B (LLM, 迁移实验用, 占位)

所有加载器输出统一的 (encoder, source_name, notes)
"""
import torch
import torch.nn as nn
from safetensors.torch import load_file


def load_sparsh(encoder: nn.Module, path: str, in_chans: int = 3):
    """Sparsh I-JEPA 权重 → VisionEncoder.
    Sparsh 是 6 通道输入 (触觉图对), patch_embed.proj 为 6 通道 Conv.
    策略: 加载全部 transformer 层 (blocks/norm/cls/pos), patch_embed 的
    6 通道权重取前 3 通道切片 (若 in_chans=3). 返回加载统计."""
    sd = load_file(path)
    # 规范化 key: 去掉 'encoder.' 前缀 (Sparsh 官方格式)
    clean = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith('encoder.'):
            k2 = k2[len('encoder.'):]
        clean[k2] = v

    model_sd = encoder.state_dict()
    loaded, skipped = {}, []
    for k, v in clean.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            loaded[k] = v
        elif k == 'patch_embed.proj.weight' and v.shape[1] == 6 and in_chans == 3:
            # 6 通道 → 前 3 通道
            loaded[k] = v[:, :3].contiguous()
        elif k in model_sd:
            skipped.append(f"{k}: {tuple(v.shape)}→{tuple(model_sd[k].shape)}")
        else:
            skipped.append(f"{k}: 不在模型中")
    # 用 loaded 覆盖 (只更新匹配层)
    for k, v in loaded.items():
        model_sd[k].copy_(v.to(model_sd[k].dtype))
    n_total = sum(1 for k in clean if k in model_sd or k.startswith('patch_embed'))
    return {
        "loaded_layers": len(loaded),
        "skipped": skipped[:5],
        "n_total": n_total,
    }


def load_timm(encoder: nn.Module, model_name: str = "vit_base_patch16_224.augreg_in21k_ft_in1k"):
    """timm ViT 预训练 → VisionEncoder (命名天然对齐)"""
    import timm
    src = timm.create_model(model_name, pretrained=True)
    src_sd = src.state_dict()
    model_sd = encoder.state_dict()
    loaded, skipped = {}, []
    for k, v in src_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            loaded[k] = v
        elif k in model_sd:
            skipped.append(f"{k}: {tuple(v.shape)}→{tuple(model_sd[k].shape)}")
    for k, v in loaded.items():
        model_sd[k].copy_(v)
    return {"loaded_layers": len(loaded), "skipped": skipped[:5], "total_params": len(src_sd)}


def summarize(encoder: nn.Module, name: str):
    """打印编码器权重来源概览 (随机 vs 已加载)"""
    from collections import Counter
    print(f"=== {name} ===")
    # 简单统计: 打印前 3 层参数来源无法直接判断, 用均值漂移近似
    # (加载过的层与随机初始化均值/std 差异大)
    return name


if __name__ == "__main__":
    from jepa_base import build_vit_base

    # 测试 timm 加载
    enc = build_vit_base()
    r = load_timm(enc)
    print(f"timm 加载: {r['loaded_layers']} 层成功, 跳过 {len(r['skipped'])}")
    if r['skipped']:
        print("  跳过示例:", r['skipped'][:3])

    # 测试 Sparsh 加载 (如果权重已下载)
    import os
    sp_path = "weights/sparsh/ijepa_vitbase.safetensors"
    if os.path.exists(sp_path):
        enc2 = build_vit_base()
        r2 = load_sparsh(enc2, sp_path)
        print(f"Sparsh 加载: {r2['loaded_layers']} 层成功, 跳过 {len(r2['skipped'])}")
        if r2['skipped']:
            print("  跳过示例:", r2['skipped'][:3])
    else:
        print("Sparsh 权重未下载, 跳过")
