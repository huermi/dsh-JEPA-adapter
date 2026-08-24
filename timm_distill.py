"""
timm 起点 + Qwen 蒸馏 (timm_distill.py)
=========================================
最后一个对照: 已有视觉结构的编码器 (timm ViT-B/16, 74.3% 线性探测)
+ Qwen 语义原型蒸馏 → LLM 语义注入是增强还是干扰?

对比 effect_check 的 timm 基线: KNN 60.1% / 线性探测 74.3%
"""
import torch
import torch.nn.functional as F
import time

from jepa_base import build_vit_base
from weight_loaders import load_timm
from effect_check import load_data, extract, knn_eval, linear_probe
from qwen_to_jepa import get_qwen_prototypes, ProtoProjector, QWEN_TEXT_PROMPTS
from qwen_distill import train_prototype_distill

EPOCHS = 5


def main():
    t0 = time.time()
    print("=" * 70)
    print("timm 起点 + Qwen 蒸馏 — LLM 语义注入对已有视觉表征的影响")
    print("=" * 70)
    train_loader, test_loader = load_data()

    # Qwen 类原型
    proto_896 = F.normalize(get_qwen_prototypes(QWEN_TEXT_PROMPTS), dim=-1)
    print(f"Qwen 类原型: {proto_896.shape}")

    # timm 起点编码器
    torch.manual_seed(42)
    enc = build_vit_base()
    load_timm(enc)
    proj = ProtoProjector()

    print("\n--- timm 起点 + Qwen 原型蒸馏 ---")
    t1 = time.time()
    train_prototype_distill(enc, proto_896, train_loader,
                            epochs=EPOCHS, projector=proj)
    print(f"蒸馏完成 ({time.time()-t1:.0f}s)")
    enc.eval()

    tr_z, tr_y = extract(enc, train_loader)
    te_z, te_y = extract(enc, test_loader)
    k = knn_eval(tr_z, tr_y, te_z, te_y)
    l = linear_probe(tr_z, tr_y, te_z, te_y)
    print(f"timm+蒸馏: KNN {k*100:.1f}% | 线性探测 {l*100:.1f}%")

    print("\n" + "=" * 70)
    print("对比 (effect_check 基线):")
    print(f"  timm 基线      : KNN 60.1% | 线性探测 74.3%")
    print(f"  timm+Qwen蒸馏  : KNN {k*100:.1f}% | 线性探测 {l*100:.1f}%")
    print(f"  差值: KNN {k*100-60.1:+.1f}pp | 线性 {l*100-74.3:+.1f}pp")
    print(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
