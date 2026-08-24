"""
Qwen 语义蒸馏主实验 (qwen_distill.py)
=======================================
验证 "LLM 权重 → JEPA 可用权重" 的蒸馏路径:

  条件 A: 随机编码器 + Qwen 类原型 InfoNCE 蒸馏 (LLM 知识注入)
  条件 B: 随机编码器 + one-hot 标签线性分类训练 (常规弱监督上界)
  条件 C: 随机编码器 (基线, 无训练)

评估: 蒸馏/训练后冻结编码器 → KNN + 线性探测 (CIFAR-10)
结论: A vs C = LLM 知识净增益; A vs B = 文本原型 vs 直接标签的信息量
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys

from jepa_base import build_vit_base
from effect_check import load_data, extract, knn_eval, linear_probe
from qwen_to_jepa import get_qwen_prototypes, ProtoProjector, QWEN_TEXT_PROMPTS

EPOCHS = 5
LR = 3e-4
FINETUNE_LAYERS = 4


def freeze_all_but_last(enc, finetune_layers):
    blocks = enc.blocks
    for i, blk in enumerate(blocks):
        if i < len(blocks) - finetune_layers:
            for p in blk.parameters():
                p.requires_grad_(False)
    for p in enc.norm.parameters():
        p.requires_grad_(True)


def train_prototype_distill(enc, proto_768, train_loader, epochs=EPOCHS, projector=None):
    """Qwen 原型 InfoNCE 蒸馏 (v2: 投影头可训练 + 更多层 + 更久)"""
    freeze_all_but_last(enc, FINETUNE_LAYERS)
    params = [p for p in enc.parameters() if p.requires_grad]
    if projector is not None:
        params += list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    total = 0
    for epoch in range(epochs):
        for x, y in train_loader:
            z = enc(x)
            if projector is not None:
                z = projector(z)                  # 图像表征映射到 Qwen 空间
            z = F.normalize(z, dim=-1)
            logits_all = z @ proto_768.T          # [B, 10] 与 Qwen 类原型相似度
            loss = F.cross_entropy(logits_all, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += 1
            if total % 40 == 0:
                acc = (logits_all.argmax(1) == y).float().mean().item()
                print(f"  distill step {total}: loss {loss.item():.4f} acc {acc:.2f}")
    for p in enc.parameters():
        p.requires_grad_(True)
    return enc


def train_label_head(enc, train_loader, epochs=EPOCHS):
    """one-hot 标签线性分类训练 (上界对照, 与蒸馏同计算预算)"""
    freeze_all_but_last(enc, FINETUNE_LAYERS)
    head = nn.Linear(768, 10)
    opt = torch.optim.AdamW(
        [p for p in enc.parameters() if p.requires_grad] + list(head.parameters()),
        lr=LR, weight_decay=1e-4)
    total = 0
    for epoch in range(epochs):
        for x, y in train_loader:
            z = enc(x)
            logits = head(z)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += 1
            if total % 40 == 0:
                acc = (logits.argmax(1) == y).float().mean().item()
                print(f"  label step {total}: loss {loss.item():.4f} acc {acc:.2f}")
    for p in enc.parameters():
        p.requires_grad_(True)
    return enc


def main():
    t0 = time.time()
    print("=" * 70)
    print("Qwen 语义蒸馏实验 — LLM 知识编码进 JEPA 编码器")
    print("=" * 70)
    train_loader, test_loader = load_data()
    print(f"数据就绪 ({time.time()-t0:.0f}s)")

    # Qwen 类原型 (896 原生空间)
    print("获取 Qwen 类原型 (文本编码)...")
    proto_896 = get_qwen_prototypes(QWEN_TEXT_PROMPTS)
    proto_896 = F.normalize(proto_896, dim=-1)
    print(f"Qwen 类原型: {proto_896.shape}")

    # 条件 A: 随机 + Qwen 蒸馏 (v2: 图像 768→896 映射到 Qwen 空间, proj 可训练)
    torch.manual_seed(42)
    encA = build_vit_base()
    projA = ProtoProjector()
    print("\n--- 条件 A: 随机初始化 + Qwen 原型蒸馏 (v2) ---")
    t1 = time.time()
    train_prototype_distill(encA, proto_896, train_loader, projector=projA)
    print(f"蒸馏完成 ({time.time()-t1:.0f}s)")
    encA.eval()
    trA, yA = extract(encA, train_loader)
    teA, yTe = extract(encA, test_loader)
    kA = knn_eval(trA, yA, teA, yTe)
    lA = linear_probe(trA, yA, teA, yTe)
    print(f"条件A: KNN {kA*100:.1f}% | 线性探测 {lA*100:.1f}%")

    # 条件 B: 随机 + one-hot 标签训练
    torch.manual_seed(42)
    encB = build_vit_base()
    print("\n--- 条件 B: 随机初始化 + one-hot 标签训练 ---")
    t2 = time.time()
    train_label_head(encB, train_loader)
    print(f"训练完成 ({time.time()-t2:.0f}s)")
    encB.eval()
    trB, _ = extract(encB, train_loader)
    teB, _ = extract(encB, test_loader)
    kB = knn_eval(trB, yA, teB, yTe)
    lB = linear_probe(trB, yA, teB, yTe)
    print(f"条件B: KNN {kB*100:.1f}% | 线性探测 {lB*100:.1f}%")

    # 条件 C: 随机基线 (effect_check 已测, 快速重测)
    torch.manual_seed(42)
    encC = build_vit_base()
    encC.eval()
    trC, _ = extract(encC, train_loader)
    teC, _ = extract(encC, test_loader)
    kC = knn_eval(trC, yA, teC, yTe)
    lC = linear_probe(trC, yA, teC, yTe)
    print(f"条件C: KNN {kC*100:.1f}% | 线性探测 {lC*100:.1f}%")

    print("\n" + "=" * 70)
    print("蒸馏实验汇总")
    print("=" * 70)
    print(f"{'条件':<34} | {'KNN':>6} | {'线性探测':>8}")
    print("-" * 56)
    print(f"{'A 随机+Qwen蒸馏':<34} | {kA*100:5.1f}% | {lA*100:7.1f}%")
    print(f"{'B 随机+标签训练':<34} | {kB*100:5.1f}% | {lB*100:7.1f}%")
    print(f"{'C 随机基线':<34} | {kC*100:5.1f}% | {lC*100:7.1f}%")
    print(f"\nA vs C (LLM 知识净增益): KNN {kA*100-kC*100:+.1f}pp | 线性 {lA*100-lC*100:+.1f}pp")
    print(f"A vs B (文本原型 vs 直接标签): KNN {kA*100-kB*100:+.1f}pp | 线性 {lA*100-lB*100:+.1f}pp")
    print(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
