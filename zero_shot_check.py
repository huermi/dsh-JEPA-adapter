"""
Zero-shot 收尾验证 (zero_shot_check.py)
=========================================
验证 "LLM 语义真正可用": 蒸馏后的 JEPA 编码器, 图像表征
经投影头映射到 Qwen 语义空间, 与 Qwen 文本原型直接匹配 (zero-shot 分类)

  Z1: timm + Qwen 蒸馏 (完整链路: encD → projD → Qwen空间 → 文本原型匹配)
  Z2: timm 无蒸馏 + 随机投影头 (对照: 无对齐 → 应≈随机 10%)
  Z3: 随机编码器 + 随机投影头 (最坏基线)

评估: 图像表征与 10 个 Qwen 类原型 (proto_896) 的余弦相似度 top-1 匹配率
"""
import torch
import torch.nn.functional as F
import time

from jepa_base import build_vit_base
from weight_loaders import load_timm
from effect_check import load_data
from qwen_to_jepa import get_qwen_prototypes, ProtoProjector, QWEN_TEXT_PROMPTS
from qwen_distill import train_prototype_distill

EPOCHS = 5


@torch.no_grad()
def zero_shot(enc, proj, proto, loader):
    """图像 → 编码 → 投影到 Qwen 空间 → 与文本原型匹配 top-1"""
    correct, total = 0, 0
    for x, y in loader:
        z = enc(x)
        z = proj(z)                              # [B, 896] → Qwen 空间
        z = F.normalize(z, dim=-1)
        logits = z @ proto.T                      # [B, 10] 余弦
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += len(y)
    return correct / total


def main():
    t0 = time.time()
    print("=" * 72)
    print("Zero-shot 收尾验证 — LLM 语义是否真正可用")
    print("=" * 72)
    train_loader, test_loader = load_data()

    # Qwen 文本原型
    proto_896 = F.normalize(get_qwen_prototypes(QWEN_TEXT_PROMPTS), dim=-1)
    print(f"Qwen 文本原型: {proto_896.shape}")

    # ── Z1: timm + Qwen 蒸馏 ──────────────────────────────
    torch.manual_seed(42)
    enc1 = build_vit_base()
    load_timm(enc1)
    proj1 = ProtoProjector()
    print("\n--- Z1: timm + Qwen 蒸馏 (5 epoch) ---")
    t1 = time.time()
    train_prototype_distill(enc1, proto_896, train_loader,
                            epochs=EPOCHS, projector=proj1)
    print(f"蒸馏完成 ({time.time()-t1:.0f}s)")
    enc1.eval()
    z1_train = zero_shot(enc1, proj1, proto_896, train_loader)
    z1_test = zero_shot(enc1, proj1, proto_896, test_loader)
    print(f"Z1 zero-shot: 训练 {z1_train*100:.1f}% | 测试 {z1_test*100:.1f}%")

    # ── Z2: timm 无蒸馏 + 随机投影 ────────────────────────
    torch.manual_seed(42)
    enc2 = build_vit_base()
    load_timm(enc2)
    proj2 = ProtoProjector()
    enc2.eval()
    z2_test = zero_shot(enc2, proj2, proto_896, test_loader)
    print(f"Z2 (timm+随机投影) zero-shot 测试: {z2_test*100:.1f}%")

    # ── Z3: 随机编码器 + 随机投影 ─────────────────────────
    torch.manual_seed(42)
    enc3 = build_vit_base()
    proj3 = ProtoProjector()
    enc3.eval()
    z3_test = zero_shot(enc3, proj3, proto_896, test_loader)
    print(f"Z3 (随机+随机投影) zero-shot 测试: {z3_test*100:.1f}%")

    print("\n" + "=" * 72)
    print("Zero-shot 汇总 (测试集)")
    print("=" * 72)
    print(f"  Z1 timm+蒸馏   : {z1_test*100:.1f}% (LLM 语义可用性)")
    print(f"  Z2 timm+随机投影: {z2_test*100:.1f}% (无对齐对照)")
    print(f"  Z3 随机+随机投影: {z3_test*100:.1f}% (最坏基线)")
    print(f"  Z1 - Z2 增益   : {z1_test*100 - z2_test*100:+.1f}pp (蒸馏对齐的贡献)")
    print(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
