"""
效果评估 (effect_check.py)
============================
CIFAR-10 表征质量对比: 随机 / timm-ViT / Sparsh-JEPA / Qwen迁移

评估方法:
  1. KNN 分类 (10-NN, 余弦距离)
  2. 线性探测 (归一化特征 + Linear + 交叉熵)

配置: resize 96×96 (CPU 友好), 训练子集 3000 / 测试 800
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

from jepa_base import build_vit_base
from weight_loaders import load_timm, load_sparsh

IMG_SIZE = 96
TRAIN_N = 3000
TEST_N = 800
BATCH = 64


def load_data():
    """CIFAR-10 via HF datasets (走 hf-mirror, 绕过 toronto 慢源/过期证书)"""
    import os
    # 规避 WorkBuddy 注入的超长环境变量 (datasets 配置传播 bug)
    os.environ.pop("ACC_PRODUCT_CONFIG_V3", None)
    os.environ.pop("ACC_PRODUCT_CONFIG", None)
    import datasets
    import torchvision.transforms as T
    from torch.utils.data import TensorDataset, DataLoader

    tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    print("从 HF 下载 CIFAR-10...")
    tr = datasets.load_dataset("uoft-cs/cifar10", split="train")
    te = datasets.load_dataset("uoft-cs/cifar10", split="test")
    torch.manual_seed(0)
    tr_idx = torch.randperm(len(tr))[:TRAIN_N].tolist()
    te_idx = torch.randperm(len(te))[:TEST_N].tolist()

    tr_x = torch.stack([tf(tr[i]["img"]) for i in tr_idx])
    tr_y = torch.tensor([tr[i]["label"] for i in tr_idx])
    te_x = torch.stack([tf(te[i]["img"]) for i in te_idx])
    te_y = torch.tensor([te[i]["label"] for i in te_idx])
    print(f"数据就绪: 训练 {tr_x.shape} 测试 {te_x.shape}")

    train_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(TensorDataset(te_x, te_y), batch_size=BATCH, shuffle=False)
    return train_loader, test_loader


def make_encoder(init):
    """四种初始化编码器"""
    enc = build_vit_base()
    if init == "random":
        pass
    elif init == "timm":
        load_timm(enc)
    elif init == "sparsh":
        load_sparsh(enc, "weights/sparsh/ijepa_vitbase.safetensors")
    elif init == "qwen":
        from qwen_to_jepa import load_qwen_encoder
        load_qwen_encoder(enc)
    enc.eval()
    return enc


@torch.no_grad()
def extract(enc, loader):
    """提取 CLS 特征 [N, 768] + 标签"""
    feats, labels = [], []
    for x, y in loader:
        z = enc(x)
        feats.append(z)
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def knn_eval(train_z, train_y, test_z, test_y, k=10):
    """10-NN 余弦 top-1"""
    tz = F.normalize(test_z, dim=1)
    trz = F.normalize(train_z, dim=1)
    sim = tz @ trz.T  # [T, N]
    _, idx = sim.topk(k, dim=1)
    knn_labels = train_y[idx]  # [T, k]
    preds = torch.mode(knn_labels, dim=1).values
    return (preds == test_y).float().mean().item()


def linear_probe(train_z, train_y, test_z, test_y, steps=300, lr=1e-2):
    """线性探测: 归一化特征 + Linear (无 bias, 类原型式)"""
    n_cls = 10
    train_z = F.normalize(train_z, dim=1)
    test_z = F.normalize(test_z, dim=1)
    head = nn.Linear(train_z.shape[1], n_cls, bias=False)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    n = len(train_z)
    for step in range(steps):
        idx = torch.randperm(n)[:512]
        logits = head(train_z[idx])
        loss = F.cross_entropy(logits, train_y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = head(test_z).argmax(1)
    return (preds == test_y).float().mean().item()


def main():
    print("=" * 70)
    print("JEPA 底座效果评估 — CIFAR-10 (resize 96, 训练 3000 / 测试 800)")
    print("=" * 70)
    t0 = time.time()
    train_loader, test_loader = load_data()
    print(f"数据加载完成 ({time.time()-t0:.0f}s)")

    results = []
    for init in ["random", "timm", "sparsh", "qwen"]:
        t1 = time.time()
        enc = make_encoder(init)
        print(f"\n--- 初始化: {init} (编码器构建 {time.time()-t1:.0f}s) ---")
        t2 = time.time()
        train_z, train_y = extract(enc, train_loader)
        test_z, test_y = extract(enc, test_loader)
        print(f"特征提取完成: {train_z.shape} ({time.time()-t2:.0f}s)")
        acc_knn = knn_eval(train_z, train_y, test_z, test_y)
        acc_lin = linear_probe(train_z, train_y, test_z, test_y)
        print(f"KNN(10) top-1: {acc_knn*100:.1f}% | 线性探测 top-1: {acc_lin*100:.1f}%")
        results.append((init, acc_knn, acc_lin))

    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    print(f"{'初始化':<10} | {'KNN top-1':>10} | {'线性探测':>10}")
    print("-" * 40)
    for init, k, l in results:
        print(f"{init:<10} | {k*100:9.1f}% | {l*100:9.1f}%")
    print(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
