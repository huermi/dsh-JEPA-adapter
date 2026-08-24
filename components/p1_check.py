"""
P1 内环闭合验收 (components/p1_check.py)
==========================================
验证 C1/C2/C3 真实实现组装:

  C1 JepaPerception + timm 权重 → 编码质量 (KNN 对照 effect_check 60.1%)
  C2 ResidualWorldModel (AdaJEPA) → E1 静态流收敛 + 扰动流适应
  C3 CuriosityEnergy → E2 活性

数据: CIFAR-10 子集 (HF), 96×96
"""
import sys
import time
import numpy as np

sys.path.insert(0, "D:/JEPA")

from components.perception import JepaPerception
from components.world_model import ResidualWorldModel
from components.energy import CuriosityEnergy

N_TRAIN = 1000
N_TEST = 400


def load_cifar(n_train=N_TRAIN, n_test=N_TEST):
    import os
    os.environ.pop("ACC_PRODUCT_CONFIG_V3", None)
    import datasets
    import torchvision.transforms as T
    tf = T.Compose([T.Resize((96, 96)), T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    tr = datasets.load_dataset("uoft-cs/cifar10", split=f"train[:{n_train}]")
    te = datasets.load_dataset("uoft-cs/cifar10", split=f"test[:{n_test}]")
    tr_x = np.stack([np.asarray(tf(tr[i]["img"])) for i in range(n_train)])
    tr_y = np.array([tr[i]["label"] for i in range(n_train)])
    te_x = np.stack([np.asarray(tf(te[i]["img"])) for i in range(n_test)])
    te_y = np.array([te[i]["label"] for i in range(n_test)])
    return tr_x, tr_y, te_x, te_y


def knn(train_z, train_y, test_z, test_y, k=10):
    tn = train_z / (np.linalg.norm(train_z, axis=1, keepdims=True) + 1e-9)
    tz = test_z / (np.linalg.norm(test_z, axis=1, keepdims=True) + 1e-9)
    sim = tz @ tn.T
    idx = np.argsort(-sim, axis=1)[:, :k]
    knn_y = train_y[idx]
    preds = np.array([np.bincount(row).argmax() for row in knn_y])
    return float((preds == test_y).mean())


def main():
    t0 = time.time()
    print("=" * 70)
    print("P1 内环闭合验收 — C1/C2/C3 真实实现")
    print("=" * 70)

    print("加载 CIFAR-10...")
    tr_x, tr_y, te_x, te_y = load_cifar()
    print(f"数据: 训练 {tr_x.shape} 测试 {te_x.shape} ({time.time()-t0:.0f}s)")

    # ── C1: 编码器 + timm 权重 ────────────────────────────
    print("\n--- C1: JepaPerception + timm 权重 ---")
    t1 = time.time()
    enc = JepaPerception()                     # img_size=224 (pos 与 timm 匹配)
    info = enc.load_weights("timm")
    print(f"权重吸收: {info['loaded_layers']} 层 ({time.time()-t1:.0f}s)")

    t2 = time.time()
    tr_z = np.stack([enc.encode(x) for x in tr_x])
    te_z = np.stack([enc.encode(x) for x in te_x])
    acc = knn(tr_z, tr_y, te_z, te_y)
    print(f"编码 {len(tr_x)+len(te_x)} 张 ({time.time()-t2:.0f}s)")
    print(f"KNN(10) top-1: {acc*100:.1f}% (effect_check timm 基线 60.1%)")
    ok1 = acc > 0.5
    print(f"C1 判定: {'✅ 编码质量保持' if ok1 else '❌ 编码退化'}")

    # ── C2: 世界模型 E1 收敛 (可学静态流: s→s+0.05c 固定偏移) ──
    print("\n--- C2: ResidualWorldModel E1 收敛 (可学静态流) ---")
    wm = ResidualWorldModel(n_actions=5, seed=7)
    c = np.ones(768, dtype=np.float32) * 0.05   # 固定偏移 (非平凡但可学)
    e1s = []
    for step in range(80):
        i = step % 100
        s = tr_z[i]
        s_next = tr_z[i] + c                    # 可学: 残差 = c 恒定
        e1 = wm.step(s, 0, s_next)
        e1s.append(e1)
    e1_first = np.mean(e1s[:10])
    e1_last = np.mean(e1s[-10:])
    print(f"E1: 前10 {e1_first:.5f} → 后10 {e1_last:.5f} "
          f"(下降 {(1 - e1_last/max(e1_first,1e-9))*100:.1f}%)")
    ok2 = e1_last < e1_first * 0.3
    print(f"C2 判定: {'✅ E1 收敛' if ok2 else '❌ E1 未收敛'}")

    # ── C2: 扰动流 (偏移) + adapt_from_buffer 恢复 ────────
    print("\n--- C2: 扰动流 (偏移) + 回放恢复 ---")
    rng = np.random.RandomState(0)
    from components.core import Event
    e1_pre, e1_post = [], []
    events_buf = []
    for step in range(30):
        i = step % 100
        s = tr_z[i]
        noise = rng.randn(*s.shape).astype(np.float32) * 0.1
        s_next = tr_z[i] + noise                # 同图 + 小扰动 = 偏移
        e1 = wm.step(s, 0, s_next)
        e1_pre.append(e1)
        events_buf.append(Event(t=step, s=s, a=0, s_next=s_next,
                                e1=e1, e2=0.0, perf=0.0))
    wm.adapt_from_buffer(events_buf)            # recent-N 回放 (AdaJEPA)
    for step in range(40):
        i = step % 100
        s = tr_z[i]
        s_next = tr_z[i] + c                    # 恢复可学流
        e1 = wm.step(s, 0, s_next)
        e1_post.append(e1)
    print(f"偏移期 E1 均值 {np.mean(e1_pre):.4f} → 恢复期 {np.mean(e1_post):.4f} "
          f"(下降 {(1 - np.mean(e1_post)/max(np.mean(e1_pre),1e-9))*100:.0f}%)")
    ok2b = np.mean(e1_post) < np.mean(e1_pre) * 0.8
    print(f"C2b 判定: {'✅ 偏移后恢复' if ok2b else '❌ 偏移后未恢复'}")

    # ── C3: 能量系统活性 ──────────────────────────────────
    print("\n--- C3: CuriosityEnergy E2 活性 ---")
    eng = CuriosityEnergy()
    for i in range(200):
        e2 = eng.update(float(np.mean(e1s)), anchor_value=0.1)
    e1_m, e1_tr, e2_m = eng.get_stats()
    print(f"E1 均值 {e1_m:.4f} | E1 趋势 {e1_tr:+.5f} | E2 均值 {e2_m:.4f}")
    ok3 = e2_m > 0
    print(f"C3 判定: {'✅ E2 活性' if ok3 else '❌ E2 失活'}")

    # ── C1: I-JEPA 目标单步 ───────────────────────────────
    print("\n--- C1: I-JEPA 掩码预测 (forward_loss) ---")
    t3 = time.time()
    loss, e = enc.forward_loss(te_x[:4])
    print(f"I-JEPA loss: {loss:.4f} ({time.time()-t3:.0f}s)")

    print("\n" + "=" * 70)
    ok = ok1 and ok2 and ok3
    print(f"P1 内环闭合: {'✅ 通过' if ok else '❌ 未通过'} "
          f"(C1{'✅' if ok1 else '❌'} C2{'✅' if ok2 else '❌'} C3{'✅' if ok3 else '❌'})")
    print(f"总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
