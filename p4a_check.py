"""
P4a 原始输入补全验证 (p4a_check.py)
====================================
验证 MultimodalPerception 全模态输入 → 统一 768d:

  [A1] 音频: 3 类 (440Hz/880Hz/噪声) × 2 变体 → 同类相似 > 异类 (可分性)
  [A2] 视频: 静态 vs 运动帧序列 → 表征可分 + 768d
  [A3] 统一接口: 文本/图片/音频/视频 全模态 → 768d 形状统一
  [A4] 回归: 图片编码 (timm) 仍工作
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import os
os.environ.pop("ACC_PRODUCT_CONFIG_V3", None)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")   # 卡点修复: 顶部即离线

import sys
import numpy as np

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "body"))

from components.perception import JepaPerception
from multimodal import MultimodalPerception

SR = 16000
DUR = 2.0


def gen_tone(freq: float, phase: float = 0.0) -> np.ndarray:
    t = np.linspace(0, DUR, int(SR * DUR), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def gen_noise(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return (0.5 * rng.randn(int(SR * DUR))).astype(np.float32)


def gen_video(motion: bool, seed: int = 0, n_frames: int = 8) -> list:
    """简单矩形视频: 静态(中心) vs 运动(左→右)"""
    frames = []
    rng = np.random.RandomState(seed)
    size = 224
    for i in range(n_frames):
        img = np.full((size, size, 3), 0.6, np.float32)
        if motion:
            x = int(30 + i * (size - 90) / (n_frames - 1))
        else:
            x = (size - 60) // 2
        img[x:x + 60, x:x + 60] = [0.2, 0.6, 0.8]
        img += rng.randn(*img.shape) * 0.01          # 轻微噪声
        frames.append(img.transpose(2, 0, 1))        # [C,H,W]
    return frames


def cosine(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    print("=" * 76)
    print("P4a 原始输入补全验证 — 全模态统一 768d 表征")
    print("=" * 76)

    print("\n加载图片编码器 (timm ViT-B/16)...")
    image_enc = JepaPerception(img_size=224)
    info = image_enc.load_weights("timm")
    print(f"  权重吸收: {info['loaded_layers']} 层")
    mm = MultimodalPerception(image_enc)             # 文本走哈希兜底

    # ── [A1] 音频可分性 ────────────────────────────────
    print("\n[A1] 音频编码 (3 类 × 2 变体)...")
    classes = {"tone440": [gen_tone(440, 0), gen_tone(440, 1.0)],
               "tone880": [gen_tone(880, 0), gen_tone(880, 0.5)],
               "noise": [gen_noise(0), gen_noise(7)]}
    z_audio = {k: [mm.encode({"type": "audio", "waveform": w,
                              "sampling_rate": SR}) for w in v]
               for k, v in classes.items()}
    same, diff = [], []
    for k, zs in z_audio.items():
        same.append(cosine(zs[0], zs[1]))
        for k2, zs2 in z_audio.items():
            if k2 != k:
                diff.append(cosine(zs[0], zs2[0]))
    sep = float(np.mean(same) - np.mean(diff))
    print(f"  同类相似 {np.mean(same):.3f} | 异类相似 {np.mean(diff):.3f} | "
          f"间隔 {sep:.3f}")
    ok_a1 = sep > 0.05 and all(z.shape == (768,) for zs in z_audio.values() for z in zs)
    print(f"  {'✅ 音频可分 (不同声音→不同表征)' if ok_a1 else '❌ 音频不可分'}")

    # ── [A2] 视频可分性 ────────────────────────────────
    print("\n[A2] 视频编码 (静态 vs 运动, 各 8 帧)...")
    v_static = mm.encode(gen_video(False))
    v_motion = mm.encode(gen_video(True))
    sim = cosine(v_static, v_motion)
    print(f"  静态 vs 运动 表征余弦: {sim:.3f} (应 < 0.95)")
    ok_a2 = sim < 0.95 and v_static.shape == (768,)
    print(f"  {'✅ 视频可分 + 768d' if ok_a2 else '❌ 视频不可分'}")

    # ── [A3] 统一接口 ──────────────────────────────────
    print("\n[A3] 全模态统一 768d...")
    obs_list = [
        ("text", "hello world this is a test"),
        ("image", np.random.randn(3, 224, 224).astype(np.float32)),
        ("audio", gen_tone(440)),
        ("video", gen_video(False)),
    ]
    shapes = {}
    for name, obs in obs_list:
        z = mm.encode(obs)
        shapes[name] = z.shape
    print(f"  各模态形状: {shapes}")
    ok_a3 = all(s == (768,) for s in shapes.values())
    print(f"  {'✅ 全模态统一 768d' if ok_a3 else '❌ 维度不统一'}")

    # ── [A4] 回归: 图片 KNN 语义 (复用真实 CIFAR) ──────
    print("\n[A4] 回归: 图片编码语义保持 (真实 CIFAR-10)...")
    import datasets
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    te = datasets.load_dataset("uoft-cs/cifar10", split="test[:20]")
    tr = datasets.load_dataset("uoft-cs/cifar10", split="train[:100]")
    MEAN = np.array([0.4914, 0.4822, 0.4465], np.float32)
    STD = np.array([0.2470, 0.2435, 0.2616], np.float32)
    prep = lambda img: ((np.array(img.resize((224, 224))).astype(np.float32)
                         / 255.0 - MEAN) / STD).transpose(2, 0, 1)
    tr_z = np.stack([mm.encode(prep(img)) for img in tr["img"]])
    tr_y = list(tr["label"])
    correct = 0
    for img, y in zip(te["img"][:10], list(te["label"])[:10]):
        z = mm.encode(prep(img))
        sims = tr_z @ (z / (np.linalg.norm(z) + 1e-9))
        pred = tr_y[int(np.argmax(sims))]
        if pred == y:
            correct += 1
    acc = correct / 10
    print(f"  图片→类别 KNN top-1: {acc*100:.0f}% (10 张抽样, 随机 10%)")
    ok_a4 = acc > 0.3
    print(f"  {'✅ 图片语义保持' if ok_a4 else '❌ 图片语义退化'}")

    print("\n" + "=" * 76)
    results = {"A1 音频可分": ok_a1, "A2 视频可分": ok_a2,
               "A3 全模态 768d": ok_a3, "A4 图片回归": ok_a4}
    for k, v in results.items():
        print(f"  {k}: {'✅' if v else '❌'}")
    print(f"判定: {'✅ P4a 原始输入补全完成' if all(results.values()) else '⚠️ 部分成立'}")
    print("=" * 76)


if __name__ == "__main__":
    main()
