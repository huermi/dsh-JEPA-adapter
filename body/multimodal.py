"""
多模态统一感知 (body/multimodal.py) — P4a 原始输入补全
=======================================================
让 JepaBody 能接收全部原始输入模态 → 统一 768d 表征:

  image: JepaPerception (timm ViT-B/16, 90.5% KNN 已验证)
  text : QwenPerception (Qwen 896d → 投影 768, P4b 已验证) 或哈希兜底
  video: 帧级 timm + 时序均值池化 (正式版换 V-JEPA 2)
  audio: numpy 频谱统计特征 → 投影 768 (正式版换 AST/CLAP,
         MIT/ast-finetuned-audioset-10-10 权重已定位, 镜像暂不可达)

接口: MultimodalPerception.encode(obs) → 768d
  obs 形态: str=文本 | np.ndarray [C,H,W]=图片 | list[np.ndarray]=视频帧 |
            dict {type:'audio', waveform:1D, sampling_rate:int} = 音频
"""
from __future__ import annotations

import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.core import D


# ─── 音频编码器 (程序化频谱特征, 正式版换 AST) ─────────────
class ASTEncoder:
    """真实音频编码器: AST (Audio Spectrogram Transformer, 86M).
    输入 waveform → AST hidden_state 均值 → 768d (AST hidden=768).
    权重: MIT/ast-finetuned-audioset-10-10-0.4593 (镜像有, 本地 weights/ast).
    加载失败 → 调用方应 fallback 到 AudioEncoder."""

    def __init__(self, local_dir: str = os.path.join(REPO_ROOT, "weights/ast")):
        self.local_dir = local_dir
        self._fe = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
        self._fe = AutoFeatureExtractor.from_pretrained(self.local_dir)
        self._model = AutoModel.from_pretrained(self.local_dir)
        self._model.eval()

    def encode(self, obs) -> np.ndarray:
        import torch
        self._load()
        if isinstance(obs, dict):
            wav = obs.get("waveform", obs.get("array"))
            sr = obs.get("sampling_rate", 16000)
        else:
            wav, sr = obs, 16000
        inputs = self._fe(np.asarray(wav), sampling_rate=sr,
                         return_tensors="pt", max_length=16000,
                         truncation=True, padding=True)
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True)
        z = out.hidden_states[-1].mean(dim=1)[0].float().numpy()   # 768d
        n = np.linalg.norm(z)
        return (z / (n + 1e-9) * np.sqrt(D)).astype(np.float32)


class AudioEncoder:
    """音频 → 768d: 分帧 FFT → 频谱统计 → 固定投影.
    特征: 总能量 / 频率质心 / 带宽 / 平坦度 / Mel-bin 均值分布.
    正式版: AST (MIT/ast-finetuned-audioset-10-10) 或 CLAP"""

    def __init__(self, sr: int = 16000, seed: int = 0):
        self.sr = sr
        rng = np.random.RandomState(seed)
        self.proj = rng.randn(68, D).astype(np.float32) / (68 ** 0.5)

    def _features(self, wav: np.ndarray) -> np.ndarray:
        wav = np.asarray(wav, np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        n_fft, hop = 512, 256
        if len(wav) < n_fft:
            wav = np.pad(wav, (0, n_fft - len(wav)))
        frames = []
        for i in range(0, len(wav) - n_fft + 1, hop):
            frames.append(wav[i:i + n_fft])
        if not frames:
            frames = [wav[:n_fft]]
        win = np.hanning(n_fft).astype(np.float32)
        specs = []
        for f in frames:
            spec = np.abs(np.fft.rfft(f * win)) ** 2
            specs.append(spec)
        spec = np.stack(specs)                       # [T, 257]
        spec = spec / (spec.sum(1, keepdims=True) + 1e-9)
        # Mel 近似: 32 个三角滤波器组
        mel = np.zeros((spec.shape[0], 32), np.float32)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sr)
        mel_pts = np.linspace(freqs[1], freqs[-1], 34)
        for m in range(32):
            lo, c, hi = mel_pts[m], mel_pts[m + 1], mel_pts[m + 2]
            w = np.clip(np.minimum((freqs - lo) / (c - lo + 1e-9),
                                   (hi - freqs) / (hi - c + 1e-9)), 0, 1)
            mel[:, m] = spec @ w
        mel = np.log1p(mel * 1e3)
        # 统计特征 (24d)
        energy = float(mel.mean())
        centroid = float(np.sum(mel.mean(0) * np.arange(32)) /
                         (mel.mean(0).sum() + 1e-9))
        flatness = float(np.exp(np.mean(np.log(mel + 1e-9), axis=0)).mean())
        return np.concatenate([
            [energy, centroid, flatness, float(mel.std())],
            mel.mean(0), mel.std(0)], dtype=np.float32)

    def encode(self, obs) -> np.ndarray:
        if isinstance(obs, dict):
            wav = obs.get("waveform", obs.get("array"))
            sr = obs.get("sampling_rate", obs.get("sampling_rate", self.sr))
            self.sr = sr
        else:
            wav = obs
        f = self._features(wav)
        z = f @ self.proj
        n = np.linalg.norm(z)
        return (z / (n + 1e-9) * np.sqrt(D)).astype(np.float32)


# ─── 视频编码器 (帧级 timm + 时序统计, 正式版换 V-JEPA 2) ─
class VideoEncoder:
    """视频 → 768d: 帧级图片编码 → (均值, 标准差) 时序统计 → 投影.
    修正: 纯均值池化把运动信息稀释 (静态/运动余弦 0.964 不可分);
    加帧间 std 维度捕捉"运动"(运动视频帧间变化大)."""

    def __init__(self, image_enc, seed: int = 0):
        self.image_enc = image_enc
        rng = np.random.RandomState(seed)
        self.proj = rng.randn(1536, D).astype(np.float32) / (1536 ** 0.5)

    def encode(self, frames) -> np.ndarray:
        zs = np.stack([self.image_enc.encode(f) for f in frames])
        feat = np.concatenate([zs.mean(axis=0), zs.std(axis=0)])   # 均值+动态
        z = feat @ self.proj
        n = np.linalg.norm(z)
        return (z / (n + 1e-9) * np.sqrt(D)).astype(np.float32)


# ─── 统一多模态感知 ─────────────────────────────────────────
class MultimodalPerception:
    """统一入口: 任意原始输入 → 768d (路由到各模态编码器)"""

    def __init__(self, image_enc, text_enc=None, audio_enc=None, seed: int = 0):
        self.image = image_enc
        self.text = text_enc                      # 可空: 空则哈希兜底
        # 音频: 优先 AST (真实权重), 不可用则程序化特征
        if audio_enc is not None:
            self.audio = audio_enc
        else:
            try:
                ast = ASTEncoder()
                ast._load()
                self.audio = ast
            except Exception:
                self.audio = AudioEncoder(seed=seed)
        self.video = VideoEncoder(image_enc)
        self._hash_rng = np.random.RandomState(1)

    @property
    def audio_backend(self) -> str:
        return "AST" if isinstance(self.audio, ASTEncoder) else "spectral"

    def encode(self, obs) -> np.ndarray:
        # 文本
        if isinstance(obs, str):
            if self.text is not None:
                return np.asarray(self.text.encode(obs), np.float32)
            return self._hash(obs)
        # 显式类型路由 (dict)
        if isinstance(obs, dict):
            t = obs.get("type")
            if t == "audio":
                return self.audio.encode(obs)
            if t == "video":
                return self.video.encode(obs["frames"])
            if t == "image":
                return self.image.encode(obs["array"])
            if t == "text":
                s = obs.get("text", "")
                return self.encode(s)
        # 隐式类型: 3D [C,H,W] = 图片; list = 视频帧
        if isinstance(obs, list):
            return self.video.encode(obs)
        arr = np.asarray(obs)
        if arr.ndim == 3:
            return self.image.encode(arr)
        if arr.ndim == 1:                          # 裸音频波形
            return self.audio.encode(arr)
        raise ValueError(f"无法识别的观测类型: {type(obs)} shape={arr.shape}")

    def _hash(self, text: str) -> np.ndarray:
        import hashlib
        vec = np.zeros(D, dtype=np.float32)
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % (2 ** 31)
            rng = np.random.RandomState(h)
            vec += rng.randn(D).astype(np.float32) * 0.01
        n = np.linalg.norm(vec)
        return (vec / (n + 1e-9)) if n > 1e-9 else vec
