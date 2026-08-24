"""
语义文本编码器 (body/mini_encoder.py)
=====================================
MiniLM (all-MiniLM-L6-v2) 语义编码 → 384d 归一化向量.

为什么换掉哈希袋:
  哈希袋 = 词频统计, 词序丢失, 无语义. "show me what python files exist
  then read the first one" 里 read 词频压过 list → 第一轮就跳步 (直接 read).
  MiniLM 句向量对语义敏感: "then"/"first" 有顺序权重, 教学中间情境可泛化到
  真实工具结果情境.

设计:
  - 惰性加载: 首次 encode 才加载模型 (kernel 启动不拖慢)
  - LRU 缓存: 教学+检索重复编码同一文本, 命中免推理 (CPU 毫秒级)
  - 失败回退: 模型缺失/加载失败 → encode 返回 None, 调用方哈希袋兜底
  - 结构化情境 s_ctx: task / last_call / last_result 三槽分段编码再拼接
    → 时序显式化: "上一步状态"是独立向量块, 检索时槽位相似直接决定命中.
    教学 (learn_call) 与运行时 (_context_z) 共用同一构造 → 空间一致.

模型位置: <REPO_ROOT>/models (HF_HOME 指向, 不占系统盘)
下载源:   hf-mirror.com (HF_ENDPOINT)
"""
from __future__ import annotations

import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import os
import threading
from typing import Optional

import numpy as np

# 模型缓存放仓库 models/ 目录 (默认; JEPA_MODEL_DIR 可覆盖)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = os.environ.get("JEPA_MODEL_DIR", os.path.join(REPO_ROOT, "models"))
os.environ.setdefault("HF_HOME", MODEL_DIR)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 默认强制离线: 模型已本地缓存 (<REPO_ROOT>/models). transformers/sentence_transformers
# import 时若联网检查 (网络不可达 → 超时重试) 会卡死数分钟 (2026-08-25 实测).
# 需要在线更新模型时显式设 HF_HUB_OFFLINE=0.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class MiniLmEncoder:
    """MiniLM 语义编码器 (惰性加载 + LRU 缓存 + 失败回退)"""

    def __init__(self, cache_size: int = 512):
        self._model = None
        self._tok = None
        self._lock = threading.Lock()
        self._cache: dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        self.dim = 384
        self.failed = False

    # ── 加载 ──────────────────────────────────────────────
    def ensure_loaded(self) -> bool:
        """加载模型 (首次调用). 失败置 failed=True, 返回 False."""
        if self._model is not None:
            return True
        if self.failed:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_DIR)
                self._tok = self._model.tokenizer
                return True
            except Exception as e:
                self.failed = True
                print(f"[mini_encoder] 模型加载失败, 回退哈希袋: "
                      f"{type(e).__name__}: {e}", flush=True)
                return False

    @property
    def available(self) -> bool:
        return self._model is not None

    # ── 编码 ──────────────────────────────────────────────
    def encode(self, text: str) -> Optional[np.ndarray]:
        """文本 → 384d 归一化向量. 模型不可用 → None (调用方兜底)."""
        if not text or not text.strip():
            return np.zeros(self.dim, dtype=np.float32)
        if not self.ensure_loaded():
            return None
        if text in self._cache:
            return self._cache[text]
        with self._lock:
            if text in self._cache:
                return self._cache[text]
            try:
                v = self._model.encode(text, normalize_embeddings=True,
                                       convert_to_numpy=True)
                v = np.asarray(v, np.float32)
            except Exception:
                return None
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[text] = v
            return v

    def encode_batch(self, texts: list[str]) -> list[Optional[np.ndarray]]:
        return [self.encode(t) for t in texts]

    # ── 结构化情境 (时序显式化核心) ───────────────────────
    def s_ctx(self, task: str = "", last_call: str = "",
              last_result: str = "") -> Optional[np.ndarray]:
        """三槽情境向量: concat(enc(task), enc(last_call), enc(last_result)).
        每槽独立归一化 → 槽位信息不被稀释; 时序 (上一步状态) 是独立维度块.
        模型不可用 → None (调用方用哈希袋字符串兜底)."""
        if not self.ensure_loaded():
            return None
        zt = self.encode(task)
        zc = self.encode(last_call)
        zr = self.encode(last_result)
        if zt is None:
            zt = np.zeros(self.dim, dtype=np.float32)
        if zc is None:
            zc = np.zeros(self.dim, dtype=np.float32)
        if zr is None:
            zr = np.zeros(self.dim, dtype=np.float32)
        v = np.concatenate([zt, zc, zr]).astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / (n + 1e-9)

    def s_ctx_dim(self) -> int:
        return self.dim * 3

    # ── 诊断 ──────────────────────────────────────────────
    def stats(self) -> dict:
        return {"available": self.available, "failed": self.failed,
                "cache": len(self._cache), "dim": self.dim}


# 模块级单例 (kernel/plugin 共享)
_inst: Optional[MiniLmEncoder] = None
_inst_lock = threading.Lock()


def get_encoder() -> MiniLmEncoder:
    global _inst
    if _inst is None:
        with _inst_lock:
            if _inst is None:
                _inst = MiniLmEncoder()
    return _inst


if __name__ == "__main__":
    # 自检: 下载模型 + 编码 + 情境向量
    enc = get_encoder()
    ok = enc.ensure_loaded()
    print(f"模型加载: {'OK' if ok else '失败'} | {enc.stats()}")
    if ok:
        import time
        t0 = time.time()
        v1 = enc.encode("show me what python files exist then read the first one")
        v2 = enc.encode("list the files in the directory")
        v3 = enc.encode("send an email to the boss")
        t1 = time.time()
        print(f"编码耗时(含首次加载): {t1-t0:.2f}s")
        print(f"list任务 vs read任务 cos: "
              f"{float(np.dot(v1, v2)):.3f}")
        print(f"list任务 vs email任务 cos: "
              f"{float(np.dot(v1, v3)):.3f}")
        ctx = enc.s_ctx("list the files", "glob: {\"pattern\": \"*.py\"}",
                        "found 12: jepa_base.py, ...")
        print(f"三槽情境向量: {ctx.shape} 范数 {float(np.linalg.norm(ctx)):.3f}")
