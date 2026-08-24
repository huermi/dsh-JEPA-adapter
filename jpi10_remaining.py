"""
JPI-10 剩余空白补齐 — 符号自动发现混合 + STM-Critic 稀疏奖励闭环
=================================================================
实验A (空白②): 符号自动发现混合方案
  对比: 手工行为指纹(6d) / 可学习编码器(6d) / 拼接(auto+manual 12d)
  / 正则化(auto 学 manual 目标)
  目标: 可学习编码器能否达到或超过纯手工 (100%)?

实验B (空白③b): STM-Critic 稀疏奖励闭环
  稀疏目标: 长运行机器(稀有) 价值=1, 其余=0 (正样本 ~0.4%)
  对比: Critic 仅即时训练 vs 记忆回放 (recent-N 缓冲)
  目标: 稀疏下记忆回放是否显著优于即时 (稠密下回放无增益已验证)
"""
import numpy as np
from collections import deque
import jpi6_behavioral_symbols as j6
import jpi7_gaps as j7  # 复用 LearnedEncoder, traj_fingerprint

STAGES = [2, 3, 4, 5]
BUDGET = 30


# ═══════════════════════════════════════════════════════════
# 实验A: 符号自动发现混合
# ═══════════════════════════════════════════════════════════
def build_pool_with_traj(n, seed, n_candidates=1000):
    """构建池 + 原始轨迹 (供可学习编码器)"""
    rng = np.random.RandomState(seed)
    rules_list = [j6.random_rules(n, rng) for _ in range(n_candidates)]
    steps_list = [j6.run_machine(r, n) for r in rules_list]
    slog = np.array([np.log1p(max(s, 0)) for s in steps_list], dtype=np.float32)
    halted = np.array([s >= 0 for s in steps_list])

    # 手工行为指纹
    F_manual = np.array([j6.extract_symbols(r, n, mode="behavior") for r in rules_list],
                        dtype=np.float32)
    for j in range(F_manual.shape[1]):
        lo, hi = F_manual[:, j].min(), F_manual[:, j].max()
        if hi > lo:
            F_manual[:, j] = (F_manual[:, j] - lo) / (hi - lo)
        else:
            F_manual[:, j] = 0.0
    # 原始轨迹
    trajs = np.array([j7.traj_fingerprint(r, n) for r in rules_list], dtype=np.float32)
    return F_manual, trajs, slog, halted


def train_auto_encoder(trajs, seed=7, iters=300):
    """可学习编码器: 从轨迹对自监督 (EMA target + L1)"""
    rng = np.random.RandomState(seed)
    enc = j7.LearnedEncoder(seed=seed)
    for _ in range(iters):
        i, j = rng.randint(len(trajs), size=2)
        enc.step(trajs[i], trajs[j])
    return enc


def eval_guided(F, slog, seed=0):
    """标准引导效率评估"""
    true_best = float(np.max(slog[slog > 0]))
    pred = j6.SymbolPredictor(s_dim=F.shape[1], seed=seed)
    return j6.guided_efficiency(pred, F, slog, budget=BUDGET, seed=seed)


def expA(n=4, seed=7):
    F_manual, trajs, slog, halted = build_pool_with_traj(n, seed)

    # 1. 纯手工
    eff_manual = eval_guided(F_manual, slog, seed)

    # 2. 纯自动 (可学习编码器)
    enc = train_auto_encoder(trajs, seed)
    F_auto = np.array([enc.encode(t) for t in trajs], dtype=np.float32)
    for j in range(F_auto.shape[1]):
        lo, hi = F_auto[:, j].min(), F_auto[:, j].max()
        if hi > lo:
            F_auto[:, j] = (F_auto[:, j] - lo) / (hi - lo)
        else:
            F_auto[:, j] = 0.0
    eff_auto = eval_guided(F_auto, slog, seed)

    # 3. 拼接 (auto + manual)
    F_cat = np.concatenate([F_auto, F_manual], axis=1)
    eff_cat = eval_guided(F_cat, slog, seed)

    # 4. 正则化编码器: 训练时加"预测 manual 指纹"辅助目标
    # 简化: 用 manual 指纹作为 soft 目标训练编码器 (有监督对齐)
    enc_reg = j7.LearnedEncoder(seed=seed)
    rng = np.random.RandomState(seed)
    for _ in range(300):
        i = rng.randint(len(trajs))
        # 目标 = 对应 manual 指纹 (让编码器学会"生成行为指纹")
        err = enc_reg.encode(trajs[i]) - F_manual[i]
        # 手动 SGD 一步
        dW2 = np.outer(np.tanh(np.dot(trajs[i], enc_reg.W1)), err)
        dH = np.dot(err, enc_reg.W2.T) * (1 - np.tanh(np.dot(trajs[i], enc_reg.W1))**2)
        dW1 = np.outer(trajs[i], dH)
        enc_reg.W2 -= 0.01 * np.clip(dW2, -0.5, 0.5)
        enc_reg.W1 -= 0.01 * np.clip(dW1, -0.5, 0.5)
        enc_reg.tW2 = enc_reg.ema * enc_reg.tW2 + (1 - enc_reg.ema) * enc_reg.W2
        enc_reg.tW1 = enc_reg.ema * enc_reg.tW1 + (1 - enc_reg.ema) * enc_reg.W1
    F_reg = np.array([enc_reg.encode(t) for t in trajs], dtype=np.float32)
    for j in range(F_reg.shape[1]):
        lo, hi = F_reg[:, j].min(), F_reg[:, j].max()
        if hi > lo:
            F_reg[:, j] = (F_reg[:, j] - lo) / (hi - lo)
        else:
            F_reg[:, j] = 0.0
    eff_reg = eval_guided(F_reg, slog, seed)

    print(f"  手工指纹(6d):       {eff_manual:5.1f}%")
    print(f"  自动编码器(6d):     {eff_auto:5.1f}%")
    print(f"  拼接(auto+manual):  {eff_cat:5.1f}%")
    print(f"  正则化(auto→manual):{eff_reg:5.1f}%")
    return eff_manual, eff_auto, eff_cat, eff_reg


# ═══════════════════════════════════════════════════════════
# 实验B: STM-Critic 稀疏奖励闭环
# ═══════════════════════════════════════════════════════════
def expB(n=5, seed=7, sparse_ratio=0.004):
    """稀疏奖励: 只有 top 0.4% 长运行机器价值=1 (模拟真实稀疏反馈)
    对比: Critic 仅即时训练 vs 记忆回放"""
    F, slog, halted = j6.build_pool(n, 100 + n * 13, "behavior")
    # 稀疏二分类目标: 运行步数 top 0.4% → 1, 其余 0
    thresh = np.percentile(slog[slog > 0], 100 * (1 - sparse_ratio))
    targets = (slog > thresh).astype(np.float32)

    n_pos = int(targets.sum())
    print(f"  池 {len(slog)} 台, 正样本(长运行) {n_pos} 台 ({n_pos/len(slog)*100:.1f}%)")

    # Critic 即时 vs 回放
    c_imm = j7.CriticReplay(seed=seed)
    c_rep = j7.CriticReplay(seed=seed)
    idx = np.random.RandomState(seed).permutation(len(F))

    imm_pos, rep_pos = 0, 0  # 训练中看到正样本数
    for i in idx:
        t = targets[i]
        c_imm.train_immediate(F[i], t)
        c_rep.train_replay(F[i], t)
        imm_pos += int(t)
        rep_pos += int(t)

    # 评估: 对池中所有机器预测, 检查正样本是否被识别 (预测排序)
    def eval_critic(c):
        preds = np.array([c.predict(f) for f in F])
        # 取 top-n_pos 个预测, 看命中多少正样本 (top-k precision)
        top_idx = np.argsort(preds)[-n_pos:]
        hit = int(targets[top_idx].sum())
        return hit / max(n_pos, 1) * 100

    p_imm = eval_critic(c_imm)
    p_rep = eval_critic(c_rep)
    print(f"  即时训练 top-{n_pos} 命中: {p_imm:.1f}%")
    print(f"  记忆回放 top-{n_pos} 命中: {p_rep:.1f}%")
    print(f"  回放改进: {p_rep - p_imm:+.1f}pp")
    return p_imm, p_rep


if __name__ == "__main__":
    print("=" * 72)
    print("JPI-10 剩余空白补齐")
    print("=" * 72)

    print("\n【实验A: 符号自动发现混合 (n=4, 3 seed)】")
    a_manual, a_auto, a_cat, a_reg = [], [], [], []
    for seed in [1, 7, 42]:
        m, au, c, r = expA(n=4, seed=seed)
        a_manual.append(m); a_auto.append(au); a_cat.append(c); a_reg.append(r)
    print(f"\n  汇总 (3 seed): 手工 {np.mean(a_manual):.1f}% | 自动 {np.mean(a_auto):.1f}% | "
          f"拼接 {np.mean(a_cat):.1f}% | 正则 {np.mean(a_reg):.1f}%")

    print("\n【实验B: STM-Critic 稀疏奖励闭环 (n=5, 3 seed)】")
    b_imm, b_rep = [], []
    for seed in [1, 7, 42]:
        i, r = expB(n=5, seed=seed)
        b_imm.append(i); b_rep.append(r)
    print(f"\n  汇总 (3 seed): 即时 {np.mean(b_imm):.1f}% | 回放 {np.mean(b_rep):.1f}% | "
          f"改进 {np.mean(b_rep)-np.mean(b_imm):+.1f}pp")
