"""
JPI-9 Configurator 漂移场景实验 — 异稳态设定点调制
====================================================
⚠️ 2026-08-24 修正声明: 本实验的 build_drift_pool 使用"行为伪影注入"
(手拍特征 F=1.0/0.8), 已被 bb_human_check.py 证明失真 — 伪影池上
"Configurator 检测到漂移但增益 0" 的结论部分失真 (真实结构池上
ε-greedy 根本无效, 正解是环内容分析). 本文件仅保留 Configurator
机制演示价值, 数值结论不作数; 真实结构池结论见
BB人类式反汇编实验与错误结论反思.md.

场景: BB 环境 n=2→5 逐级递增 = 天然漂移环境 (规则空间 6000→17万亿,
最优"引导预算比例"随 n 漂移)。

Configurator = 系统参数设定点的异稳态调节器:
  - 监测信号: 每阶段引导效率 (guided_best / true_best)
  - 漂移检测: 效率 < 历史均值 × 阈值 → 阶段分布变了 (n 变大的信号)
  - 调制动作: 漂移 → 上调引导预算比例 guided_frac (设定点异稳态上调);
            效率稳定 → 下调 (回到基准)
  - 生物对应: 发烧 = 感染时身体主动上调体温设定点;
            这里 = n 变大时系统主动上调"引导依赖"设定点

对照:
  A. 固定参数: guided_frac=0.5 跑完所有 n (现状)
  B. Configurator: 每阶段检测漂移, 调制 guided_frac

预期: n=4→5 随机采样效率骤降处, Configurator 应检测到并上调引导,
      最终效率高于固定参数 (尤其 n=5)。
"""
import numpy as np
from collections import deque
import jpi6_behavioral_symbols as j6

STAGES = [2, 3, 4, 5]
BUDGET = 30
N_CANDIDATES = 2000


def make_counter_machine(n):
    """人工构造"计数器"机器: 线性扩张磁带, 运行极长 (模拟 BB 长运行候选机)
    机制: 状态0 看到0→写1右移; 看到1→写1右移+切状态; 状态1 交替...
    这类机器的符号特征: 磁带线性扩张 + 状态循环 → 符号引导可识别, 随机极难碰"""
    rules = np.zeros((n * 2, 3), dtype=np.int32)
    for st in range(n):
        # 看到 0: 写 1, 右移, 切到 (st+1)%n
        rules[st * 2 + 0] = [1, 1, (st + 1) % n]
        # 看到 1: 写 1, 左移, 切到 (st+2)%n (制造振荡)
        rules[st * 2 + 1] = [1, 0, (st + 2) % n]
    # 最后一个状态的 1 分支停机 (保证会停)
    rules[(n - 1) * 2 + 1] = [1, 0, -1]
    return rules


def build_drift_pool(n, seed, n_candidates=N_CANDIDATES, inject_long=False):
    """漂移池: n=5 时注入 8 台稀有长运行机器 (log 步数 ~6, 即 ~400 步 vs 池中 ~1-2)
    模拟真实 BB 长尾: 随机采样几乎碰不到; 符号引导(线性扩张)可识别"""
    F, slog, halted = j6.build_pool(n, seed, "behavior")
    if inject_long and n == 5:
        rng = np.random.RandomState(seed)
        for k in range(8):
            idx = rng.randint(len(slog))
            # 注入"计数器机器"的行为指纹 (线性扩张, 长运行)
            F[idx] = np.array([1.0, 1.0, 0.0, 0.8, 1.0, 0.8], dtype=np.float32)
            slog[idx] = np.log1p(100000)  # ~400 步 (远大于池中均值)
    return F, slog, halted


class Configurator:
    """异稳态设定点调制器 (Configurator 外环的简化实现)
    监测信号: 随机采样部分的效率 (rand_best/true_best, 百分数)
    —— n 变大时规则空间爆炸, 随机采样效率骤降 (漂移信号)
    调制动作: 随机效率骤降 → 上调引导预算比例 (设定点异稳态上调)"""
    def __init__(self, init_frac=0.5, rand_drift_thresh=25.0, up_step=0.2,
                 down_step=0.1, lo=0.3, hi=0.9, window=3):
        self.frac = init_frac          # 设定点: 引导预算比例
        self.rand_drift_thresh = rand_drift_thresh  # 百分数阈值
        self.up_step = up_step
        self.down_step = down_step
        self.lo, self.hi = lo, hi
        self.rand_eff_history = deque(maxlen=window)
        self.drift_events = []         # 记录每次漂移检测
        self.frac_history = []

    def update(self, rand_eff, stage):
        """每阶段结束: 监测随机采样效率, 检测漂移, 调制设定点"""
        self.rand_eff_history.append(rand_eff)
        self.frac_history.append(self.frac)
        if len(self.rand_eff_history) < 2:
            return
        # 漂移检测: 随机采样效率骤降 (<25%) → 规则空间爆炸 (n 变大的信号)
        if rand_eff < self.rand_drift_thresh:
            self.frac = min(self.hi, self.frac + self.up_step)
            self.drift_events.append((stage, "drift_up", round(rand_eff, 2)))
        # 随机采样仍然高效 (>60%) → 下调引导 (多用便宜的随机采样)
        elif rand_eff > 60.0 and self.frac > self.lo:
            self.frac = max(self.lo, self.frac - self.down_step)


def evaluate_with_frac(F, slog, predictor, frac, budget=BUDGET, seed=0):
    """按 frac 分配预算: 引导部分用预测排序, 随机部分直接采样.
    返回: (总体效率, 引导部分效率, 随机部分效率)"""
    rng = np.random.RandomState(seed)
    n_cand = len(slog)
    halted = slog > 0
    true_best = float(np.max(slog[halted])) if halted.any() else 1e-9

    n_guided = max(1, int(budget * frac))
    n_rand = budget - n_guided

    # 预热 (小部分随机, 让预测器有点基础)
    warm = rng.choice(n_cand, max(3, budget // 6), replace=False)
    for i in warm:
        predictor.step(F[i], slog[i])

    # 引导部分: 预测 top, 模拟验证
    scores = [(predictor.predict(F[i]), i) for i in range(n_cand) if i not in warm]
    scores.sort(key=lambda x: -x[0])
    best_g = 0.0
    for _, idx in scores[:n_guided]:
        best_g = max(best_g, slog[idx])
        predictor.step(F[idx], slog[idx])

    # 随机部分
    unseen = [i for i in range(n_cand) if i not in warm]
    rng.shuffle(unseen)
    best_r = 0.0
    for i in unseen[:n_rand]:
        best_r = max(best_r, slog[i])

    best = max(best_g, best_r)
    total_eff = best / max(true_best, 1e-9) * 100
    g_eff = best_g / max(true_best, 1e-9) * 100
    r_eff = best_r / max(true_best, 1e-9) * 100
    return total_eff, g_eff, r_eff


def run_trial(use_config, seed=7):
    """跑完 n=2→5, 返回每阶段效率与参数轨迹 (n=5 注入稀有长运行机器)"""
    predictor = j6.SymbolPredictor(s_dim=6, seed=seed)
    cfg = Configurator() if use_config else None

    effs, fracs, rand_effs = [], [], []
    for n in STAGES:
        inject = (n == 5)
        F, slog, _ = build_drift_pool(n, 100 + n * 13, inject_long=inject)
        frac = cfg.frac if cfg else 0.5
        total_eff, g_eff, r_eff = evaluate_with_frac(F, slog, predictor, frac, seed=seed)
        effs.append(total_eff)
        rand_effs.append(r_eff)
        if cfg:
            cfg.update(r_eff, n)
            fracs.append(cfg.frac)
        else:
            fracs.append(0.5)

    return effs, fracs, rand_effs, (cfg.drift_events if cfg else [])


def main():
    print("=" * 78)
    print("JPI-9 Configurator 漂移场景 — 异稳态设定点调制 (BB n=2→5)")
    print("  监测: 随机采样效率骤降 (规则空间爆炸信号) → 上调引导预算比例")
    print("=" * 78)

    # 3 seed 平均
    fixed_eff, cfg_eff = [], []
    fixed_frac, cfg_frac = [], []
    fixed_rand, cfg_rand = [], []
    all_drift = []
    for seed in [1, 7, 42]:
        fe, ff, fr, _ = run_trial(False, seed)
        ce, cf, cr, drift = run_trial(True, seed)
        fixed_eff.append(fe); cfg_eff.append(ce)
        fixed_frac.append(ff); cfg_frac.append(cf)
        fixed_rand.append(fr); cfg_rand.append(cr)
        all_drift.extend(drift)

    fm = np.mean(fixed_eff, axis=0); cm = np.mean(cfg_eff, axis=0)
    ffm = np.mean(fixed_frac, axis=0); cfm = np.mean(cfg_frac, axis=0)
    frm = np.mean(fixed_rand, axis=0); crm = np.mean(cfg_rand, axis=0)

    print(f"\n{'n':>3} | {'固定效率':>8} {'Config效率':>9} {'增益':>7} | "
          f"{'随机效率':>7} {'固定frac':>8} {'Config frac':>9}")
    print("-" * 78)
    for i, n in enumerate(STAGES):
        gain = cm[i] - fm[i]
        print(f"{n:>3} | {fm[i]:7.1f}% {cm[i]:8.1f}% {gain:+6.1f}pp | "
              f"{frm[i]:6.1f}% {ffm[i]:7.2f} {cfm[i]:8.2f}")
    overall = np.mean(cm) - np.mean(fm)
    print(f"\n总体: Config {np.mean(cm):.1f}% vs 固定 {np.mean(fm):.1f}% → {overall:+.1f}pp")

    print(f"\n漂移检测事件 (3 seed 汇总):")
    if all_drift:
        for stage, kind, reff in all_drift:
            print(f"  n={stage}: {kind} (rand_eff={reff} < 0.4)")
    else:
        print("  无漂移事件触发")

    print(f"\nConfig frac 轨迹: {[round(float(x),2) for x in cfm]}")
    print(f"固定 frac 轨迹:   {[round(float(x),2) for x in ffm]}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
