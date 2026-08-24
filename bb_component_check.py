"""
BB 组件化引导对照 (bb_component_check.py)
============================================
⚠️ 2026-08-24 修正声明: 本文件使用"行为伪影注入" (手拍特征值 F=1.0/0.8),
已被 bb_human_check.py 证明失真 — ε-greedy 43.3% 是伪影假象
(真实结构池上 24.8% 且命中 0), 回归 24% 被低估 (真实池 50.1%).
本文件仅保留为"随机探索在伪影目标上的演示"; 真实结论以
BB人类式反汇编实验与错误结论反思.md 和 bb_human_check.py 为准.

用组件系统 (内稳态 C4 + Configurator C10) 重新做 Busy Beaver 引导,
对照 jpi6 单体预测器引导, 并检验"生物学比喻参数"在 BB 上的标定.

BB 任务映射 (网格世界 → BB):
  网格世界: hunger 驱动"去资源"   → BB: hunger 驱动"利用预测" (探索-利用门控)
  网格世界: 进食 = 靠近资源        → BB: 进食 = 验证到长运行机器 (log > 阈值)
  网格世界: 死亡 = 长期饥饿        → BB: 死亡 = 长期没发现好机器 (探索失败期)
  网格世界: tick 数千次            → BB: 预算仅 30-80 次验证 (时间尺度不同!)

对照模式:
  A 固定引导 (jpi6 基线): warm 20% + 预测 top 80%
  B 内稳态引导: hunger 门控探索-利用 + 进食/死亡
  C B + Configurator: 调制 hunger_rate (绩效观测)

参数扫描: hunger_rate ∈ {0.01, 0.05, 0.1, 0.3} (按"验证次数"尺度)
"""
import os as _os
REPO_ROOT = _os.path.dirname(_os.path.abspath(__file__))
import sys
import numpy as np

sys.path.insert(0, REPO_ROOT)
from jpi6_behavioral_symbols import build_pool
from jpi4_symbolic_bb import SymbolPredictor
from components.configurator import PerfConfigurator

N_CAND = 1500
BUDGET = 60
WARM = 10
FEED_THRESH_LOG = 4.0        # log 步数 > 4 (≈55 步) 视为"好机器" (进食)
DEATH_DELAY = 15             # 连续 15 次没发现好机器 → 探索失败期
EXPLORE_GATE = 0.5           # hunger 超过此值才利用预测


class BBHomeostasis:
    """BB 内稳态: hunger 驱动利用, 进食=发现长机器, 死亡=探索失败期
    v2: explore_temp (ε-greedy) — Configurator 调制探索概率"""
    def __init__(self, hunger_rate=0.05, feed_thresh=FEED_THRESH_LOG,
                 death_delay=DEATH_DELAY, gate=EXPLORE_GATE,
                 explore_temp=0.3):
        self.hunger_rate = hunger_rate
        self.feed_thresh = feed_thresh
        self.death_delay = death_delay
        self.gate = gate
        self.explore_temp = explore_temp
        self.hunger = 0.0
        self.death_counter = 0
        self.n_feed = 0
        self.n_death = 0

    def tick(self) -> bool:
        """hunger 增长 + 死亡检测. 返回是否死亡"""
        self.hunger = min(1.0, self.hunger + self.hunger_rate)
        if self.hunger >= 1.0:
            self.death_counter += 1
            if self.death_counter >= self.death_delay:
                self.n_death += 1
                self.death_counter = 0
                self.hunger = 0.0
                self.explore_temp = min(0.9, self.explore_temp + 0.3)  # 疼痛→探索新方向
                return True
        else:
            self.death_counter = 0
        return False

    def feed(self, log_steps: float) -> None:
        """验证到长机器 → 进食 (需求下降 + 降低探索 = 利用)"""
        if log_steps > self.feed_thresh:
            self.hunger = max(0.0, self.hunger - 0.5)
            self.explore_temp = max(0.05, self.explore_temp - 0.15)
            self.n_feed += 1

    def hungry(self) -> float:
        return self.hunger


def guided(predictor, F, slog, mode, homeo=None, cfg=None, budget=BUDGET, seed=0):
    """引导循环. mode: 'fixed' | 'homeo' | 'homeo_cfg'
    返回 (效率%, 统计)"""
    rng = np.random.RandomState(seed)
    n = len(slog)
    true_best = float(np.max(slog[slog > 0])) if (slog > 0).any() else 1e-9
    best_found = 0.0

    # warm-up
    warm_idx = rng.choice(n, WARM, replace=False)
    for i in warm_idx:
        predictor.step(F[i], slog[i])
        best_found = max(best_found, slog[i])
    seen = set(warm_idx.tolist())

    n_left = budget - WARM
    perf_window = []          # Configurator 绩效
    stats = {"feed": 0, "death": 0, "explore": 0, "exploit": 0}

    for step in range(n_left):
        unseen = [i for i in range(n) if i not in seen]
        if mode == "fixed":
            # 固定引导: 预测 top
            scores = [(predictor.predict(F[i]), i) for i in unseen]
            scores.sort(key=lambda x: -x[0])
            idx = scores[0][1]
            stats["exploit"] += 1
        else:
            # 内稳态门控 + ε-greedy (explore_temp): 饥饿 → 利用; 随机 → 探索
            if homeo.hungry() > homeo.gate and \
                    rng.rand() > homeo.explore_temp and len(unseen) > 5:
                scores = [(predictor.predict(F[i]), i) for i in unseen]
                scores.sort(key=lambda x: -x[0])
                idx = scores[0][1]
                stats["exploit"] += 1
            else:
                idx = unseen[rng.randint(len(unseen))]
                stats["explore"] += 1

        # 验证
        seen.add(idx)
        log_steps = slog[idx]
        best_found = max(best_found, log_steps)

        # 学习
        predictor.step(F[idx], log_steps)

        if mode != "fixed":
            # 内稳态: 进食 + 死亡
            homeo.feed(log_steps)
            if homeo.tick():
                stats["death"] += 1
                # 探索失败期: 清空绩效窗口 (Configurator 看到绩效骤降)
                perf_window = []
            if log_steps > homeo.feed_thresh:
                stats["feed"] += 1
            # Configurator: 绩效 = 窗口平均发现 log (量纲: log 步数)
            if cfg is not None:
                perf_window.append(log_steps)
                if len(perf_window) > 8:
                    perf_window = perf_window[-8:]
                perf = float(np.mean(perf_window))
                cfg.observe(perf, step, died=homeo.n_death > stats["death"])
                # BB 关键自由度 = 探索温度 (重映射: 绩效高 → 降探索)
                rate = cfg.get_config().get(
                    __import__("components.core", fromlist=["ParamKind"]).ParamKind.HOMEOSTASIS_RATE,
                    homeo.hunger_rate)
                homeo.hunger_rate = rate
                if perf > 3.0:
                    homeo.explore_temp = max(0.05, homeo.explore_temp - 0.1)

    eff = best_found / true_best * 100
    return eff, stats


def run_mode(mode, hunger_rate, seed=7):
    predictor = SymbolPredictor(s_dim=6, seed=seed)
    homeo = BBHomeostasis(hunger_rate=hunger_rate) if mode != "fixed" else None
    cfg = PerfConfigurator(base_rate=hunger_rate) if mode == "homeo_cfg" else None
    return guided(predictor, F, slog, mode, homeo, cfg, seed=seed)

def inject_long_machines(F, slog, n_inject=4, seed=7):
    """注入稀有长运行机器 (计数器机器特征, jpi9 教训: 随机采样碰不到长尾).
    behavior 6 维: [状态多样度, 磁带扩张率, 停机, 范围扩散, 线性扩张, 状态熵]"""
    rng = np.random.RandomState(seed)
    for k in range(n_inject):
        idx = rng.randint(len(slog))
        # 计数器机器行为指纹: 线性扩张、不停机、状态多样
        F[idx, 0] = 1.0    # 状态多样度
        F[idx, 1] = 1.0    # 磁带扩张率 (线性)
        F[idx, 2] = 0.0    # 短模拟不停机
        F[idx, 3] = 0.8    # 磁带范围扩散
        F[idx, 4] = 1.0    # 线性扩张迹象 (计数器候选)
        F[idx, 5] = 0.8    # 状态熵
        slog[idx] = 8.0 + k * 1.0    # log 步数 8-11 (稀有长运行)
    return F, slog


def main():
    global F, slog
    print("=" * 78)
    print("BB 组件化引导对照 — 内稳态/Configurator 在结构长尾上的检验")
    print("=" * 78)
    print("构建候选池 (n=4, behavior 指纹, 1500 候选) + 注入 4 台长机器...")
    F, slog, _ = build_pool(4, 113, "behavior", n_candidates=N_CAND)
    F, slog = inject_long_machines(F, slog)
    true_best = float(np.max(slog[slog > 0]))
    print(f"真实最长 log 步数: {true_best:.2f} (注入后) | 停机机器: {(slog>0).sum()}/{N_CAND}")

    print(f"\n{'模式':<22} | {'hr':>5} | {'效率':>7} | {'喂':>3} {'死':>3} {'探':>4} {'用':>4}")
    print("-" * 70)

    # 基线 A: 固定引导 (3 seed)
    effs_a = []
    for seed in [1, 7, 42]:
        eff, st = run_mode("fixed", 0.0, seed)
        effs_a.append(eff)
    print(f"{'A 固定引导 (jpi6)':<22} | {'-':>5} | {np.mean(effs_a):6.1f}% |"
          f" {'-':>3} {'-':>3} {'-':>4} {'-':>4}")

    # 参数扫描: 内稳态 hunger_rate
    results = {}
    for hr in [0.01, 0.05, 0.1, 0.3]:
        for mode, tag in [("homeo", "B 内稳态"),
                          ("homeo_cfg", "C 内稳态+Config")]:
            effs, feeds, deaths, exps, exps_ = [], [], [], [], []
            for seed in [1, 7, 42]:
                eff, st = run_mode(mode, hr, seed)
                effs.append(eff)
                feeds.append(st["feed"]); deaths.append(st["death"])
                exps.append(st["explore"]); exps_.append(st["exploit"])
            m = np.mean(effs)
            results[(mode, hr)] = m
            print(f"{tag:<22} | {hr:>4.2f} | {m:6.1f}% |"
                  f" {int(np.mean(feeds)):>3} {int(np.mean(deaths)):>3}"
                  f" {int(np.mean(exps)):>4} {int(np.mean(exps_)):>4}")

    # 汇总
    print("\n" + "=" * 78)
    best = max(results, key=lambda k: results[k])
    print(f"最优配置: {best[0]} hr={best[1]:.2f} → {results[best]:.1f}%")
    print(f"固定引导基线: {np.mean(effs_a):.1f}%")
    gain = results[best] - np.mean(effs_a)
    print(f"内稳态/Configurator 增益: {gain:+.1f}pp")
    print(f"\n生物学比喻参数检验结论:")
    print(f"  内稳态比喻 (hunger 驱动利用/进食/死亡) 在 BB 上{'有效' if gain>2 else '无效'} —")
    print(f"  hunger_rate 最优值 {best[1]} vs 网格世界 0.001-0.03:")
    print(f"  {'量纲一致' if 0.001 <= best[1] <= 0.03 else '需按验证次数重标定 (BB tick=每次验证, 预算仅60)'}")


if __name__ == "__main__":
    main()
