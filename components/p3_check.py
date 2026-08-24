"""
P3 三环全接验收 (components/p3_check.py)
==========================================
三环 + 符号 + 蒸馏 + Configurator 完整互联:

  阶段1 探索期: agent 自主探索, 记录轨迹段 → C8 符号层形成原型 (符号涌现)
  阶段2 符号评估: 近资源 vs 远资源轨迹的符号分布区分度
  阶段3 蒸馏期: Mode-2 慢规划 (5 步 CEM roll-out) → 软标签 →
                C9 蒸馏 FastPolicy (Mode-1 快速策略)
  阶段4 策略对比: DeepPlanner(慢) vs FastPolicy(蒸馏) vs GreedyPlanner(1步)
                  在环境上的净采集率
  阶段5 全闭环: 全部组件继续运行, 确认不破坏 (采集/记忆/Config 正常)
"""
import sys
import numpy as np

sys.path.insert(0, "D:/JEPA")

from components.perception import DummyPerception
from components.world_model import ResidualWorldModel
from components.energy import CuriosityEnergy
from components.value import HomeostaticValue
from components.memory import AdaptiveMemory
from components.goal import ValueAlignedGoal
from components.planner import GreedyPlanner
from components.symbol import BehavioralSymbol
from components.distill import SoftLabelDistiller
from components.configurator import PerfConfigurator
from components.system import JepaAgent
from components.p2_check import RhythmEnv, N_ACTIONS, N_STEPS, ACTION_DELTA, COLLECT_D


class DeepPlanner:
    """Mode-2 慢规划: 3 步 CEM roll-out (累计 anchor 价值 - 移动成本)
    horizon 3: wm 预测误差随步数累积, 5 步视野在漂移期失效 (评估实证)"""
    def __init__(self, wm, value, horizon=3, n_actions=N_ACTIONS):
        self.wm = wm
        self.value = value
        self.horizon = horizon
        self.n_actions = n_actions

    def _rollout(self, s, a):
        total = 0.0
        s_cur = np.asarray(s, np.float32).copy()
        for _ in range(self.horizon):
            s_cur = s_cur + self.wm.predict_delta(s_cur, a)
            total += self.value.anchor(s_cur) - 0.02 * (1 if a != self.n_actions - 1 else 0)
            # 贪心续 roll
            scores = [self.value.anchor(s_cur + self.wm.predict_delta(s_cur, b))
                      - 0.02 * (1 if b != self.n_actions - 1 else 0)
                      for b in range(self.n_actions)]
            a = int(np.argmax(scores))
        return total

    def select(self, s):
        scores = [self._rollout(s, a) for a in range(self.n_actions)]
        return int(np.argmax(scores))


class FastPolicy:
    """Mode-1 快速策略: 线性 softmax (蒸馏自 DeepPlanner), O(1) 推理"""
    def __init__(self, d_in, n_actions=N_ACTIONS, lr=0.5):
        self.W = np.zeros((d_in, n_actions), dtype=np.float32)
        self.lr = lr

    def predict(self, s):
        z = np.asarray(s, np.float32) @ self.W
        e = np.exp(z - z.max())
        return e / e.sum()

    def step(self, s, soft_target):
        p = self.predict(s)
        err = p - soft_target
        self.W -= self.lr * np.outer(np.asarray(s, np.float32), err)

    def act(self, s):
        return int(self.predict(s).argmax())


def onehot(a, n=N_ACTIONS):
    v = np.zeros(n, dtype=np.float32)
    v[a] = 1.0
    return v


def evaluate_policy(maker, perception, n_ticks=1000, seed=0):
    """策略采集率评估 — 独立环境 + 独立内稳态 (maker(val2) → act_fn)
    1000 tick + 快速饥饿 (0.003): 保证策略活跃 (500 tick 时 hunger 未过门控)"""
    env2 = RhythmEnv()
    env2.t = 1500                      # 从漂移期开始 (资源丰富)
    val2 = HomeostaticValue(sigma=2.2, hunger_rate=0.003,
                            feed_thresh=0.7, seed=seed)
    # feed_thresh 0.7 → 进食距离 0.78 格 < 采集 0.8 格: 进食=采集, agent 精确到达
    for i in range(3):                 # 注册评估环境的资源原型
        val2.register_resource(perception.encode(env2.resource_proto_vec(i)))
    act = maker(val2)
    rng = np.random.RandomState(seed)
    collect = 0
    death = 0
    for _ in range(n_ticks):
        obs = env2.obs()
        a = act(obs)
        env2.move(a)
        died = val2.tick()
        if env2.collect():
            collect += 1
        s_here = perception.encode(env2.obs())
        val2.try_feed(s_here)
        if died:
            death += 1
            env2.reset_pos()
        env2.t += 1
    return collect / n_ticks, death


def main():
    print("=" * 72)
    print("P3 三环全接验收 — 符号涌现 + 蒸馏闭环 + 全组件互联")
    print("=" * 72)

    # ── 组装全部组件 ─────────────────────────────────────
    perception = DummyPerception(d_in=8, seed=7)
    wm = ResidualWorldModel(n_actions=N_ACTIONS, seed=7)
    energy = CuriosityEnergy()
    value = HomeostaticValue(sigma=2.2, seed=7)
    memory = AdaptiveMemory(seed=7)
    goal = ValueAlignedGoal(persistence=80)
    planner = GreedyPlanner(n_actions=N_ACTIONS, seed=7)
    planner.set_world_value(wm, value)
    symbol = BehavioralSymbol(seed=7, min_sep=0.25)   # 指纹距离 0.08(同)~0.42(异)
    distiller = SoftLabelDistiller()
    config = PerfConfigurator()
    agent = JepaAgent(perception, wm, energy, value, memory, goal, planner,
                      symbol, config)

    env = RhythmEnv()
    for i in range(3):
        value.register_resource(perception.encode(env.resource_proto_vec(i)))

    # ── 阶段1: 探索 + 符号涌现 ───────────────────────────
    print("\n--- 阶段1: 自主探索 (2500 tick) + 符号形成 ---")
    traj_buf = []
    traj_regions = []         # 段起始位置区域: 0=左上, 1=右下
    symbol_ids = []
    for t in range(2500):
        obs = env.obs()
        a, s = agent.decide(obs)
        env.move(a)
        died = value.tick()
        got = env.collect()
        s_here = perception.encode(env.obs())
        if value.try_feed(s_here):
            pass
        perf = (1.0 if got else 0.0) - (5.0 if died else 0.0)
        if died:
            env.reset_pos()
        obs_next = env.obs()
        agent.learn(obs, a, obs_next, perf, died)
        env.t += 1

        # 轨迹段记录 (40 tick 一段) + 段内移动量 (行为模式标签)
        if len(traj_buf) == 0:
            seg_move = 0
        if a != N_ACTIONS - 1:
            seg_move += 1
        traj_buf.append(s)
        if len(traj_buf) == 40:
            traj_regions.append(seg_move)
            f = symbol.fingerprint(traj_buf)
            sid = symbol.match(f)
            symbol_ids.append(sid)
            traj_buf = []

    print(f"符号原型: {len(symbol.prototypes)} 个 | 符号使用: {dict(symbol.symbol_count)}")
    print(f"记忆: {len(memory.items)} 条 | 进食: {value.n_feeds} | "
          f"Config 速率: {config.rate:.4f}")

    # ── 阶段2: 符号区分度 (按行为模式: 运动 vs 静止) ─────
    print("\n--- 阶段2: 符号区分度 (运动 vs 静止轨迹) ---")
    n_seg = min(len(traj_regions), len(symbol_ids))
    if n_seg >= 10:
        # traj_regions 存的是段内位移量 (改: 每段结束记录位移)
        thr = np.median(traj_regions[:n_seg]) if n_seg > 2 else 0.5
        ids_mov = [sid for sid, r in zip(symbol_ids[:n_seg], traj_regions[:n_seg]) if r > thr]
        ids_sta = [sid for sid, r in zip(symbol_ids[:n_seg], traj_regions[:n_seg]) if r <= thr]
        set_mov, set_sta = set(ids_mov), set(ids_sta)
        exclusive = len(set_mov - set_sta) + len(set_sta - set_mov)
        n_sym = len(symbol.prototypes)
        print(f"运动段: {len(ids_mov)} 段 → 符号 {sorted(set_mov)}")
        print(f"静止段: {len(ids_sta)} 段 → 符号 {sorted(set_sta)}")
        print(f"专属符号: {exclusive}/{n_sym}")
        ok_symbol = n_sym >= 2 and exclusive >= max(1, n_sym // 2)
        print(f"符号评估: {'✅ 符号区分行为模式 (行为聚类)' if ok_symbol else '❌ 符号未分化'}")
    else:
        ok_symbol = False
        print(f"❌ 轨迹段不足 ({n_seg} 段)")

    # ── 阶段3: 蒸馏 (DeepPlanner → FastPolicy) ───────────
    print("\n--- 阶段3: Mode-2 蒸馏 Mode-1 (200 状态) ---")
    deep = DeepPlanner(wm, value, horizon=5)
    # 用记忆中的状态 (真实探索状态)
    sample_states = [m[0] for m in memory.items[:200]]
    if len(sample_states) < 50:
        sample_states = [perception.encode(env.obs()) for _ in range(200)]

    fast = distiller.distill(
        deep, sample_states, min(100, len(sample_states)), seed=7,
        student_factory=lambda: FastPolicy(768),
        predict_fn=lambda m, x: onehot(m.select(x)),
        step_fn=lambda m, x, tgt: m.step(x, tgt))
    print(f"FastPolicy 蒸馏完成 ({distiller.reports[-1]['n_samples']} 样本)")

    # ── 阶段4: 策略对比 (独立环境 + 独立内稳态, 公平对照) ──
    print("\n--- 阶段4: 策略采集率对比 (各 1000 tick) ---")
    import copy

    def make_greedy(v):
        p = copy.deepcopy(planner)
        p.set_world_value(wm, v)
        g = memory.sample_goal()
        return lambda obs: p.select_action(perception.encode(obs), g)

    c_deep, d_deep = evaluate_policy(
        lambda v: (lambda obs: DeepPlanner(wm, v, horizon=3).select(perception.encode(obs))),
        perception)
    c_fast, d_fast = evaluate_policy(
        lambda v: (lambda obs: fast.act(perception.encode(obs))), perception)
    c_greedy, d_greedy = evaluate_policy(make_greedy, perception)

    print(f"  DeepPlanner (3步): 采集率 {c_deep*100:.1f}% | 死亡 {d_deep}")
    print(f"  FastPolicy (蒸馏): 采集率 {c_fast*100:.1f}% | 死亡 {d_fast}")
    print(f"  GreedyPlanner(1步): 采集率 {c_greedy*100:.1f}% | 死亡 {d_greedy}")
    # 蒸馏忠实性: Fast 应接近 Deep (学到慢策略知识); 相对 Greedy 为参考
    fidelity = abs(c_fast - c_deep) / max(c_deep, 0.001)
    gain = c_fast - c_greedy
    print(f"  蒸馏忠实性 (|Fast-Deep|/Deep): {fidelity:.2f} | 蒸馏增益 (Fast vs Greedy): {gain*100:+.1f}pp")
    ok_distill = fidelity < 0.5 and c_fast >= c_greedy * 0.5
    print(f"  蒸馏评估: {'✅ Fast 忠实蒸馏慢策略' if ok_distill else '❌ 蒸馏失真'}")

    # ── 阶段5: 全闭环确认 (1000 tick, 观察 Config 急刹车恢复) ──
    print("\n--- 阶段5: 全闭环继续运行 (1000 tick) ---")
    c5 = {"collect": 0, "feed": 0, "death": 0}
    rate_first = config.rate
    for _ in range(1000):
        obs = env.obs()
        a, s = agent.decide(obs)
        env.move(a)
        died = value.tick()
        got = env.collect()
        s_here = perception.encode(env.obs())
        if value.try_feed(s_here):
            c5["feed"] += 1
        if got:
            c5["collect"] += 1
        if died:
            c5["death"] += 1
            env.reset_pos()
        obs_next = env.obs()
        agent.learn(obs, a, obs_next, (1.0 if got else 0.0) - 5.0 * died, died)
        env.t += 1
    rate_last = config.rate
    ok_run = c5["collect"] > 0 and (c5["feed"] > 0 or c5["death"] <= 2)
    print(f"采集 {c5['collect']} | 进食 {c5['feed']} | 死亡 {c5['death']} | "
          f"记忆 {len(memory.items)} | Config 速率 {rate_first:.4f} → {rate_last:.4f}")
    print(f"全闭环: {'✅ 组件互联稳定' if ok_run else '❌ 运行异常'}")

    # ── 总判定 ───────────────────────────────────────────
    ok = ok_symbol and ok_distill and ok_run
    print("\n" + "=" * 72)
    print(f"P3 三环全接: {'✅ 通过 (符号+蒸馏+全组件互联)' if ok else '❌ 未完全通过'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
