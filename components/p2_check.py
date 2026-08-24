"""
P2 中环闭合验收 (components/p2_check.py)
==========================================
自主探索闭环: 内稳态需求 (饥饿) + 好奇心 (E2) 驱动,
无外部任务 — agent 自主移动、采集、记忆、切换目标.

环境: 节奏匹配 (jpi12 动力学) — 12×12, 3 资源, 再生周期随阶段
      漂移 (静止 500 / 漂移 80), agent 必须采集维持 (饥饿致死).

组装: C1 DummyPerception(768d) + C2 ResidualWorldModel + C3 CuriosityEnergy
      + C4 HomeostaticValue(v2 内稳态) + C5 AdaptiveMemory + C6 ValueAlignedGoal
      + C7 GreedyPlanner(v2 导航场) + C10 PerfConfigurator

验证: 自主性(移动率) / 记忆增长 / 进食采集 / 目标切换 / E2 活性 / 不失控(死亡)
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
from components.configurator import PerfConfigurator
from components.system import JepaAgent

WORLD_N = 12
N_ITEMS = 3
PHASE_LEN = 1500
N_STEPS = 4000
N_ACTIONS = 5
ACTION_DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
RES_BASE = [(3.0, 3.0), (9.0, 3.0), (3.0, 9.0)]
COLLECT_D = 0.8
FEED_THRESH = 0.55


class RhythmEnv:
    """节奏匹配环境: 资源再生周期随阶段漂移 (jpi12 动力学)"""
    def __init__(self, size=WORLD_N, phase_len=PHASE_LEN):
        self.size = size
        self.phase_len = phase_len
        self.t = 0
        self.pos = np.array([6.0, 6.0], dtype=np.float32)
        self.cooldown = [0, 0, 0]
        self.n_collect = 0

    def phase(self):
        return (self.t // self.phase_len) % 2

    def regen(self):
        return 500 if self.phase() == 0 else 80

    def item_pos(self, i):
        bx, by = RES_BASE[i]
        if self.phase() == 0:
            return np.array([bx, by], dtype=np.float32)
        return np.array([bx + 2.5 * np.sin(self.t * 0.03 + i * 2.1),
                         by + 2.5 * np.cos(self.t * 0.035 + i * 1.7)], dtype=np.float32)

    def obs(self):
        """观测: [agent_pos(2), item1(2), item2(2), item3(2)] = 8 维"""
        parts = [self.pos]
        for i in range(N_ITEMS):
            parts.append(self.item_pos(i))
        return np.concatenate(parts).astype(np.float32)

    def move(self, a):
        dx, dy = ACTION_DELTA[a]
        self.pos = np.clip(self.pos + np.array([dx, dy], dtype=np.float32),
                           0, self.size - 1)

    def collect(self):
        """资源再生检查 + 采集 (物理距离)"""
        got = False
        self.cooldown = [max(0, c - 1) for c in self.cooldown]
        for i in range(N_ITEMS):
            if self.cooldown[i] <= 0:
                d = float(np.linalg.norm(self.pos - self.item_pos(i)))
                if d < COLLECT_D:
                    self.cooldown[i] = self.regen()
                    self.n_collect += 1
                    got = True
        return got

    def reset_pos(self):
        self.pos = np.array([6.0, 6.0], dtype=np.float32)

    def resource_proto_vec(self, i):
        """构建"agent 位于资源 i 处"的观测向量 (用于注册资源原型)"""
        save = self.pos.copy()
        self.pos = self.item_pos(i).copy()
        v = self.obs()
        self.pos = save
        return v


def main():
    print("=" * 72)
    print("P2 中环闭合验收 — 自主探索闭环 (节奏匹配环境)")
    print("=" * 72)

    # ── 组装 ─────────────────────────────────────────────
    perception = DummyPerception(d_in=8, seed=7)
    wm = ResidualWorldModel(n_actions=N_ACTIONS, seed=7)
    energy = CuriosityEnergy()
    value = HomeostaticValue(sigma=2.2, feed_thresh=FEED_THRESH, seed=7)
    memory = AdaptiveMemory(seed=7)
    goal = ValueAlignedGoal(persistence=80)
    planner = GreedyPlanner(n_actions=N_ACTIONS, seed=7)
    planner.set_world_value(wm, value)
    config = PerfConfigurator()
    agent = JepaAgent(perception, wm, energy, value, memory, goal, planner,
                      configurator=config)

    env = RhythmEnv()
    # 资源原型注册 (感知编码: agent 位于资源处时的观测表征)
    for i in range(N_ITEMS):
        proto = perception.encode(env.resource_proto_vec(i))
        value.register_resource(proto)
    print(f"资源原型: {len(value.resource_protos)} 个注册")

    # ── 自主探索循环 (手动编排: 内稳态 hook) ─────────────
    stats = {"move": 0, "collect": 0, "feed": 0, "death": 0, "switch": 0}
    e2s, perfs, e1s = [], [], []
    goal_hold_prev = 0

    for t in range(N_STEPS):
        obs = env.obs()
        a, s = agent.decide(obs)
        env.move(a)
        if a != N_ACTIONS - 1:
            stats["move"] += 1

        # 内稳态: hunger 推进 + 死亡检测
        died = value.tick()
        # 采集 (资源再生)
        got = env.collect()
        # 进食 (价值系统内, 基于资源原型表征)
        s_here = perception.encode(env.obs())
        if value.try_feed(s_here):
            stats["feed"] += 1

        perf = (1.0 if got else 0.0) - (5.0 if died else 0.0)
        if got:
            stats["collect"] += 1
        if died:
            stats["death"] += 1
            env.reset_pos()

        obs_next = env.obs()
        ev = agent.learn(obs, a, obs_next, perf, died)
        e1s.append(ev.e1); e2s.append(ev.e2); perfs.append(perf)

        # 目标切换统计
        if agent._goal_hold < goal_hold_prev:
            stats["switch"] += 1
        goal_hold_prev = agent._goal_hold
        env.t += 1

    # ── 验证 ─────────────────────────────────────────────
    move_rate = stats["move"] / N_STEPS
    n_mem = len(memory.items)
    n_proto = len(memory.prototypes)
    e2_active = max(e2s) > min(e2s)
    print(f"\n{'指标':<16} | {'值':>10} | 判定")
    print("-" * 52)
    checks = []
    ok_move = move_rate > 0.3
    ok_mem = n_mem > 10
    ok_feed = stats["feed"] > 0
    ok_collect = stats["collect"] > 0
    ok_switch = stats["switch"] > 0
    ok_e2 = e2_active
    ok_death = stats["death"] < N_STEPS // 100
    print(f"{'移动率':<16} | {move_rate:>8.2f}  | {'✅ 自主行动' if ok_move else '❌ 死寂'}")
    print(f"{'记忆条目':<16} | {n_mem:>8d}  | {'✅ 惊讶门控写入' if ok_mem else '❌ 无记忆'}")
    print(f"{'原型数':<16} | {n_proto:>8d}  |")
    print(f"{'进食次数':<16} | {stats['feed']:>8d}  | {'✅ 内稳态循环' if ok_feed else '❌ 无进食'}")
    print(f"{'采集次数':<16} | {stats['collect']:>8d}  | {'✅ 资源获取' if ok_collect else '❌ 无采集'}")
    print(f"{'目标切换':<16} | {stats['switch']:>8d}  | {'✅ 坚持性预算' if ok_switch else '❌ 无切换'}")
    print(f"{'E2 活性':<16} | {max(e2s)-min(e2s):>8.3f}  | {'✅ 价值信号' if ok_e2 else '❌ 失活'}")
    print(f"{'死亡次数':<16} | {stats['death']:>8d}  | {'✅ 内稳态生效' if ok_death else '❌ 失控'}")
    print(f"{'E1 均值':<16} | {np.mean(e1s):>8.5f}  |")
    print(f"{'绩效均值':<16} | {np.mean(perfs):>8.5f}  |")
    print(f"{'Config 速率末值':<16} | {config.rate:>8.4f}  |")

    ok = all([ok_move, ok_mem, ok_feed, ok_collect, ok_switch, ok_e2, ok_death])
    print("\n" + "=" * 72)
    print(f"P2 中环闭合: {'✅ 自主探索闭环成立' if ok else '❌ 未完全通过'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
