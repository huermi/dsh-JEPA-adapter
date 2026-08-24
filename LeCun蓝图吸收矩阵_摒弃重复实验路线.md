# LeCun 蓝图吸收矩阵：摒弃重复实验，聚焦空白

> 日期：2026-08-24 ｜ 目的：把 Yann LeCun 2022 position paper + 2025-2026 阵营全部工作吃透，
> 明确哪些已实现（直接吸收）、哪些是空白（我们的战场），避免重复实验。
> 依据：LeCun 2022 v0.9.2；ETH Zürich 2026-06 演讲；AMI Labs 动态；I/V-JEPA、LeWorldModel、Micro-World Models、Curiosity-Critic、AdaJEPA、LLM-JEPA、EB-JEPA、stable-worldmodel。

---

## 0. 一句话

**LeCun 蓝图六个模块中，三个已被阵营实现到可复用程度（Perception/World Model/Actor 骨架），一个刚出现首次实现（H-JEPA），两个基本空白（Configurator、Short-term Memory 与 Cost 的完整接口）——而我们的模拟实证恰好落在"Cost 模块如何正确实现"这个空白上，且已经产出了 LeCun 没说清的三个关键细节。**

---

## 1. LeCun 2022 蓝图完整拆解（六模块 + 双模式）

| 模块 | LeCun 原设计 | 状态标签 |
|---|---|---|
| Configurator | 执行控制器，从所有模块收信号，调制 Perception/WM/Cost/Actor 的参数与注意力；**"possibly rule-based or learned via meta-optimization"**（他自己也没定） | 🔴 蓝图级（未实现） |
| Perception | 传感器 → 世界状态估计，层级表征，Configurator 调制提取哪些 | 🟢 已实现 |
| World Model | JEPA：预测表征非像素；潜变量 z 表达多值未来；H-JEPA 分层 | 🟢 已实现（JEPA）；🟡 H-JEPA 首次实现 |
| Cost | Intrinsic Cost（硬编码，不可训练）+ Critic（可训练，预测未来成本）；输出标量"不适度" | 🟡 部分（无完整实现） |
| Short-term Memory | 存当前/预测状态 + 关联成本；供 WM 回看、Critic 改进预测 | 🔴 蓝图级（无 JEPA 阵营实现） |
| Actor | 生成动作序列，最小化预测成本；Mode-1 反应 / Mode-2 规划（MPC）；只执行第一步 | 🟢 已实现（MPC/CEM） |

**双模式**：Mode-2 慢规划（世界模型模拟 + 成本评估 + 梯度优化动作序列）→ 蒸馏为 Mode-1 快速反应策略。**"编译 System-2 进 System-1"——这个蒸馏机制也没有系统实现。**

**关键力学**：所有模块可微，成本梯度可反传 WM→Actor 做推理时规划（energy minimization）。

---

## 2. 2025-2026 实现状态对照（阵营做了什么）

| 蓝图件 | 阵营实现 | 成熟度 | 吸收方式 |
|---|---|---|---|
| Perception + WM | I-JEPA / V-JEPA 2（百万小时视频）/ LeWorldModel（15M 两 loss 端到端） | ★★★★★ | **直接复用权重/配方** |
| 防坍缩 | SIGReg（LeJEPA，~50 行单超参）/ VICReg（EB-JEPA） | ★★★★★ | 直接移植 |
| Actor/规划 | stable-worldmodel CEM/iCEM/MPPI；V-JEPA 2-AC 潜空间规划 | ★★★★ | 直接复用求解器 |
| 在线适应 | AdaJEPA（1 步梯度，测试时自适应） | ★★★★ | 直接抄配方 |
| H-JEPA 分层 | Micro-World Models（2026.06，三体碰撞，双尺度 tap，无 EMA） | ★★ | 复现骨架，扩真实场景 |
| 符号层 | LLM-JEPA（text/code 视图预测）/ VL-JEPA（嵌入空间跟踪） | ★★ | 作为查询接口 |
| 价值/内在动机 | Curiosity-Critic（累积改进−基线）；Value-Guided JEPA（价值塑形） | ★★ | 借基线分离机制 |
| 记忆持久性 | Remember-to-be-Curious（3DGS 持久模型 + 情景记忆） | ★★ | 借"持久性前提"结论 |
| **Configurator** | **无任何实现** | ★ | **空白** |
| **Cost 完整模块** | **无完整实现**（V-JEPA 2-AC 用 goal 距离代替） | ★ | **空白** |
| **STM 与 WM 接口** | **无** | ★ | **空白** |
| **Mode-1/Mode-2 蒸馏** | **无系统实现** | ★ | **空白** |

---

## 3. 吸收清单（站在肩上，不复做）

以下全部有开源代码/论文，**直接吸收，绝不重复发明**：

1. **预测器**：LeWorldModel 两 loss 配方（预测 + SIGReg 高斯正则）——不要自己设计 loss。
2. **防坍缩**：SIGReg（`pip install lejepa`，~50 行核心）——不要手工调范数裁剪。
3. **规划器**：stable-worldmodel 的 CEM/MPPI 求解器——不要自己写搜索。
4. **编码器**：V-JEPA 2 开源权重（需要视觉时）；轻量场景用 LeWM 自训。
5. **在线适应**：AdaJEPA 配方（1 步梯度 + 最后层 + recent-5 回放 + lr 5e-4/1e-5）。
6. **时间分层**：Micro-World Models 骨架（双尺度 tap + VICReg 无 EMA）——复现而非重造。
7. **符号接口**：LLM-JEPA 的视图预测目标（需要配对数据时）；VL-JEPA 的嵌入空间跟踪。
8. **价值分离**：Curiosity-Critic 的"误差−基线"分离机制（认识论 vs 偶然误差）。

---

## 4. 空白清单（我们的战场——只有这些值得做）

### 4.1 Configurator（最大空白，LeCun 自己都没定）
- LeCun 原文："possibly rule-based or learned via meta-optimization"——**他不知道怎么实现**。
- **我们已有的实证**：饥饿速率扫描证明"内稳态动力学参数必须由外环自适应调制"——这就是 Configurator 的第一个具体机制！它不是一个神秘模块，而是"监测各环健康度 + 调制中环参数"的可计算控制器。
- 我们的定义：Configurator = 外环，输入是 E1/E2/覆盖/目标坚持度等统计量，输出是内稳态参数（饥饿速率）、探索温度、目标切换阈值。

### 4.2 Cost 模块的完整形态（我们的实证恰好在此）
- LeCun 只说了 intrinsic cost + critic 两个组件，没说：锚的形式（静态 vs 内稳态）、critic 如何前瞻、参数如何调制。
- **我们已实证**：锚必须"前瞻 + 内稳态 + 外环调制"三重结构；静态锚固着、当前锚无效、无锚死寂。
- 下一步：把 critic 实现为**可训练的**（预测未来 intrinsic cost），而不是我们模拟里的解析式 value_anchor——这是 LeCun 蓝图里 critic 的正确定义，我们还没做。

### 4.3 Short-term Memory 与 WM/Cost 的接口
- LeCun 说 STM 存"当前/预测状态 + 成本"，供 WM 回看、Critic 改进。
- 我们的记忆层（surprise 门控 + 价值加权采样）已有雏形，但**没有接 Critic 的训练信号**（Critic 应从记忆中的 (状态, 实际成本) 对训练）——这是蓝图要求、我们未做。

### 4.4 Mode-1/Mode-2 蒸馏
- LeCun 说 Mode-2 规划结果应蒸馏为 Mode-1 快速策略。
- 我们完全没有做。这是"从慢到快"的编译机制——蓝图有、无人实现。

---

## 5. 我们的实证在蓝图坐标系中的定位（最新）

| 我们的发现 | LeCun 蓝图对应 | 关系 |
|---|---|---|
| 锚必须内稳态（饥饿循环） | Intrinsic Cost 的形式细节 | **填补**（LeCun 没说） |
| 锚必须前瞻（value(s_next)） | Critic 的工作方式 | **填补**（LeCun 只说"预测未来成本"） |
| 外环调制内稳态参数 | Configurator 的具体机制 | **填补**（LeCun 说"possibly learned"但没定） |
| 两级能量统一失败 | Cost 与 JEPA 分离 | **回归**（验证 LeCun 正确） |
| 死寂失效模式 | Cost 存在的理由 | **实证**（LeCun 直觉，我们给了证据链） |

---

## 6. 差异化路线（摒弃重复后的推进计划）

| 阶段 | 做什么 | 不做什么（已吸收） | 对应空白 |
|---|---|---|---|
| P0 | 模拟器加**可训练 Critic**（从记忆的 (状态,实际成本) 对训练，预测未来成本） | 不重写预测器（用 LeWM 配方） | Cost 模块 |
| P1 | 模拟器加**Configurator 外环**（监测 E1/E2/覆盖 → 调制饥饿速率） | 不重造防坍缩（SIGReg） | Configurator |
| P2 | **STM→Critic 接口**（Critic 用 STM 数据训练，回授 WM） | 不重写规划器（CEM） | STM 接口 |
| P3 | **Mode-1 蒸馏**（Mode-2 规划结果训练快速策略） | 不重造编码器（V-JEPA 2） | 双模式 |
| P4 | 扩 Micro-World 骨架到真实场景 | 不重新设计 H-JEPA | 时间分层 |

**核心原则：四个空白（Configurator/Cost 完整形态/STM 接口/Mode 蒸馏）中，我们的模拟器已经站在 Cost 和 Configurator 两个空白的前沿——继续往这两个方向推，其余全部吸收。**

---

## 7. 结论

**LeCun 给的是"车的设计图"，阵营做出来了"发动机"（JEPA 世界模型）和"轮子"（规划器），但"方向盘"（Configurator）、"油门刹车"（Cost 完整形态）、"仪表盘"（STM 接口）和"变速箱"（Mode 蒸馏）还没人装上。** 我们的模拟实证已经证明自己坐在"油门刹车"（Cost）和"方向盘"（Configurator）这两个空白上——这正是继续推进的方向，其余全部吸收，绝不重复。
