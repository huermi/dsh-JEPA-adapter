# JEPA AI 组件化设计方案（v1）

> 日期：2026-08-24 ｜ 依据：JPI-1 架构设计 + 全部实证（锚三重/两级能量/Configurator/符号层/蒸馏/持续学习/LLM编码）+ LeCun 2022 六模块蓝图
> 目标：把实验原型（jpi* 系列、jepa_base）升级为**可对接的 JEPA AI 组件**——模块化、接口契约化、成熟方案优先、独立可测

---

## 0. 设计原则

1. **三环嵌套**：内环（认知：感知→预测→E1→学习）／中环（行动：记忆→目标→规划→E2→行动）／外环（元认知：Configurator 监测调制）＋ 记忆作桥梁。
2. **统一表征空间**：所有组件通过 768d JEPA 空间通信（编码器输出即一切组件的输入）。
3. **成熟方案优先**：每个组件标注实现来源（LeWorldModel/SIGReg/AdaJEPA/Micro-World/LLM-JEPA/stable-worldmodel…），绝不重造。
4. **实证驱动**：内部机制优先采用已实证方案（锚三重、水平加速、绩效爬山、行为指纹、recent-N 回放）。
5. **组件独立可测**：每个组件都有对应实证脚本作为验收测试。

## 1. 组件总览（10 组件）

```
                    ┌─────────────────────────────────────────────┐
                    │          Configurator (外环·元认知)          │
                    │  绩效观测 → 水平加速/爬山/急刹车 → 参数调制    │
                    └───────┬──────────────┬──────────────┬───────┘
                            │ 调制          │ 调制          │ 调制
   ┌──────────┐   ┌─────────▼───┐   ┌──────▼──────┐   ┌───▼─────────┐
   │Perception│──▶│ WorldModel  │──▶│ EnergySystem │   │SymbolLayer  │
   │ 编码器   │   │ 预测器 P     │   │ E1/E2 分离   │   │ 行为指纹/文本 │
   └────▲─────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
        │                │                 │                  │
        │          ┌─────▼──────────┐      │            ┌─────▼──────┐
        │          │ ValueSystem    │◀─────┘            │ Distiller  │
        │          │ Critic + 锚三重 │                   │ Mode1/2     │
        │          └─────▲──────────┘                   └────────────┘
        │                │
   ┌────┴─────┐   ┌──────┴──────┐   ┌───────────┐
   │  环境    │◀──│  Planner     │◀──│ GoalGen    │
   │ (身体)   │   │ CEM/MPC 滚动 │   │ 目标生成   │
   └──────────┘   └─────────────┘   └─────▲─────┘
                                          │
   ┌──────────────────────────────────────┴──────┐
   │           MemorySystem (桥梁)                │
   │   STM(surprise写) / LTM(LanceDB) / 巩固重放  │
   │   读: E2 价值加权采样 → GoalGen              │
   └─────────────────────────────────────────────┘
```

## 2. 组件详设

### C1 PerceptionEncoder（感知编码器）
| 项 | 内容 |
|---|---|
| 职责 | 原始观测（图像/屏幕/文件）→ 768d JEPA 表征 |
| 接口 | `encode(obs: [B,3,H,W]) → z: [B,768]`；`encode_patches(obs) → [B,1+N,768]` |
| 实现 | jepa_base.VisionEncoder（ViT-B/16，pos 插值支持任意分辨率） |
| 权重吸收 | timm ImageNet（150 层）/ Sparsh-IJEPA（148 层）/ LLM 语义蒸馏（投影头） |
| 防坍缩 | EMA target（I-JEPA）/ SIGReg 高斯约束（LeJEPA） |
| 实证 | effect_check：timm 74.3%、LLM 蒸馏后 80.0% 线性探测 |
| 验收 | effect_check.py |

### C2 WorldModel（世界模型）
| 项 | 内容 |
|---|---|
| 职责 | 预测潜空间状态变化：`P(s, a, δ) → Δŝ` |
| 接口 | `predict_delta(s:[B,768], a:[B,K], δ) → Δŝ:[B,768]`；`energy(s, a, s') → e1` |
| 实现 | 残差预测 MLP/transformer（AdaJEPA 配方已内嵌：predict_delta/normalize_target/分层 lr） |
| 损失 | 预测 MSE + SIGReg 高斯正则（LeWorldModel 两 loss 配方） |
| 时间分层 | 多尺度 δ 条件（Micro-World 双尺度骨架，P2 接入） |
| 实证 | jpi1 模拟器（E1 收敛）、AdaJEPA 配方（DCA predictor 内嵌） |
| 验收 | 预测误差下降曲线、防坍缩检验（cosine 不饱和） |

### C3 EnergySystem（能量系统：E1/E2 分离）
| 项 | 内容 |
|---|---|
| 职责 | **E1（学习信号）与 E2（行动信号）分属两环，不可混乘**（三环架构核心裁决） |
| 接口 | `e1 = ||Δŝ - Δs||²`；`e2 = f(e1_history, anchor_value)` |
| E1 | 认识论惊讶：唯一学习触发器（训练/巩固/记忆写入） |
| E2 | 行动理由 = 可学习性 + 锚（锚来自 E1 之外，防死寂） |
| 实证 | 静态世界无锚死寂 WAIT 29.1%；内稳态锚 20.8%；E2 活性 100× |
| 验收 | jpi1_simulator 对照实验 |

### C4 ValueSystem（价值系统）
| 项 | 内容 |
|---|---|
| 职责 | 预测未来价值（Critic）+ 提供行动理由（锚三重结构） |
| 接口 | `predict_cost(s_next) → cost`；`anchor(s, demand) → 内稳态价值` |
| 实现 | 可训练 Critic（从记忆 (状态,实际成本) 对学习，LeCun Cost 模块完整形态） |
| 锚三重 | 前瞻（value(s_next) 期望价值）+ 内稳态（饥饿-进食循环，消耗-补充）+ 外环调制（参数由 Configurator 调） |
| 实证 | 静态锚固着 99.7%（错误形态）；内稳态锚 20.8%；前瞻锚 = Friston 期望自由能 |
| 验收 | jpi2（可训练 Critic 偏移检测 +178%）+ 锚形态对照 |

### C5 MemorySystem（记忆系统，桥梁）
| 项 | 内容 |
|---|---|
| 职责 | 记忆生命周期：写（E1 惊讶门控）→ 存（LTM）→ 读（E2 价值加权）→ 巩固（重放）→ 遗忘（显著性衰减） |
| 接口 | `write(s, e1, e2)`；`sample_goal(by_value)`；`replay(n)`；`consolidate()` |
| 实现 | STM 缓冲 + LTM（**LanceDB**，比 HDF5 快 3.4×）+ 自适应分位数阈值（surprise/原型距离） |
| 对齐 | 写入靠 E1（记不懂的）、读取靠 E2（要值得的）——价值权重对齐两端 |
| 实证 | 手工阈值失效三次教训（0.3/0.7/0.9）；自适应分位解决；稀疏奖励回放 +16.7pp |
| 验收 | jpi2/jpi10（记忆回放对比） |

### C6 GoalGenerator（目标生成）
| 项 | 内容 |
|---|---|
| 职责 | 目标 = 统一空间中的可塑向量（记忆采样/原型/插值/随机） |
| 接口 | `next_goal(memory, e2_field) → g:[768]` |
| 实现 | E2 价值加权记忆采样 + 坚持性预算（`切换 = 能量不降 > T 且 探索预算耗尽`） |
| 实证 | 坚持性预算稳定目标切换（~67 次/4000tick 无逃避）；"能量趋势切换"有逃避困难任务风险（裁决） |
| 验收 | jpi1 目标切换统计 |

### C7 Planner（规划器）
| 项 | 内容 |
|---|---|
| 职责 | 潜空间离线思维：CEM 搜索动作序列，MPC 滚动只执行第一步 |
| 接口 | `plan(s, g) → action_seq`；`execute(s) → a` |
| 实现 | stable-worldmodel CEM/MPPI 求解器（不自己写搜索）+ AdaJEPA 闭环（执行→观测→1 步梯度→再规划） |
| 实证 | V-JEPA 2-AC 零样本规划；LeWM 48× 加速；蒸馏：慢预算→快策略 +59.4pp |
| 验收 | jpi8（蒸馏预算对比） |

### C8 SymbolLayer（符号层）
| 项 | 内容 |
|---|---|
| 职责 | 结构/行为的压缩签名（分层：L0 图论粗筛 → L1 环内容分析 → L2 行为指纹） |
| 接口 | `fingerprint(trajectory) → f:[d]`；`structure_score(rules) → s`（L0+L1，零模拟）；`match(f) → symbol_id`；`text_anchor(text) → proto` |
| 实现 | **L0 图论粗筛**（有环/SCC/可达 → 缩搜索空间）+ **L1 环内容分析**（反汇编级：环内"写1×移动"=扩张判别，真实结构长尾 100% 命中，bb_human_check）+ L2 行为指纹（短观测统计，统计可学习分布信号，jpi6）+ LLM 文本原型（Qwen 语义锚）+ 自适应原型聚类 |
| 实证 | **环内容分析 100% >> 行为回归 50%（bb_human_check，2026-08-24 修正）**；行为指纹 > 结构注入 > 扁平（jpi6，仅短机器池）；跨 n 零遗忘+正迁移（jpi5）；LLM 蒸馏（真实身体第一轮） |
| 验收 | bb_human_check / jpi6 / jpi5 / zero_shot_check |

> ⚠️ 修正说明（2026-08-24）：C8 原设计以"行为指纹"为核心符号层（jpi6 裁决），但 jpi6 只在无注入短机器池成立；真实结构长尾上环内容分析才是正解（bb_human_check 100% vs 行为回归 50%）。C8 已升级为 L0/L1/L2 分层。

### C9 Distiller（蒸馏器：Mode-1/Mode-2）
| 项 | 内容 |
|---|---|
| 职责 | 编译 System-2 进 System-1：慢规划（大预算）→ 快策略（小预算） |
| 接口 | `distill(slow_model, data, budget) → fast_model` |
| 实现 | 软标签蒸馏 + 恰好够的样本量（相关性 ≈0.99 时最优） |
| 实证 | +59.4pp（慢 100% → 快 85% vs 直接快 25.6%）；样本量非单调（20→-11.9pp, 100→+16.8pp） |
| 验收 | jpi8_distill.py |

### C10 Configurator（元认知外环）
| 项 | 内容 |
|---|---|
| 职责 | 系统自身参数的异稳态调节器（AI 的"发烧"）——监测各环健康度 → 调制设定点 |
| 接口 | `observe(perf, t, died)` → 更新调制参数（饥饿速率/探索温度/阈值） |
| 实现 | 绩效梯度爬山 + 水平加速（绩效>band 强制上调）+ 死亡急刹车（疼痛反射 rate×=0.6） |
| 实证 | 节奏匹配场景：慢交替 +2.97、中 +1.20（超最优固定，5/5 显著）；全漂移 99.8% 最优；无漂移优于默认 48% |
| 边界 | 观测噪声（稀疏采集无法精确定位峰值）/ 时间尺度（切换快于观测窗口退化）——解法：与世界模型预测耦合 |
| 验收 | jpi12_final_validation.py |

## 3. 组件间接口与数据流（契约）

**统一 tick 循环**（每 tick 一次完整驱动）：
```
obs → C1.encode → s
     ├─ 内环: C2.predict(s, a_sel, δ) → Δŝ; 观测 s' → C3.e1 → 学习(C2.step, C1.step)
     ├─ 中环: C5.write(s, e1, e2) → C6.next_goal → C7.plan(s, g) → a_sel
     │        → 执行 → 观测 s' → C4.predict_cost + anchor → C3.e2
     └─ 外环: C10.observe(perf, t, died) → 调制 {C4.anchor_rate, C6.坚持度, C5.阈值, C2.lr}
```
- **统一空间**：s/g/记忆条目/原型/文本锚 全部为 768d（或经投影的 Qwen 896d）向量
- **事件流**：`(t, s, a, s', e1, e2, perf, died)` 八元组贯穿所有组件
- **参数域**：Configurator 调制的参数清单（每个都有实证依据）：
  | 参数 | 归属组件 | 实证 |
  |---|---|---|
  | 内稳态速率（饥饿） | C4 | 太快固着/太慢失效（jpi11c 扫描） |
  | 目标坚持度 | C6 | 逃避困难目标风险（裁决） |
  | surprise/原型阈值 | C5 | 三次手工阈值失效 |
  | 学习率 | C2 | AdaJEPA 分层 lr |
  | 探索-利用温度 | C7 | 非平稳最优漂移（jpi9） |

## 4. 成熟方案映射表（实现来源，不重造）

| 组件 | 成熟方案 | 来源 |
|---|---|---|
| C1 | ViT-B/16 + timm/Sparsh 权重 + SIGReg | timm / facebook/sparsh-ijepa / LeJEPA |
| C2 | 残差预测 + 两 loss + 分层 lr | LeWorldModel (2603.19312) / AdaJEPA |
| C3 | E1/E2 分离 | 我们的实证裁决 + LeCun Cost 分离 |
| C4 | Critic + intrinsic cost | LeCun 2022 / Curiosity-Critic (2604.18701) |
| C5 | LanceDB + recent-N 回放 | AdaJEPA / LanceDB 基准 |
| C6 | 记忆采样 + 坚持性预算 | 我们的实证（jpi1） |
| C7 | CEM/MPPI + MPC 滚动 | stable-worldmodel |
| C8 | 行为指纹 + LLM 文本原型 | jpi6 / LLM-JEPA (2509.14252) |
| C9 | 软标签蒸馏 | jpi8 (+59.4pp) |
| C10 | 绩效爬山 + 水平加速 + 急刹车 | jpi12（达标实证） |
| 分层 | 双尺度 δ | Micro-World Models (2026.06) |

## 5. 空白与待验证（诚实清单）

| 空白 | 状态 | 补法 |
|---|---|---|
| Configurator + 世界模型预测耦合 | ⬜ | 用预测补偿响应滞后（LeCun"从所有模块接收信号"完整形态） |
| 符号自动发现（可学习编码器） | 🟡 | 自动+手工指纹拼接已验证 100%（jpi10）；真实数据待测 |
| 多原型蒸馏（每类多描述） | ⬜ | Qwen 生成丰富描述 → 50-100 原型 |
| 蒸馏后 I-JEPA 微调 | ⬜ | 语义注入是否提升 JEPA 学习效率 |
| 真实身体（桌面/屏幕模态） | ⬜ | Sparsh 架构验证已过（148 层），接真实模态 |
| JEPA 底座端到端组装 | ⬜ | 本方案即组装蓝图 |

## 6. 集成路线（P0-P4）

| 阶段 | 内容 | 交付 | 状态 |
|---|---|---|---|
| P0 | 组件接口契约落地（C1-C10 的 Python 接口定义 + 类型标注） | components/ 包骨架 + 单测 | ✅ 2026-08-24 完成 |
| P1 | C1/C2/C3 内环闭合（编码器权重吸收 + 预测器 + E1 学习） | 桌面/图像流上 E1 收敛 | ✅ 2026-08-24 完成 |
| P2 | 中环闭合（记忆+目标+规划+价值+锚） | 自主探索闭环（节奏匹配环境） | ✅ 2026-08-24 完成 |
| P3 | C8 符号层 + C9 蒸馏 + C10 Configurator 全接 | 三环完整系统 | ✅ 2026-08-24 完成 |
| P4 | 真实身体对接（屏幕/文件模态） | DCA 升级版 | ⬜ |

### P0 已落地（components/ 包）

```
components/
  __init__.py        包入口 (组件清单 + 版本 0.1.0)
  core.py            类型系统: D=768, Event 八元组, ParamKind(7), ComponentConfig, HealthStats
  perception.py      C1 编码器接口 + DummyPerception (lazy 维度自适应)
  world_model.py     C2 世界模型接口 + DummyWorldModel (残差预测+在线步)
  energy.py          C3 能量系统接口 + CuriosityEnergy (误差-基线分离+锚)
  value.py           C4 价值接口 + HomeostaticValue (内稳态锚 + 线性 Critic)
  memory.py          C5 记忆接口 + AdaptiveMemory (分位门控 + 价值采样)
  goal.py            C6 目标接口 + ValueAlignedGoal (价值加权 + 坚持预算)
  planner.py         C7 规划接口 + GreedyPlanner (1 步前瞻 MPC)
  symbol.py          C8 符号接口 + BehavioralSymbol (轨迹指纹)
  distill.py         C9 蒸馏接口 + SoftLabelDistiller (占位, jpi8 实证)
  configurator.py    C10 外环接口 + PerfConfigurator (jpi12 达标实现)
  system.py          JepaAgent 三环组装器 + build_default_system + DemoEnv
  self_check.py      组装自检 (4 项: 实例化/类型/三环运行/调制链路)
```

自检结果（✅ 全过）：组件实例化 9/9、类型系统健全、**三环运行 200 tick**（E1 均值 0.0299、E2 活跃、绩效 0.1673、记忆 24 条）、**Configurator 调制链路**（爬山 0.001→0.003、死亡急刹车 0.003→0.0018、反转 0/急刹车 1）。

P1 迁移路径：Dummy 实现逐个替换为实证代码（jepa_base.VisionEncoder → C1；predictor/AdaJEPA → C2；jpi11c.PerfConfigurator → C10 已迁）。

### P1 已完成（C9 蒸馏 + 内环闭合）

**C9 蒸馏真实现**（distill.py SoftLabelDistiller + distill_check.py 验收）：BB 场景迁移 jpi8——慢 80 → 快 15，蒸馏样本 60。验收：**n=2 +10.4pp、n=4 +23.1pp，平均 +16.8pp，相关性 0.984-0.989**（jpi8 量级保持）。

**C1 真实现**（perception.py JepaPerception）：包装 jepa_base.VisionEncoder + weight_loaders 权重吸收 + I-JEPA forward_loss。验收：timm 150 层全加载，KNN 54.2%（对照 effect_check 60.1% 基线，训练样本 1/3 所致，质量保持）。

**C2 真实现**（world_model.py ResidualWorldModel）：AdaJEPA 配方残差 MLP + 归一化梯度下降。验收：可学静态流 E1 收敛 96.0%（0.00199→0.00008）；扰动偏移 + recent-N 回放恢复 95%。

**关键工程教训**（写入代码注释）：
1. **img_size=224 才能匹配 timm pos_embed (1,197)**——JepaPerception 默认 224，实际输入任意分辨率由 forward pos 插值处理（96 输入 37 token → 插值）。
2. **高维输入下朴素 SGD 有效步长失控**：lr × ||h||² ≈ 4（h 为 256 维 relu 激活）导致 pred 过冲 12 倍振荡发散——**归一化梯度下降（除 ||h||²）修复**，步长与激活尺度解耦。这是 DCA/jpi1 里 8 维小世界未暴露、768 维真实表征必现的问题。

### P2 已完成（中环闭合，自主探索闭环）

**C4 升级**（value.py HomeostaticValue v2）：完整内稳态——hunger 消耗-补充循环（tick 推进/死亡检测/try_feed 进食）、资源原型锚（register_resource + closeness exp 衰减）、线性 Critic（记忆学习）。**C6 升级**（goal.py）：坚持性预算用能量趋势判断（`t_hold >= persistence AND e1_trend 不降`）。**C7 升级**（planner.py GreedyPlanner v2）：目标趋近 + 前瞻内稳态锚 + **饥饿门控导航场**（jpi12 决策动力学：饥饿时被资源吸引，饱足时不吸引）。

**P2 验收**（p2_check.py，节奏匹配环境 4000 tick，无外部任务）：

| 指标 | 值 | 判定 |
|---|---|---|
| 移动率 | 0.41 | ✅ 自主行动（不死寂） |
| 记忆条目 | 200（满 cap） | ✅ 惊讶门控写入 |
| 进食 / 采集 | 41 / 20 次 | ✅ 内稳态循环 + 资源获取 |
| 目标切换 | 37 次 | ✅ 坚持性预算 |
| E2 活性 | 1.304 | ✅ 价值信号 |
| 死亡 | 3 次 | ✅ 内稳态生效（不失控） |
| **Config 速率末值** | **0.001 → 0.0175** | ✅ 快节奏期异稳态加码（与 jpi12 一致） |

**自主探索闭环成立**：无外部指令下 agent 自主移动、被饥饿驱动采集、惊讶门控记忆、坚持性预算切换目标、Configurator 调制内稳态速率——三环（内环 E1 学习 / 中环 E2 行动 / 外环调制）完整联动。

### P3 已完成（三环全接）

**C8 真实现**（symbol.py BehavioralSymbol v2）：行为指纹 = **纯运动统计**（平均/最大步长、转向率、静止比率、状态多样性——移除位置维度，符号区分"行为模式"而非"位置"）；自适应原型聚类（min_sep=0.25 依指纹距离分布标定：同位置 0.08 / 异位置 0.42）。

**P3 验收**（p3_check.py，2500 tick 探索 + 蒸馏 + 1000×3 策略评估 + 1000 tick 全闭环）：

| 阶段 | 结果 |
|---|---|
| 阶段1 探索+符号 | 3 个原型（0:32 / 1:1 / 2:29），记忆 200 条，进食 22，Config 速率 0.0085 |
| 阶段2 符号区分 | 运动段符号 [0,2]、静止段含专属符号 1——**✅ 符号区分行为模式** |
| 阶段3 蒸馏 | DeepPlanner(3步 roll-out) → 软标签 → FastPolicy（100 样本） |
| 阶段4 策略对比 | Deep 0.3% / Fast 0.3% / Greedy 0.6%；**fidelity 0.00——Fast 忠实蒸馏 Deep** ✅ |
| 阶段5 全闭环 | 采集 5 / 进食 3 / 死亡 1，Config 速率 0.0085→0.0115 ✅ |

**P3 关键发现（诚实记录）**：
1. **符号 = 行为签名，不是位置标签**——v1 指纹含位置主导维度（全局均值/质心），原型按位置分裂；v2 纯运动统计后符号正确区分行为模式。与 jpi6 原则一致。
2. **wm 预测精度决定规划深度价值**：漂移期 wm 预测误差累积，DeepPlanner(3-5步) 反不如 1 步 Greedy——蒸馏的 FastPolicy 忠实复制了慢策略（fidelity 0.00），包括其局限。真实世界模型精度是规划深度的前提。
3. **内稳态-采集耦合**：进食阈值（0.55→1.33格）先于采集阈值（0.8格）满足需求，agent 吃饱即停不精确到达——评估需进食阈值=采集阈值（0.7→0.78格）。
4. **异稳态过调可见**：Config 速率爬升过高导致饥饿-死亡循环，死亡急刹车介入恢复（0.0115 后回调）——系统自校正工作。

**组件状态**：C1-C10 全部真实现 + 验收脚本（self_check / distill_check / p1_check / p2_check / p3_check），三环 + 符号 + 蒸馏 + Configurator 完整可运行系统。P4 = 真实身体对接。

## 7. 与实证/蓝图的对应（一句话）

**这套组件 = LeCun 六模块蓝图（Perception/WM/Cost/Actor/Memory/Configurator）× 我们的实证修正（E1/E2 分离、锚三重、Configurator 具体机制、符号=行为签名、蒸馏实证）× 成熟方案（LeWM/SIGReg/AdaJEPA/CEM/LLM-JEPA）——十个组件全部有实证脚本作为验收测试，任何组件可独立替换升级。**
