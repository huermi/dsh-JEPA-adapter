# DCA × JEPA 底座重构：概念映射与实现方案 v1

> 日期：2026-08-24 ｜ 立场：以 JEPA 为第一性原理，重新审判 DCA 每个概念的存在合理性。
> 原则：**一切认知活动是统一空间中的条件预测；一切运行参数从预测误差中自组织涌现。**
> 注意：本文不是 JEPA×DSH（LLM 外壳工程）的延续，而是 DCA 认知体的概念级重构。

---

## 0. 一句话结论

DCA 在 JEPA 底座下 = **一个编码器 + 一个双向条件预测器 + 一个能量函数 + 一个记忆库 + 一个规划器 + 一个自指预测器**，全部活在同一个 128~256 维统一空间里。
**四个旧概念要被裁决：InfoDrives 三因子约化为能量的投影；Veto/Self-model 必须补硬编码锚；记忆不押注"反向预测生成"；目标切换要加坚持性预算。**

---

## 1. 底座定义：JEPA 五件套

| 底座件 | 数学形态 | 职责 |
|---|---|---|
| 编码器 fθ | s = fθ(obs)，obs∈{屏幕像素, 文件状态, 系统状态} | 观测 → 统一空间；丢弃不可预测细节（LeCun JEPA 核心） |
| 双向条件预测器 P | ŝ = P(s, a, δ, c)，δ∈ℝ 可正可负 | 未来（δ>0）、过去（δ<0）、任意条件 c 下的状态映射 |
| 能量函数 E | E = ‖ŝ − sg(s')‖²（训练）；E = ‖目标 − 预测‖²（规划）；E = ‖记忆 − 当前‖²（熟悉度） | 唯一内部信号：惊讶、触发学习、目标距离、记忆热度 |
| 记忆库 | 统一空间中的状态点集合 | 存储/索引/重放，生命周期由能量管理 |
| 规划器 | MPC：CEM 搜动作序列，min ΣE | 把"目标即向量"变成动作 |
| 自指预测器 Self-model | ŝ_self = P_self(s_self, a_self, δ) | 对自身状态（内存/速度/健康）做预测；元控制器 |

辅助件（有定理背书的必须项）：
- **辅助任务头**（behavior_predictor 的 JEPA 形态）：Yu et al. 2025 证明"潜在转移一致性 + 辅助回归头"联合训练 → 数学上无病态坍缩。它是表征的锚，不是附加功能。
- **SIGReg 高斯正则**：强制嵌入各向同性高斯（Cramer-Wold 随机投影 + Epps-Pulley 检验），单超参、~50 行、替代 EMA/手工特征工程。LeJEPA/LeWorldModel 已实证。
- **intrinsic cost 锚**：少量硬编码（危险记忆、健康基线），LeCun 蓝图本就保留不可训练的 intrinsic cost。

---

## 2. 概念映射总表

| # | DCA 旧概念 | JEPA 底座对应 | 实现方式（含可抄的 JEPA 生态） | 裁决 |
|---|---|---|---|---|
| 1 | InfoDrives（IG×PE×CE）唯一决策原则 | 能量 E 及其时间导数 | E=‖ŝ−s‖²；IG ≡ −dE/dt（信息增益 = 能量下降速率）；PE ≡ E（惊讶）；CE ≡ 正则/基线（SIGReg λ） | **约化**：三因子是 E 的三种投影，非三个独立信号 |
| 2 | ConvEncoder 8×8 像素特征 | 观测编码器 fθ | 8×8 块 token → 小型 ViT；防坍缩换 SIGReg | 保留，防坍缩简化 |
| 3 | JEPA 预测器（重设计版） | 条件预测器 P(s,a,δ,c) | 残差预测 Δŝ=g(s,a)；加 δ 条件（多尺度）；集成 4 个输出 mean+var | 保留，扩展 δ 双向 |
| 4 | ConsequenceSim 多分支后果模拟 | 潜变量多分支预测 + MPC | CEM：K=64×H=16 rollout + 可信度截断；参考 EB-JEPA `ac_video_jepa`（Two Rooms 规划，现成单 GPU 代码） | 保留，能量驱动 |
| 5 | Veto 机制（核心记忆相似度否决） | Guardrail 能量项 + 记忆原型距离 | 危险结局原型库 → 距离<τ 对候选动作加能量惩罚；参考 LeCun Guardrail Objective | **需补锚**：危险定义 = 硬编码 intrinsic cost |
| 6 | Outcome embeddings 8d + 原型聚类（符号涌现） | 统一空间涌现聚类 | SIGReg 高斯约束下训练；原型 = 低能量中心；符号 = 簇 id；参考 Koopman 不变量/慢特征分析 | 保留，分布约束化 |
| 7 | Self-model 5d（内存/速度/健康） | 自指预测器 + Configurator | 对内部状态做 JEPA 预测；健康基线 = 硬编码 intrinsic cost | **需补锚**：健康是外部基线，不是预测误差 |
| 8 | behavior_predictor | 辅助任务（防坍缩定理） | 辅助回归头 + 潜在转移一致性联合训练；Yu et al. 2025 | 保留，定理背书 |
| 9 | dist_active / gaze_familiarity | 空间方差活跃度 / −log E（熟悉度） | 纯统计诊断量，SIGReg 方差项即 dist_active 的控制端 | 保留为诊断工具 |
| 10 | Playground 256×256 / Sandbox | 观测源 + 隔离执行环境 | 环境通道，不占架构；Sandbox = 执行器隔离 | 保留，不进核心 |
| 11 | 社会学习（多实例） | 表征蒸馏 | 多实例共享目标编码器 / 蒸馏损失到 teacher | 保留 |
| 12 | Sim-Exec 循环 | MPC 闭环（测试时自适应） | 规划→执行→观测→**1 步梯度更新**→再规划；参考 AdaJEPA（recent-5 回放、最后几层、lr 5e-4/enc 1e-5） | 保留，抄 AdaJEPA 配方 |
| 13 | 七阶段能力暴露 | Configurator 阶段调制 | 配置器按阶段调制编码器/预测器/规划器的输入与注意力 | 保留 |

---

## 3. 四条关键裁决（交锋区）

### 3.1 InfoDrives 必须约化
IG×PE×CE 作为"唯一决策原则"，在 JEPA 底座下自相矛盾：底座只有**一个**信号（能量 E），IG 不是独立驱动力，而是 **−dE/dt**（能量下降的读数）。把 IG 独立成因子，等于在底座里偷偷塞回第二个评价体系。
- 新形态：`决策信号 = E（当前惊讶） + dE/dt（信息增益趋势） + λ·SIGReg（复杂度正则）`——后两项是 E 的导数和正则，不是并列因子。
- 你的 60k tick 观察里 dist_active 上升、familiarity 上升，本质都是 E 的统计量在演化，无需独立机制解释。

### 3.2 Veto 与 Self-model 必须补锚（构想三的老问题在此落地）
能量只回答"可预测吗/熟悉吗"，不回答"该否决吗/健康吗"。一个可预测但致命的状态（能量低）恰恰不该被选。
- Veto 的危险原型库、Self-model 的健康基线，必须来自**硬编码 intrinsic cost**（LeCun 的饥饿/疼痛类比：内存耗尽=疼痛、撞上已知危险结局=疼痛）。
- 这是底座允许的：LeCun 蓝图 = intrinsic cost（不可训练）+ critic（可训练）。DCA 此前拒绝任何外部信号，这正是规划失败与目标漂移的根源。

### 3.3 记忆不押注"反向预测生成"，押注"存储+检索+重放"
构想二（回忆=反向预测）在概念上优雅，但 BiJEPA（2026）实证双向对称训练会 representation explosion，且时间反演是 ill-posed 逆问题。
- **底座期**：记忆 = 存储（surprise 门控写入，E>τ 才存）+ 索引（能量相似度检索）+ 重放（AdaJEPA recent-5 式回放触发 1 步梯度）。
- **远期扩展**：等双向预测器稳定后，再尝试"回忆=反向任务"。

### 3.4 目标切换要加坚持性预算
"能量不再下降就换目标"会摧毁坚持性——能量不降可能是探索不足（局部极小），不是不可达。
- 新规则：`切换条件 = (E 持续不降 > T_坚持) AND (探索预算已耗尽)`。T_坚持 是硬编码下限（可随经验增长），防止系统永远逃避困难目标。

---

## 4. 新架构一轮 tick（数据流）

```
观测(屏幕/文件/系统) 
  → 编码器 fθ → s∈ℝ^128
  → 记忆检索: familiarity = −log‖s − 最近原型‖   (诊断: gaze_familiarity)
  → 目标选择: g = 记忆点 / 采样 / 插值 / 探索      (目标内化)
  → MPC: CEM K=64 × H=16 rollout 通过 P(s,a,δ,c)
       ├─ Veto 检查: 任何一步撞近危险原型 → 该候选 +惩罚
       └─ 选择 min ΣE 的动作序列, 执行第一步
  → 观测 s' → E = ‖ŝ−s'‖²
       ├─ E>τ → 写入记忆 (surprise 门控)
       ├─ 1 步梯度更新 P (AdaJEPA 配方: 最后层, lr 5e-4)
       └─ Self-model: 更新自身状态预测, 检查健康锚
  → 下一 tick (Sim-Exec 闭环)
```

所有学习信号 = E；所有超参 = 从 E 的统计量自适应（λ 除外，SIGReg 单超参）。

---

## 5. 落地路线（先诊断，再改）

| 阶段 | 动作 | 验证 |
|---|---|---|
| R0 诊断基线 | 在现有 DCA 上接诊断工具：把 IG/PE/CE 分别与 E、dE/dt 的相关系数算出来 | 若 IG 与 −dE/dt 相关 >0.9 → 坐实约化裁决 |
| R1 底座替换 | 防坍缩换 SIGReg；Outcome embeddings 加高斯约束 | 原型聚类稳定性 vs 现在 |
| R2 闭环 | 抄 AdaJEPA 的 1 步梯度闭环进 Sim-Exec | 分布偏移下规划成功率 |
| R3 锚定 | 引入 intrinsic cost（危险原型、健康基线） | Veto 触发率、Self-model 预警准确率 |
| R4 双向 | 受限双向预测（单步反演）作为远期扩展 | 回忆质量 vs 检索基线 |

---

## 附录：关键参考

- LeCun 2022 position paper（intrinsic cost + Configurator 蓝图）
- Balestriero & LeCun, LeJEPA (arXiv:2511.08544) + SIGReg 代码 galilai-group/lejepa
- Maes et al., LeWorldModel (arXiv:2603.19312) + 代码 lucas-maes/le-wm
- Terver et al., EB-JEPA (arXiv:2602.03604) + 代码 facebookresearch/eb_jepa
- Wang et al., AdaJEPA (arXiv:2606.32026) + 代码 agentic-learning-ai-lab/adajepa
- Yu et al., Auxiliary Tasks Improve JEPA (arXiv:2509.12249) —— 防坍缩定理
- Huang, BiJEPA (arXiv:2603.00049) —— 双向不稳定实证
- Destrade et al., Value-Guided JEPA (arXiv:2601.00844) —— 纯距离规划失效实证
