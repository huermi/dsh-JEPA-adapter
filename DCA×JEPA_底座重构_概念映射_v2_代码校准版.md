# DCA × JEPA 底座重构：概念映射 v2（真实代码校准版）

> 日期：2026-08-24 ｜ v2 变更：基于 `D:/workbuddyproject/20260327171412/dca4` 真实代码逐条校准 v1 裁决。
> 读码范围：main.py（tick 循环）、cognition/info_drives.py、prediction/predictor.py、planning/consequence.py、memory/pools.py、encoding/sparse_proj.py、core/config.py。

---

## 0. 一句话结论（v2）

**代码现状比预期更接近 JEPA 底座**：残差预测、目标归一化防坍缩、AdaJEPA 分层学习率 + 单步梯度 + 局部缓冲回放、反事实因果检验、Veto、5 层记忆——这些"该从 JEPA 生态吸收的"已经在 Predictor/InfoDrives/ConsequenceSim 里实现了。
**真正还差的三件事**：① 统一空间没统一（现在是 512d 双通道拼接 + JL 稀疏投影，不是学出来的 JEPA 空间）；② Self-model 是手搓监控向量，不是自指预测器；③ InfoDrives 公式内部有冗余（IG 与 PE 高度共线），且缺 SIGReg 式分布约束。

---

## 1. v1 裁决 × 代码事实 × v2 更新（核心校准表）

| # | v1 裁决 | 真实代码事实（文件:行） | v2 更新 |
|---|---|---|---|
| 1 | InfoDrives 三因子应约化为能量投影 | **IG 确实是 PE 的单调函数**：`IG(a) = 1/(1+avg_PE) − 1/(1+base_PE)`（info_drives.py:65-81），且 `V(a) = IG × (1−PE_norm) × CE × temp`（info_drives.py:352-382）——**IG 和 (1−PE_norm) 都是 PE 的递减函数，乘法后 PE 被平方加权，两因子冗余**，多样性实际只来自 CE | **坐实 + 加强**：三因子公式有内部冗余，IG×(1−PE) 应合并为单一 PE 项，真正需要的是 CE（探索）与 PE（利用）的对立 |
| 2 | ConvEncoder 8×8 → 编码器 fθ | conv_encoder.py 8×8 特征 + sparse_proj.py 稀疏随机投影 12544→256（JL 引理保距，±1 权重，非学习） | **部分校准**：空间投影是固定的（非学习），保留近似距离但**不是 JEPA 的"可丢弃不可预测细节"的编码器**——它不丢弃，只压缩 |
| 3 | 预测器保留 + 扩展 δ 双向 | predictor.py 已实现 `predict_delta=True`（残差预测:59）、`normalize_target`（目标归一化防坍缩:145）、**AdaJEPA 全套**：`adapt_step` 分层 lr（encoder 0.001 / predictor 0.05:57-58,211-267）、`adapt_from_buffer`（recent-N 采样:269-281）、`causal_effect`（反事实检验 predict_without_action:318-361） | **坐实 + 重要校准**：AdaJEPA 配方**已内嵌**（代码注释明写"AdaJEPA 借鉴参数"），不是待办。缺：δ 条件（多尺度时间）、双向预测、EMA/SIGReg 防坍缩 |
| 4 | ConsequenceSim → MPC 规划 | consequence.py 已接入主循环（main.py:272-275 调用 simulate）；`_forward_simulate` 是**单步/衰减模拟**（n_sim_steps, decay 0.8），`compare_branches` 做 WAIT 基线反事实对比（:71-90） | **部分校准**：ConsequenceSim 存在且接入，但是"1 步前向 + 分支对比"，**不是真正的多步 MPC rollout（K 候选 × H 步）**。要成为 JEPA 规划器需补 CEM 搜索 |
| 5 | Veto 需补锚 | veto_check 已实现且**已接入主循环**（main.py:472, 596）：L1=预期 PE 超阈值（压缩保护）、L2=预测状态不匹配核心记忆（retrieve_core_similar）；有 veto_count 统计（main.py:520） | **校准**：Veto 已存在，L1 是能量锚（PE 阈值=硬编码），L2 是记忆锚（核心记忆相似度）。v1 说"需补锚"——代码里锚是**阈值常数 + 核心记忆**，不是 intrinsic cost 表。可用但"否决什么"仍无价值语义 |
| 6 | Outcome embeddings 8d + SIGReg | predictor.py W3 输出层 `outcome_dim=8`（:47-51,105-125）；info_drives.py `outcome_archetypes` 聚类（余弦<0.7 建新原型，上限 20:399-408） | **部分校准**：8d 后果嵌入 + 原型聚类已实现，但**无分布约束**（非高斯、无方差/协方差正则），原型靠手工阈值，不是 SIGReg 式涌现 |
| 7 | Self-model 需补锚 | main.py `_build_self_state` 返回 5d：[内存, tick耗时, W1范数, 活力, 压缩进展]（main.py:437-457），输入 Predictor 的 self_dim=5 | **重要校准**：Self-model 5d 是**手搓监控向量**（psutil 内存、perf_counter、权重范数），**不是自指预测器**——它不预测自身未来状态，只是把当前健康读数拼进状态。距"自指 JEPA"还差一个"预测自身状态"的闭环 |
| 8 | behavior_predictor（防坍缩定理） | 未在 main.py 主循环中发现独立 behavior_predictor；社会学习相关在 social/ 与 experiments/ | **需确认**：辅助任务头（Yu 2025 防坍缩定理）在主循环中未显式出现，Predictor 只有 outcome 头 |
| 9 | 记忆库（存储/索引/重放） | memory/pools.py 5 层池：work/self/core/episodic/procedural；decay 衰减（SELF_MEM_DECAY=0.9999）+ 低显著性剔除（<0.3 遗忘:87）+ 晋升 core（salience 百分位）+ 溢出 episodic；consolidation/replay.py ReplayConsolidation 存在 | **坐实**：记忆生命周期已按显著性（≈能量代理）管理，与"低能量淡出"一致。缺：surprise 门控写入（当前是 salience 阈值） |
| 10 | Playground/Sandbox | config.py: PLAYGROUND_RES=768, SANDBOX 动作集（a-z/0-9/space）；main.py `_desktop_env` 虚拟桌面路由 | **坐实** |
| 11 | 社会学习 → 蒸馏 | social/ 目录存在；memory/pools.py 有 `_entry_origin` 记忆来源追踪（多智能体:45-47） | 存在雏形，未见蒸馏损失 |
| 12 | Sim-Exec 闭环 | tick() 已实现完整闭环：capture→encode→drive→recall→simulate→action→learn（main.py:178-317） | **坐实**：闭环已存在，且 _learn_from_feedback 用空间差分反馈（自我通道 224×224） |

---

## 2. 三个真正的差距（v2 新增，比 v1 更尖锐）

### 2.1 统一空间没有"统一"
当前状态向量 = `STATE_DIM=512`，是**桌面通道 256d + Playground 通道 256d 的拼接**（config.py:14），且来源是稀疏随机投影（固定的 ±1 矩阵，JL 保距）——**不是学习出来的 JEPA 空间**。
- 后果：桌面和 Playground 是**两个独立的几何**，Predictor 学的是拼接空间的动力学，目标/记忆/状态在拼接空间里没有统一的"可预测性结构"。
- JEPA 底座要求：一个编码器 fθ，把两类观测映射到**同一个学习空间**，可预测性决定保留什么。JL 投影保留的是距离，不是可预测性——距离近 ≠ 动力学相似。

### 2.2 Self-model 不是自指预测器
5d 是"当前健康读数"，不是"对自身未来状态的预测"。自指 JEPA 应该是：`ŝ_self = P_self(s_self, a_self, δ)`，用自身状态做目标，能量 = ‖ŝ_self − s_self'‖²。现在的实现是"把传感器读数拼进输入"，是**感知不是自指**。

### 2.3 InfoDrives 公式冗余 + 无分布约束
`V = IG × (1−PE_norm) × CE × temp` 中 IG 与 (1−PE_norm) 共线（都随 PE 单调减），相乘把 PE 的惩罚平方化，等价于 `V ≈ (1−PE_norm)² × CE × temp`——**三公理实际是两因子**。且状态/后果嵌入无方差-协方差或高斯约束，坍缩防护靠手工阈值和范数裁剪（predictor.py:183-191），不是分布级约束。

---

## 3. 修正后的落地路线（v2，基于代码现状）

| 阶段 | 动作 | 对应代码改动 | 验证 |
|---|---|---|---|
| R0 诊断（保留） | 算现有 IG 与 (1−PE_norm) 的相关性，坐实冗余 | 不改代码，纯读诊断 | 相关 >0.9 → 合并因子 |
| R1 统一空间 | **去掉 512d 拼接**，改为单编码器 fθ（ConvEncoder 输出 + 可学习投影）映射到单一 s∈ℝ^256 | 改 main.py 状态构造 + sparse_proj 换成可学习投影 | 桌面/Playground 状态可交叉预测（跨通道预测误差下降） |
| R2 自指闭环 | Self-model 从"读数拼接"升级为"自指预测器" | 新增 P_self，预测自身 5d 的下 tick 值 | 自身状态预测误差可诊断 |
| R3 InfoDrives 修正 | 合并 IG×(1−PE_norm) → 单一 PE 项；V = PE_term × CE × temp | 改 info_drives.py get_action_drive | 行为多样性不降（CE 已承担） |
| R4 SIGReg/分布约束 | 给 outcome 8d + 状态空间加高斯约束 | 移植 lejepa SIGReg（~50 行） | 原型聚类稳定性 |
| R5 MPC 多步 | ConsequenceSim 从 1 步模拟升级为 CEM K×H 搜索 | 参考 EB-JEPA ac_video_jepa | 规划成功率 |

---

## 4. 与 v1 的关键差异（一句话版）

- v1 说"抄 AdaJEPA 配方" → **代码已实现**（adapt_step/分层lr/缓冲/反事实因果全在 predictor.py）。
- v1 说"Veto 需补锚" → **Veto 已接入主循环**，锚是 PE 阈值 + 核心记忆相似度（阈值型，非 intrinsic cost 表）。
- v1 说"三因子约化" → **代码坐实且更严重**：IG 与 (1−PE) 共线，公式实际是两因子。
- v1 未覆盖 → **统一空间是拼接的、Self-model 是手搓的**——这两个才是真差距。
