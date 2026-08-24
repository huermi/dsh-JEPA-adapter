# 以 JEPA 为基础的持续学习框架（CL-JEPA）：研究谱系与可行方案

> 整理人：知微 🪄 ｜ 日期：2026-08-17 ｜ 用途：为"双系统辩证引擎 / 持续世界模型"项目提供 JEPA 侧的文献底盘与一个可实现的持续学习（Continual Learning, CL）框架草案。
>
> 说明：本文不重复造轮子——JEPA 家族本身已经自带 EMA 慢速目标编码器与不对称预测器，这两件正是抗遗忘的天然结构。本文的核心论点：**JEPA 是持续自监督学习（CSSL）的天然基底，把现有 CSSL 武器库"翻译"到 JEPA 的三件套上即可得到一套可行框架**。

---

## 0. 摘要（TL;DR）

1. **JEPA 不是某个模型，而是一类"在表征空间而非像素/词元空间做预测"的训练目标**（LeCun 2022 蓝图 + I/V/VL/LLM-JEPA 等具体实现）。
2. 表征空间的预测目标让**编码器可以自由丢弃不可预测的细枝末节**，学到的表示更语义化、更可迁移、更抗过拟合——这与"抗遗忘"在结构上同源。
3. 已有充分证据证明 **SSL 模型本身比有监督模型更抗遗忘**（CaSSLe, CVPR 2022）；且 JEPA 自带的 **EMA 目标编码器 = 免费的"动量知识蒸馏"慢锚**，不对称 **预测器 = 可复用的对齐投影头**。
4. 提出 **CL-JEPA**：以"正则（EMA-MKD + 潜空间蒸馏）+ 回放（样例 + 潜嵌入）+ 架构（Adapter/LoRA + 预测器协同训练）"三层防御，配合"在线流 + 任务边界"双循环，并引入 Clin-JEPA 的"rollout 漂移"作为世界模型专属评估指标。
5. **当前缺口**：截至检索，尚无直接以"Continual JEPA"为题的发表工作；但 Clin-JEPA（EHR 轨迹）、TS-JEPA（时序）、CNN-JEPA 已把 JEPA 推向流式/轨迹场景，其稳定性手段可直接转为 CL 手段。

---

## 1. JEPA 研究谱系

| 年份 | 工作 | 域 | 关键贡献 | 出处 |
|---|---|---|---|---|
| 2022 | **A Path Towards Autonomous Machine Intelligence** | 理论 | JEPA 蓝图；六模块（感知/世界模型/代价/记忆/动作/配置器）；H-JEPA 分层世界模型 | LeCun, OpenReview |
| 2023 | **I-JEPA** | 图像 | 首个具体 JEPA；多块大掩码逼迫语义预测；无数据增强；ViT-H/14 线性探针 79.7%（MAE 68.0%），<1200 GPUh | Assran et al., CVPR 2023, arXiv:2301.08243 |
| 2023 | **MC-JEPA** | 视频 | 内容 + 运动双表征联合学习 | Bardes, Ponce, LeCun, arXiv:2307.12698 |
| 2024 | **V-JEPA** | 视频 | ~90% 时空掩码 + L1 + EMA teacher；冻结骨干 K400 81.9% / SSv2 72.2% / ImageNet 77.9%（仅视频预训练） | Bardes et al., arXiv:2404.08471 |
| 2025 | **V-JEPA 2** | 视频/机器人 | 十亿参数级；ViT-L(300M)+预测器 ViT-S(~22M)；3D-RoPE；VideoMix22M(~2200万视频≈100万小时)；零样本机器人规划；**V-JEPA 2-AC** 动作条件版（仅 62h 机器人数据） | Assran et al., arXiv:2506.09985 |
| 2025 | **VL-JEPA** | 视觉-语言 | 预测连续文本嵌入而非自回归解码；解码成本降 ~2.85×，可训练参数减半 | Meta AI, arXiv:2512.10942 |
| 2025 | **LLM-JEPA** | 语言 | 把 JEPA 扩展到 LLM：多视角（文本/代码）配对；[PRED] token 权重绑定预测器；超参 λ、k | Huang, LeCun, Balestriero, arXiv:2509.14252 |
| 2025 | **CNN-JEPA** | 图像 | 把 JEPA 适配 CNN（ResNet-50），分割/分类数据效率优于 I-JEPA | Kalapos, Gyires-Tóth, CVIU, doi:10.1016/j.cviu.2025.104595 |
| 2025 | **TS-JEPA** | 时间序列 | 时序表征学习；分类+预测双任务均衡 | Ennadir et al., arXiv:2509.25449 |
| 2026 | **GMM-JEPA** | 语音 | 冻结 GMM 软聚类锚点防表征坍缩 | arXiv:2602.09040 |
| 2026 | **Clin-JEPA** | EHR 轨迹 | 五阶段协同训练（predictor warmup → joint refinement → EMA target alignment → hard sync → predictor finalization）稳定自回归 rollout；48h 潜空间 rollout 漂移收敛 −15.7% | Yang et al., arXiv:2605.10840 |

### 1.1 JEPA 核心三件套与损失

```
        x (可见上下文)                y (被掩码/未来) 
            │                            │
     Context Encoder f_θ  ──┐      Target Encoder f_ξ (EMA, stop-grad)
     (梯度更新)             │      (慢速跟随, 不回传梯度)
            │              │            │
        s_x = f_θ(x)       │       s_y = sg(f_ξ(y))
            │              │            │
            └──── Predictor g_φ ────────┘
                  (轻量, 不对称)
                        │
                  ŝ_y = g_φ(s_x, z)      z = 目标位置等辅助信息
                        │
              L_JEPA = D( ŝ_y , sg(s_y) )      # L2 / L1 / 余弦
```

- **防坍缩三件套**：EMA（目标编码器慢跟随）+ stop-gradient（目标不回传）+ 预测器不对称（唯一可学习的"间隙"）。
- **为什么天然利于 CL**：损失在表征空间 → 编码器可丢弃不可预测细节；EMA 目标编码器本身就是"过去知识的慢速快照"；预测器本身就是"把一种潜表示映射到另一种潜表示"的模块——这两点恰是下文 CL 手段的现成零件。

---

## 2. 持续自监督学习（CSSL）现有武器库

| 方法 | 类别 | 机制 | 关键结论 |
|---|---|---|---|
| **CaSSLe** (CVPR 2022) | 正则/蒸馏 | "SSL 即蒸馏"：冻结上一任务模型，加预测器把当前特征映射到旧特征空间，复用原 SSL 损失做时序蒸馏 | 类增量 CIFAR100 +6.8%、ImageNet100 +4%、DomainNet +4.4%；提出 **Mappability Principle**（旧信息保持"可映射"而非"完全相同"，保留可塑性） |
| **PFR** (2022) | 正则 | Predictive Feature Regularization，负余弦对齐 | CaSSLe 前身之一 |
| **OSIRIS** (2024, arXiv:2404.19132) | 统一框架 | 把 CaSSLe/PFR/ER/ER+/ER++/DER 统一为 `L = L_SSL + λ1·L_cross + λ2·L_past` | 显式区分"跨任务整合"与"过去保持"两项 |
| **ER / ER+ / ER++** | 回放 | 记忆缓冲；ER++ 在当前批+记忆批上做全 SSL | 回放隐式允许发现新特征，改善跨任务整合 |
| **DER / DER++** | 回放+蒸馏 | 在输出空间对"存入时刻的特征"做 L2 正则 | 经典强基线 |
| **CLA** (CoLLAs 2025, arXiv:2507.10434) | 正则(在线) | **EMA 网络对齐**：用 EMA 副本作对齐目标，无需任务边界；CLA-b(无回放)/CLA-E(EMA目标)/CLA-R(回放) | 在线 CSSL 关键：动量知识蒸馏（MKD）应作为核心组件；减小特征漂移、提升反向迁移 |
| **GLARE** | 架构 | 冻结骨干，只训 Adapter；动量编码器 + 全局/区域/局部三级一致性 | 持续预训练提升分割 |
| **MedCoSS** (CVPR 2024) | 多模态回放 | 医学多模态持续 SSL， rehearsal-based | 通用多模态表征 |
| Branch-Tuning / InfoUCL / CroMo-Mixup / Pseudo-Negatives | 各类 | 稳定性-可塑性平衡、信息最大化、跨模型混合、伪负例 | 2024 年集中爆发 |
| **Awesome-CSSL** (github) | 清单 | jhairgallardo/awesome-continual-self-supervised-learning | 论文全景 |
| **Survey** (2025, arXiv:2607.09785) | 综述 | "Lifelong Representations: A Survey on Continual Self-Supervised Learning for Vision Models" | 系统梳理 |

### 2.1 可直接迁移到 JEPA 的三条经验

1. **SSL 比有监督更抗遗忘**（CaSSLe 实证）——JEPA 作为 SSL 的"表征空间"极端版，理论上更稳。
2. **EMA 对齐 = 在线抗遗忘的免费午餐**（CLA）——而 JEPA 本来就有 EMA 目标编码器。
3. **Mappability > Identity**（CaSSLe）——不需要冻结旧潜空间，只需保持"可映射"，这正契合 JEPA"预测潜表示"的本质。

---

## 3. 关键映射：为什么 JEPA 是天然好的 CL 基底

| JEPA 既有结构 | 在 CL 中的角色 | 对应 CSSL 手段 |
|---|---|---|
| EMA 目标编码器 f_ξ | 稳定性慢锚（= CLA 的 EMA 网络 θ′） | Momentum Knowledge Distillation |
| 不对称预测器 g_φ | 复用为"当前潜→过去潜"的对齐投影头 | CaSSLe 的 predictor/projector |
| 表征空间损失 L_JEPA | 直接当蒸馏损失用（一致目标，零额外超参思想） | CaSSLe "SSL-as-distillation" |
| 非生成式（不重建像素） | 回放可行"潜嵌入"而非"像素"，内存固定且便宜 | DER 式 latent replay |
| 预测器在 V-JEPA 2-AC 中保留用于规划 | 持续 rollout 需防漂移 → Clin-JEPA 五阶段课 | rollout 稳定性训练 |

> **辩证小结（稳定性 vs 可塑性）**：CL 的根本矛盾是"记住旧"与"学会新"的对立统一。JEPA 的结构恰好把这对矛盾**内置**了——EMA 目标编码器代表"稳定的过去"（正题），当前上下文编码器代表"流动的新知"（反题），预测器在两者间不断逼近（合题）。因此 CL-JEPA 不是外挂遗忘防护，而是把 JEPA 自身的训练动力学直接用作 CL 机制。

---

## 4. 提出的框架：CL-JEPA

### 4.1 三层防御

**P1 正则层（Regularization）— EMA-MKD + 潜空间蒸馏**
- 复用 JEPA 的 EMA 目标编码器作为**在线慢锚**（CLA-b 思想，无需任务边界）。
- 任务边界处：快照 `{f_θ, f_ξ}` 并冻结；新增 CaSSLe 式预测器 `h_ψ`，最小化
  `L_distill = L_JEPA( h_ψ(f_θ(x)) , sg(f_ξ^{t-1}(x)) )`
  即"用同一种 JEPA 损失把当前潜表示蒸馏回旧潜表示"，保持可映射性而非恒等。
- 长周期再加一个**慢速 EMA 网络 θ′**（比 f_ξ 更慢）做 CLA 式持续对齐。

**P2 回放层（Replay）— 样例 + 潜嵌入双缓冲**
- **样例缓冲**（ER++）：存少量原始样本，当前批 ∪ 记忆批做全 JEPA SSL，隐式促进跨任务整合。
- **潜嵌入缓冲**（DER 式）：存入样本时顺手存下 `f_ξ(x)`，回放时只对固定大小潜向量做 L2 正则——**内存恒定、不随分辨率和模态膨胀**，对视频/3D 尤其关键。
- 世界模型专属：额外存 `(s_context, s_future)` 潜对，支持"潜空间 rollout 回放"。

**P3 架构层（Architecture）— Adapter/LoRA + 预测器协同训练**
- 新任务冻结骨干，只训轻量 Adapter（GLARE 思想）→ 参数隔离，骨干零遗忘。
- 若需持续规划（世界模型）：保留预测器并按 **Clin-JEPA 五阶段课**协同训练，抑制自回归 rollout 漂移：
  `predictor warmup → joint refinement → EMA target alignment → hard sync → predictor finalization`

### 4.2 双循环训练

```
┌─────────────── 在线流循环（无任务边界）───────────────┐
│  每 mini-batch:                                       │
│   L = L_JEPA(current)                                │
│     + ω1 · L_align( f_θ , EMA θ′ )        # CLA-b    │
│     + ω2 · L_replay_latent( buffer )      # DER式    │
│  更新 EMA θ′ , EMA f_ξ                                │
└──────────────────────────────────────────────────────┘
            ↓ 检测到任务边界 / 新模态接入
┌─────────────── 任务边界循环 ──────────────────────────┐
│  1. 快照冻结 {f_θ, f_ξ}^{t-1}                         │
│  2. 扩展 Adapter / 新预测器 h_ψ                       │
│  3. L = L_JEPA(new) + λ·L_distill(CaSSLe) + replay    │
│  4. 若为世界模型：跑 Clin-JEPA 五阶段课稳定 rollout    │
└──────────────────────────────────────────────────────┘
```

### 4.3 评估协议（含世界模型专属指标）

| 指标 | 含义 | 来源 |
|---|---|---|
| 反向迁移 BWT（遗忘量） | 旧任务性能下降 | 标准 CL |
| 正向迁移 FWT | 学新任务时旧知识助力 | CaSSLe |
| **潜空间 rollout 漂移** | 自回归 rollout 48h 潜表示发散度 | Clin-JEPA（−15.7% 为收敛基准） |
| Mappability | 当前潜→旧潜 线性/MLP 可映射误差 | CaSSLe 原理 |
| 回放预算 | 单位内存下的保持率 | ER/DER |

---

## 5. 实施路线与首次验证实验

**阶段 0（可行性，2–3 周）**
-  backbone：I-JEPA 或 V-JEPA 预训练权重（facebookresearch/ijepa）。
-  场景：类增量 ImageNet100 / 域增量 DomainNet。
-  对照：plain fine-tune / ER / CaSSLe-on-JEPA（把 CaSSLe 预测器直接挂到 JEPA 潜空间）。
-  预期：CaSSLe-on-JEPA 应优于 plain fine-tune；验证"JEPA 潜空间可直接当蒸馏目标"。

**阶段 1（在线化）**
-  引入 CLA-b EMA 对齐 + 小潜缓冲，去掉任务边界依赖；对标 CLA 论文曲线。

**阶段 2（世界模型持续化）**
-  backbone：V-JEPA 2-AC。
-  场景：SSv2 / K400 持续动作预测；跟踪 rollout 漂移（Clin-JEPA 指标）。
-  关键：预测器按五阶段课持续协同训练，避免 rollout 发散。

**阶段 3（双系统辩证引擎对接）**
-  把 CL-JEPA 作为"感性/世界模型"一侧，与用户既有的"辩证推理"系统对接：JEPA 提供可预测的世界状态潜流，推理系统在其上做反题-合题推进；持续学习保证两侧在世界演化中都不遗忘。

---

## 6. 开放风险与待解问题

1. **尚无直接发表**：检索未见"Continual JEPA"原题工作；本文框架为合理综合，需实验验证。
2. **EMA 动量调度**：CL 下慢锚的 τ 需在"稳定"与"可塑性"间重调（CL 比原训练更敏感），建议 τ 随任务数退火。
3. **配对/掩码视图成本**：LLM-JEPA 的 3× 训练开销、JEPA 需掩码或多视角，对持续流不友好；可用在线随机掩码缓解。
4. **在线 vs 任务增量**：CLA 表明无任务边界时小 batch 是瓶颈，回放几乎必需；纯正则（CLA-b）alone 不够。
5. **记忆预算**：视频/3D 下样例回放贵，潜嵌入回放（固定大小）是更优解，但需验证其跨任务整合能力弱于样例回放。
6. **评测缺失**：世界模型持续学习尚无标准 benchmark，rollout 漂移需自建。

---

## 7. 参数持续更新机制（内存/算力有界）

> 本节直接回答一个更尖锐的问题：**不做一次性预训练固定，而是每来一点新数据就动一次权重，且内存与算力占用都不随任务数增长，该怎么更新参数？**

### 7.1 先把"不可行"的两条路排除
- **(a) 每次全量重训**：你把历史全跑一遍才更新——这正是要回避的"预训练一次性固定"的反面，且算力随历史线性爆炸。
- **(b) 专家库/参数隔离无限扩张**：PackNet、MoE、分支扩展让参数量随任务线性增长——直接违背"有限内存"。

可行路只有一条：**单份在线权重 + 至多一份慢速平均副本 + 固定大小回放缓冲**，每步只做 O(minibatch) 运算。

### 7.2 有界内存预算分解
| 项 | 占用 | 是否随任务增长 |
|---|---|---|
| 在线上下文编码器 f_θ | 1× 模型 | 否（同一份） |
| 慢速长周期锚 θ′（EMA，比 f_ξ 更慢） | 1× 模型 | 否（仅一份移动平均） |
| 目标编码器 f_ξ（EMA） | 训练内复用，可并入 θ′ | 否 |
| 预测器 g_φ + CaSSLe 投影头 h_ψ | ≪1× 模型（2 层 MLP） | 否 |
| 回放缓冲 M：原始样例 | 固定 K（如单任务 1–5%） | 否（环形覆盖） |
| 回放缓冲 M：潜嵌入 z* | K × 嵌入维（≈KB/样本） | 否（固定大小） |

**结论**：参数侧峰值 ≈ **2.0–2.1× 单模型**（online + 1 个 EMA 长锚 + 极小投影头）；数据侧为固定 K。二者均与任务数无关 → 满足"有限内存"。

### 7.3 有界算力：每步只做 O(minibatch)
- 每个流步仅一次 前向+反向（当前批 ∪ 少量回放），计算量 O(batch)，**与历史总长无关**——这是 streaming/online CL 的定义。
- 目标编码器 f_ξ 走 EMA、**不回传梯度**；慢锚 θ′ 也只做加权移动平均（几乎零算力）。
- 反向只作用于 f_θ 与 g_φ（+Adapter）；可在 f_θ 上挂 LoRA 进一步削显存与更新算力。
- **关键诀窍**：JEPA 的目标本就在潜空间，所以"潜回放"是原生一等公民——你连像素都不必重建，回放一个 768 维向量即可做 DER 式 L2 钉定，内存比存原始帧小 2–3 个数量级（对视频/3D 尤其关键）。

### 7.4 单步更新算法（可直接落代码）
```
θ_online, θ_target(=EMA θ_online), θ_anchor(=更慢的 EMA)
h_ψ : 极小投影头(CaSSLe)            M : 固定缓冲 {(x, z*)}
for x_t in stream:
    z_on = Enc_online(x_t)                       # 梯度
    z_tg = Enc_target(x_t)                       # EMA, stop-grad
    z_pr = Pred(z_on)                            # 梯度
    L_task   = dist(z_pr, sg(z_tg))              # = L_JEPA
    (x_r, z*_r) = sample(M)                      # 潜回放
    L_replay = ||Enc_online(x_r) - z*_r||^2      # DER 钉定, 防漂移
    L_anchor = dist(h_ψ(Enc_online(x_t)), Enc_anchor(x_t))  # CLA/CaSSLe
    L = L_task + λ_r·L_replay + λ_a·L_anchor
    backward(L); step();  EMA 更新 θ_target, θ_anchor
    store (x_t, Enc_online(x_t)) into M           # 以存潜嵌入为主
# 任务边界(可选, 仅 1 份快照):
    freeze {f_θ, f_ξ}^{t-1}; 扩展 Adapter;  以 CaSSLe 蒸馏项继续
```
- **无限漂移防护**：纯 EMA 锚本身会缓慢漂移，极旧知识仍会渗漏。对策二选一（不必都做）：**周期性 re-anchor**（把 θ_anchor 重置为当前 θ 并继续），或仅在任务边界保留**一份固定快照**（峰值仍 ≤2 份，不增长）。

### 7.5 内存/算力有界下的折中表
| 手段 | 内存代价 | 算力/步 | 抗极旧遗忘 | 备注 |
|---|---|---|---|---|
| 纯在线 EMA 对齐(CLA-b) | 2× 模型 | O(batch) | 中（锚会漂移） | 最简，无回放 |
| + 潜回放(DER 式) | 2× 模型 + K·dim | O(batch) | 高 | **推荐基线** |
| + 任务边界 1 快照(CaSSLe) | ≤2× 模型 + 1 快照 | O(batch) | 最高 | 边界可检测时最优 |
| Adapter/LoRA 冻结骨干 | 1× 骨干 + 小 Adapter | O(batch) 更低 | 骨干零遗忘 | 可塑性受限 |
| 样例回放(ER++) | 2× 模型 + K·帧 | O(batch) 略高 | 高 | 视频下贵，用潜回放替代 |

### 7.6 辩证小结（权重如何"流动而不溃散"）
一次性预训练把权重焊死（静止=正题的极端）；朴素 SGD 持续更新则让权重失控漂移（纯粹流动=反题）。CL-JEPA 的更新律是**合题**：用 EMA 长锚把"过去的权重"以缓慢平均持续注入当下，用潜回放把"过去的表征"以固定向量钉在当前流上，用 CaSSLe 投影头保持"旧潜空间对当前可映射"。于是参数在流动中始终保有一个可回溯的重心——这正是"在有限占用下持续学，而非一次性固定"。

### 7.7 人类"参数更新"的生物映射（数学对应）
把"人如何更新自己的参数"逐条映射到 JEPA 的数学结构。**先立一个反直觉的纠正**：人并不是像 SGD 那样对每次经验都改写核心参数——那是将连接主义/行为主义误解套到人身上。人真正的核心参数改写是**离线、批处理、压缩、惊喜门控**的（睡眠巩固 + 偶发洞见重构）。因此 JEPA 里"在线每步梯度"对应的是人的**意识/工作记忆推理**，而非人格/世界观的改变；持久的"自我"对应**慢速 EMA + 巩固快照 + 潜空间几何**。

| 人类"参数更新"现象 | JEPA 数学结构 | 解释 |
|---|---|---|
| 只从"意外/惊奇"中学（注意力门控） | 梯度 ∝ ∂L/∂θ，仅当 ŝ_y ≠ s_y（预测误差非零） | 更新集中在不可预测结构上 |
| 睡眠/离线巩固（批处理、压缩回放） | 周期性 latent replay + EMA 慢锚 re-anchor / 任务快照 | 在线推理 ≠ 长期改写；长期改写是离线的 |
| 长时记忆稳、短时易变 | f_ξ（EMA τ≈0.996–0.999）慢流形 vs f_θ 快权重 | "自我"=慢 EMA 不动点；"当下思绪"=快权重 |
| 心智模拟/计划/反事实 | 预测器 g_φ：s_x→ŝ_y（V-JEPA 2-AC: +动作 a） | 内部世界模型 = 前瞻认知 |
| 遗忘无关细节（功能性遗忘） | 潜空间损失只保可预测结构，编码器压缩 | 人为可控有损压缩，非灾难性遗忘 |
| 情节记忆是语义草图非原事件 | 潜回放（存 z*）vs 样例回放（存 x） | 潜回放更"人类" |
| 出生即有强先验（身体/基因） | 大规模预训练骨干 + Adapter/LoRA 残差更新 | 终身学 = 强先验上的残差 |
| 多巴胺调制可塑性（更新规则可学） | 当前 λ/ω 固定；缺 meta-learned plasticity | JEPA 的开放前沿 |
| 皮层层级、不同速率 | H-JEPA 分层，各级不同 τ | 反射快、概念慢 |

**形式化（human-JEPA 最小模型）**
```
θ_h(t+1) = (1−α(t))·θ_h(t) + α(t)·G(surprise_t, context_t)
  α(t) ≈ 0   除非 |prediction error| > 阈值        # 惊喜门控：人只从意外中学
  ξ_h     = EMA(θ_h),  τ≈0.996–0.999              # 持久自我 = 慢流形不动点
  # 离线巩固窗口（睡眠类比）:
  replay {z*_i} 批量重放 → DER 式钉定 → 更新 ξ_h → 精修预测器 g（梦境/计划演练）
```
- **清醒（在线）**：推理 + 工作记忆，θ 微漂，ξ 慢追。
- **睡眠（离线）**：回放缓冲 {z*} 重放，ξ 与快照巩固，g 精修。

**设计启示**：人并不是"逐样本真正连续地"改参数，而是**稀疏门控 + 离线巩固批**。所以 CL-JEPA 应更进一步偏重"仅高惊喜才动权重 + 离线巩固批"，而非朴素逐样本 SGD——这既更省算力，也更贴近生物。JEPA 当前仍是"更新规则固定（λ/ω 手写）"，离真正"人类式"最近的开放前沿是**可学习的可塑性门控（meta-learned plasticity / 多巴胺类比）**。

---

## 9. 最小设计（重新审视）：资源自优化 + 有限经验持续学

> 本节是对前面三层防御框架的"减法"。前面章节把 CSSL 武器库尽量搬了进来（CaSSLe 投影头、样例/潜回放缓冲、Adapter/LoRA、睡眠巩固、Clin-JEPA 五阶段协调）——那是一份"能力清单"，不是"最小必需"。本节回到用户的根本目的重新做**删除测试（drop test）**：哪些组件拿掉后，要么不再是 JEPA，要么无法在**有限内存/算力**下**从有限经验持续学习**。

### 9.1 根本目的（重述）
模型须同时满足两点，且二者都在 JEPA 内部完成：
1. **资源自优化**：内存占用恒定不增、并随学习*变便宜*；性能（每步算力）随熟悉度下降。
2. **有限经验持续学**：无需标签、从稀少观测中持续吸收，且不灾难遗忘。

### 9.2 删除测试：什么不可删
| 组件 | 删掉会怎样 | 结论 |
|---|---|---|
| 在线编码器 `f_θ` | 没有学习者 | **不可删** |
| 目标编码器 `f_ξ`（**EMA 形式**） | 退化成 I-JEPA 式 stop-grad 单编码器 → 在持续流中表征漂移、遗忘；EMA 是已知最廉价的抗遗忘（O(1) 参数空间回放） | **不可删（且必须 EMA）** |
| 预测器 `g_φ` | 退化成孪生/对比匹配，不再是"在潜空间预测"——失去世界模型核心 | **不可删** |
| 潜空间损失 `D(ŝ_y, s*_y)` | 无训练信号 | **不可删** |

**这四点即最小核心**：`{f_θ, f_ξ=EMA(f_θ), g_φ, D}`。参数峰值仅 **2× 编码器 + 1 个轻量预测器**，且与任务数无关。

### 9.3 从框架中删除（非最小）
- **CaSSLe 跨任务投影头**：EMA 目标已自带"自我蒸馏"，且无任务边界的世界模型流也不需要跨任务对齐头 → 删。
- **样例/潜回放缓冲**：缓冲是*会增长*的内存，直接违背"有限内存"。EMA 目标本身就是 O(1) 的参数空间回放（对所有历史编码器的滑动平均），**缓冲是 EMA 的冗余** → 删。
- **Adapter / LoRA**：那是"冻结巨量预训练主干 + 小增量"的工具，假定已有大底座。本目标是受限条件下的在线持续学，非适配大模型 → 删（仅当必须继承预训练底座时才回头加）。
- **睡眠/离线巩固窗口**：生物类比与优化项；在线 EMA + 惊喜门控已持续巩固，睡眠是可选的二期 → 从核心删。
- **Clin-JEPA 五阶段协调器**：为长 rollout 稳定而设，过度工程 → 删。

### 9.4 最小核心如何满足两个根本目的
**目的 2（有限经验持续学）**
- *自监督样本倍增*：每个观测 `x` 随机掩出 N 块 → N 个免费预测任务。稀少经验 → 大量梯度信号（JEPA 掩码即免费监督）。
- *惊喜门控*：梯度仅在 `E = D(ŝ_y, s*_y)` 超过阈值处生效 → 有限经验只触发"定点更新"，不熟知结构零浪费。
- *无标签*：纯自监督，"有限经验"不必标注。
- *EMA 目标*：O(1) 内存抗遗忘。

**目的 1（资源自优化）** —— 关键统一：**预测误差 `E` 同时是学习信号与资源表**。单一标量驱动三条自调节杠杆：
- **内存恒定**：`f_θ + f_ξ + g_φ` 固定大小，任务数无关。
- **内存变便宜（压缩）**：给潜表示加一个 L0 门 `m`，鼓励稀疏；稳定维度（跨 E 方差↓）被门控关闭 → 有效潜宽随时间收缩，腾出的活跃维度预算给新结构（= 以更少活跃比特表示更多 = 信息论压缩，对应 DCA 的 IG 轴）。
- **性能变便宜（算力）**：掩码率 `r` 随 `E` 自适应——熟悉(`E↓`)→掩更多→流经 `f_θ` 的 token 更少→算力↓；稳定(`E↓`)→`τ→1`→`ξ` 冻结→梯度幅值→0→更新趋零；预测器短路：对 (上下文簇) 缓存 `ŝ_y` 原型，`f_θ(x)` 落入 ε 内直接返回、跳过 `g` 前向（行为缓存）。

### 9.5 最小设计（一页伪代码）
```
状态（固定大小，O(1) 于任务数）:
  f_θ : 在线上下文编码器（唯一可学习主干）
  f_ξ : = EMA(f_θ, τ≈0.996)        # 慢速目标 = 过去知识的 O(1) 压缩快照（抗遗忘）
  g_φ : 轻量预测器（潜空间→潜空间，世界模型核心）
  m   : 可选 L0 潜门（稀疏，初始全开）

每步（仅一次前向+反向，O(minibatch)）:
  1. 取一个有限经验片段 x（可无标签）
  2. 随机掩码出 N 个目标块 → N 个免费预测任务      # 样本倍增
  3. ŝ  = g_φ( f_θ(x_vis) ) ;  s* = sg( f_ξ(x_mask) )
  4. E  = D(ŝ, s*)                                # 预测误差 = 学习信号 ∧ 资源表
  5. if E > ε_surprise:                           # 惊喜门控（有限经验→定点更新）
         θ ← θ − η·∇_θ E ;  φ ← φ − η·∇_φ E
     else: 仅推理，不更新                          # 熟悉区零算力改写
  6. ξ ← (1−τ)ξ + τ·θ                             # 持续把"过去"平均注入"现在"
  7. 资源自调节（由 E 驱动）:
        r_mask ↑ 当 E↓   （熟悉→掩更多→更少 token→更省算力）
        τ     ↑ 当 E↓     （稳定→ξ 冻结→更新趋零）
        m     稀疏化当某维 var↓（稳定结构→门控关→有效潜宽↓）
输出: 一个会自己变便宜、且不忘旧的、从稀少经验中持续学得的 JEPA。
```

### 9.6 与 DCA 4.0 的对应（为什么这非凭空设计）
DCA 的 `InfoDrives = IG × PE × CE` 中，JEPA 的 `E` 即 **PE（预测误差）= IG 的来源**；9.5 的惊喜门控即 **IG 驱动的动作/更新选择**。因此最小 CL-JEPA 本质上就是 DCA 的"感知 + 世界模型"子闭环，新增的"资源自调节"相当于把 **RE（资源经济性）** 作为一个 InfoDrive 并入——你已经在朝同一结构建造。区别只在：DCA 用离散标签联觉做符号涌现，本最小设计用 JEPA 潜空间做连续预测，二者可在 `f_θ` 的输出层对接（潜向量 → 离散符号候选）。

### 9.7 辩证收口
之前的三层框架是把"能加的都加上"；本节是"能删的都删"。最后剩下的恰好是 JEPA 的原生三件套 + 一个惊喜门控：**抗遗忘不靠外挂缓冲，而靠 EMA 目标自身的参数空间回放；样本效率不靠数据扩增，而靠掩码自监督；资源优化不靠调度器，而靠预测误差这一标量同时充当学习信号与资源表**。这正是"在有限占用下持续学、而非一次性固定"的最瘦实现——也最贴近人脑"只在意外时改写、在熟悉处节电"的运作。

---

## 10. DCA 4.0 完整介绍（信息论认知智能体）

> 本节给本文档补上"另一端"——DCA 4.0 是本文 JEPA 讨论的目的地。本版依据真实代码（`D:/workbuddyproject/20260327171412/dca4` 及同级设计文档 `DCA-4.0-DESIGN.md` / `DCA-LAYER0-DESIGN.md`）逐条核实，修正了早期依据记忆草稿的若干偏差（见 §10.15 校正说明）。关乎 `huermi` 最新代码的具体数值/命名以代码为准。

### 10.1 定位与根本目的
- **DCA 4.0 = 信息论认知智能体（Information-theoretic Cognitive Agent）**，由 huermi 独立开发，项目性质为个人研究（代码 ~32 个 .py / ~4637 行，24 个功能模块，每个 <400 行；README 标为 4.0.0-alpha）。
- **核心愿望**：构建一个**独立存在、能自主学习与发现**的认知体，而非被调用的工具。作者自我定位是"研究者而非产品经理"——目标是*观察 DCA 学习*，而非管理它。
- **路线立场**：对纯 LLM 范式持明确批判，称其"只有智能的影子"；坚持从杨立昆 JEPA 视角审视架构；反对把文字编码为连续向量，坚持**离散标签联觉**作为符号涌现路径。

### 10.2 设计哲学
- **辩证双系统**：作者自构"双系统辩证引擎"方案，与 LeCun 的 JEPA 形成对照/竞争路线；二者都拒绝像素/词元级生成，转向表征/潜空间预测。
- **信息论视角**：以信息增益（IG）、预测误差（PE）为核心驱动力，决策由 `InfoDrives` 的信息论量统一辖制。
- **失败即数据**：把崩溃、异常、低显著性都当作可观测信号，而非需要掩盖的 bug。
- **先诊断后改**：工作流遵循"诊断工具 → 验证运行 → 数据分析 → 改代码"，反对表面修补。

### 10.3 核心循环与模块架构
核心循环（HANDOFF 交接文档确认）：`capture → encode → drive → predict → memory → simulate → action`。

| 层 | 模块（真实路径） | 角色 | 关键事实（核实） |
|---|---|---|---|
| 感知 | `perception/capture.py` | 双分辨率帧捕获（桌面 + Playground） | 主帧源(桌面)→256d，副帧源(Playground)→256d，双通道→**512d** |
| 感知 | `perception/conv_encoder.py` | 像素块→256d 特征 | 局部像素块（约 8–16×16）→256d；刻意保留可区分性（cos≈0.4，远低于 CNN 的 0.9）以防特征坍缩 |
| 编码 | `encoding/sparse_proj.py` | 稀疏随机投影 | 10816/12544d → **256d**（`SparseProjection`） |
| 驱力 | `cognition/drive.py` | DriveUnit（dca3 遗产） | 行动条件预测器 `_W1/_W2` + 偏好衰减 + 自模型 |
| 驱力 | `cognition/needs.py` | 内稳态需求（Layer 0） | `HomeostaticNeeds`：change/effective/novel 三维，0(饱和)~1(饥饿) 自然漂移 |
| 驱力 | `cognition/info_drives.py` | 信息论驱力（v4.1 零预设） | `IG×PE×CE` 自然涌现，无人工分类 |
| 世界模型 | `prediction/predictor.py` | AdaJEPA 预测器 | 两层 MLP，状态×动作→Δz；W3=符号涌现层 |
| 规划 | `planning/consequence.py` | ConsequenceSim + Veto | 多分支反事实；`veto_check` 已接入主循环 |
| 记忆 | `memory/pools.py` `distributed.py` | 5 层记忆池 + 分布式矩阵 | 每动作 K 个 256d 槽位，EMA 更新；SparseTopK 检索 |
| 注意 | `attention/gaze.py` | 凝视控制器 | 外源/内源仲裁 |
| 交互 | `action/playground.py` `playground_window.py` | Playground 虚拟环境 | VirtualDisplay(768×768) + Tkinter 窗口，作为隔离训练场 |
| 巩固 | `consolidation/replay.py` | 回放巩固（设计） | 每 5000 tick 高频重要经验回放，低频自然衰减 |

### 10.4 驱力系统（三条路线并存，融合待验证）
真实代码里**并非单一决策原则**，而是三条驱力来源共存（§10.15 校正了"DriveUnit 已移除"的旧说法）：
1. **DriveUnit**（`cognition/drive.py`，dca3 遗产）：行动条件预测器 + 偏好自动衰减 + 自模型。
2. **内稳态需求**（`cognition/needs.py`，基于 `DCA-LAYER0-DESIGN.md`）：change/effective/novel 三维，随时间向"饥饿"漂移、特定行动满足之——提供**无外部目标也能行动**的内生价值。
3. **信息论驱力**（`cognition/info_drives.py`，v4.1 零预设）：三个公理自然涌现——
   - 公理1 信息增益 IG：选最能降低世界模型不确定性的行动（好奇心）
   - 公理2 预测误差 PE：偏好后果可预测的行动（胜任感）
   - 公理3 条件熵 CE：在探索/利用间自我调节（反厌倦）
   - 统一驱力值 `V(a) = IG(a) × (1−PE_norm(a)) × CE(a) × temp`（Boltzmann 采样选动作）。
   - **符号涌现内嵌**：`update_outcome` 维护 `outcome_archetypes` 原型聚类（上限 20，余弦 >0.7 合并）——把行动后果收束为离散符号候选。
   - HANDOFF 明确：`drive.py` 与 `needs.py` 的内稳态融合**尚未测试**。

### 10.5 世界模型 Predictor（AdaJEPA —— JEPA 训练原则的落地）
`prediction/predictor.py` 是一个**行动条件的潜空间前向模型**，也是 DCA 与本文 JEPA 的最直接对接点。其 JEPA 式训练原则（`tests/test_adajepa.py` 四项验证）确属事实：
- **A. 预测误差反馈**：用潜空间预测误差替代像素差作为 fb；
- **B. 连续在线适应**：每 tick 1 梯度步，**分层学习率**（编码器 W1 `lr=0.001` 远低于预测器 W2 `lr=0.05`）；
- **C. 防坍缩**：目标 L2 归一化（stop-gradient 等价物），预测 **Δz = z_{t+1}−z_t**（残差模式，避免绝对状态坍缩）；
- **D. 局部 recent-N 缓冲**：`adaptive_buffer.py` 存最近转移做在线回放。
结构：输入 `(state_256d, action_onehot)[+self_256d]` → 隐层 128/64 → 输出 Δz；**W3（outcome_dim 维）= 符号涌现层**，把行为后果投影为符号候选嵌入。

### 10.6 ConsequenceSim 与 Veto（多分支反事实 + 分层否决）
- **ConsequenceSim**（`planning/consequence.py`）：对 top-3 候选动作各前向模拟 1 步，并以 `WAIT` 为基线做反事实分支对比（因果增益 = `‖pred(a)−pred(WAIT)‖`），输出 `corrected_bias = direct_bias + 期望IG×置信×好奇权重`。
- **Veto**（`veto_check`，已接入 `main.py:472`）为**两层安全网**（非 8d 原型）：
  - **L1 压缩保护**：预期 PE > `VETO_L1_PE_THRESHOLD`(=1.5) → ×`VETO_L1_PENALTY`(=0.3)；
  - **L2 高阶特征保护**：预测状态与核心记忆相似度 < 0.3 → ×`VETO_L2_PENALTY`(=0.5)。
- 注意：HANDOFF 指出完整 `simulate→corrected_bias` **尚未验证**；当前主循环仅确证接入了 `veto_check`。

### 10.7 记忆系统（5 层池 + 分布式矩阵）
- `memory/pools.py`：5 层记忆池（含情节 episodic）。
- `memory/distributed.py`：分布式矩阵，**每动作 K 个 256d 槽位，EMA 更新最近匹配槽位**。
- `memory/sparse_activation.py`：SparseTopK 检索（替代全矩阵点积，约省 90% 算力）。
- `memory/diff_chain.py` / `categories.py`：差分链 / 概念类别系统。

### 10.8 符号涌现（两条互补机制）
1. **Predictor W3 后果嵌入**（`outcome_dim` 维，可配置）——每次预测的"行为后果"投影；
2. **InfoDrives outcome_archetypes 聚类**（上限 20，余弦 >0.7 合并）——把行动后果收束为离散符号候选（联觉层）。
二者共同构成"连续潜向量 → 离散符号"的涌现路径，对应作者坚持的**离散标签联觉**。

### 10.9 自模型 Self-model（`drive.py: update_self_model`）
反射式自监控：用 SGD 预测"自身内部状态"（`actual_self`），权重 `_Ws1/_Ws2`，**学习率减半以更稳定**，维护内存/速度/健康类自我表征。这是 DCA 的"自我"构件，与 §9.5 的"持久自我=慢不动点"精神一致（但实现为显式前向模型，而非 EMA 潜流形）。

### 10.10 身体与交互架构（双视觉通道 + 四类用户）
- **双视觉通道**：状态 = 桌面 256d + Playground 256d = **512d**（`STATE_DIM=512`）。Playground 不是 256×256 缓冲，而是 `action/playground.py` 的 `VirtualDisplay(768×768)` + `PlaygroundWindow`（Tkinter）——DCA 在桌面截图模式下**自然看到这个窗口**，相当于在隔离环境里"打开应用"而**不干扰真实桌面**。
- **四类用户**：A 观察者 / B 工具用户 / C 非技术老年人 / D 维护者。
- **根本架构矛盾（未解）**：DCA 的"身体"等于用户的电脑——物理上不可能与用户共享输入设备，且束缚在用户桌面上无法独立存在；Playground 是当前的工程化缓解。
- 说明：早期草稿提到的 Win32 `PrintWindow` Sandbox 隔离窗口，在当前 `dca4` 代码中未检索到实现，应为设计阶段方案，待作者确认。

### 10.11 长期运行验证（来自运行日志/报告）
- **60k tick 纯观察运行**：`dist_active` 5% → 48.6%；`gaze_familiarity` 0 → 0.846；`CP` 稳定于 −100 钳位、无崩溃；Predictor 稳定。
- **社会学习实验**：10 段全部显著（p < 0.000001）；准确率 **67.6%** vs 随机基线 **19.3%**。
- **已修复 7 个关键 Bug**：`dist_active=0`(P0)、`OR` 逻辑(P1)、`vitality scaling`(P2)、`save/load` 不匹配(P0)、`mss buffer` 崩溃(P0)、`CP` 振荡(P1)、`phase_lock`(P1)。

### 10.12 诊断工具与开发三原则
- **诊断指令**：`?recall`（回忆）、`?memstats`（记忆统计）、`?diag`（综合诊断，`core/diagnostics.py`）监控运行状态。
- **开发三原则**：稳定性优先于功能性；失败即数据；时间预算硬约束；先建诊断再改代码（HANDOFF）。

### 10.13 与 JEPA / CL-JEPA 的关系（代码级对接点）
- **世界模型对接**：DCA 的 `Predictor`（行动条件潜空间前向模型）= §9 最小核心的 `g_φ`。其"目标归一化/分层 LR/在线 Δz 适应/局部回放"是 DCA 自有的 JEPA 式训练落地。
- **预测误差同构**：DCA 的 `PE` = JEPA 的 `E` = IG 来源；§9 的惊喜门控 ↔ InfoDrives 以 IG 驱动探索。
- **可补的 EMA 目标编码器**：§9 最小核心用 **EMA 目标编码器 `f_ξ`** 抗遗忘；当前 DCA Predictor 用"目标归一化"而非 EMA 目标编码器——这是把 §9 框架回填进 DCA 的**具体切入点**（加一个 EMA 慢副本即获得 O(1) 参数空间回放）。
- **资源自调节**：§9.4 的"掩码率↑/τ→1/潜门稀疏随 E↓"对应在 DCA 中把 **RE（资源经济性）** 作为一个 InfoDrive 并入 `IG×PE×CE`。
- **符号层对接**：§10.8 的符号涌现（W3 + InfoDrives 聚类）挂在 `f_θ`/`Predictor` 输出层——JEPA 给连续潜向量，联觉层收束为离散符号候选，补上 §9 最小核心"只有连续预测、缺符号涌现"的一侧。
- **世界模型扩展**：`ConsequenceSim` 多分支反事实 = §9 `g_φ` 从单步潜预测扩展到反事实多分支；`Veto` 两层的 `PE`/`核心记忆` 门槛 = §9"仅高惊喜才更新"的显式安全化。

### 10.14 小结
DCA 4.0 不是本文框架的"应用实例"，而是它的**母系统**：§9 的最小 CL-JEPA 是从 DCA 中抽取、可被独立验证的"感知—世界模型—资源自调节"内核。代码已证实 DCA 自带 JEPA 式预测器（AdaJEPA）与信息论驱力，二者与 §9 框架同构；把 §9 的 **EMA 目标编码器 + 惊喜门控 + 潜空间回放**回填进 `Predictor` 与 `InfoDrives`，即获得一套已论证的、**内存/算力有界**的持续学习实现；而 DCA 的离散符号涌现与显式自模型，补上纯 JEPA 缺失的"符号"与"自我"。

### 10.15 校正说明（本版相对记忆草稿的修正）
| 旧草稿（记忆） | 代码核实后 |
|---|---|
| Outcome/Veto 为 8d 嵌入 | 实际潜变量为 **256d**（通道 256，状态 512）；Veto 为 L1-PE + L2-核心记忆相似度两层，非 8d 原型 |
| DriveUnit 已移除，InfoDrives 唯一 | DriveUnit(`drive.py`)、内稳态(`needs.py`)、InfoDrives(`info_drives.py`)**三路线并存**，融合未测 |
| Playground = 256×256 像素缓冲 | Playground = 双视觉通道之一（768×768 VirtualDisplay + Tkinter 窗口），构成 512d 状态 |
| Sandbox = Win32 PrintWindow | 当前代码未检索到 PrintWindow 实现，应为设计阶段方案 |
| Self-model 5d | `drive.py: update_self_model` 以 256d 特征预测自身内部状态，lr 减半 |

---

## 8. 参考文献（精选）

- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*. OpenReview.
- Assran et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)*. CVPR 2023. arXiv:2301.08243
- Bardes, Ponce, LeCun (2023). *MC-JEPA*. arXiv:2307.12698
- Bardes et al. (2024). *Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA)*. arXiv:2404.08471
- Assran et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*. arXiv:2506.09985
- Meta AI (2025). *VL-JEPA*. arXiv:2512.10942
- Huang, LeCun, Balestriero (2025). *LLM-JEPA*. arXiv:2509.14252
- Kalapos, Gyires-Tóth (2025). *Exploring joint embedding predictive architectures for pretraining CNNs (CNN-JEPA)*. CVIU, doi:10.1016/j.cviu.2025.104595
- Ennadir et al. (2025). *TS-JEPA*. arXiv:2509.25449
- (2026). *GMM-JEPA*. arXiv:2602.09040
- Yang et al. (2026). *Clin-JEPA*. arXiv:2605.10840
- Fini et al. (2022). *Self-Supervised Models are Continual Learners (CaSSLe)*. CVPR 2022.
- Gomez-Villa et al. (2022). *PFR*.
- (2024). *Integrating Present and Past in Unsupervised Continual Learning (OSIRIS)*. arXiv:2404.19132
- Cignoni, Cossu et al. (2025). *CLA: Latent Alignment for Online Continual Self-Supervised Learning*. CoLLAs 2025. arXiv:2507.10434
- (2025). *Lifelong Representations: A Survey on Continual Self-Supervised Learning for Vision Models*. arXiv:2607.09785
- github: jhairgallano/awesome-continual-self-supervised-learning

---

*附：配套架构图见同目录 `cl-jepa-architecture.svg`（亦在对话中内联呈现）。*
