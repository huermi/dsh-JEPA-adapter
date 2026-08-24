# dsh-JEPA-adapter

> 适配调用 JEPA 模型的插件 —— 一个**本地的 JEPA 认知体**（DCA-4.0 的检索式实现），**可在家用计算机 CPU 上运行**。

本项目提供一个 OpenAI 兼容的本地模型服务（模型 ID `jepa-1`）：任意支持 function-calling 的 harness（DeepSeek / Claude Code / 自定义客户端）在模型列表中选择 `jepa-1` 即可对话与调用工具。内核不是 LLM 参数生成，而是 **「情境 → 记忆检索 → 决策」** 的符号-向量混合架构：它会**持续学习**（受阻触发学习、受阻-通过沉淀、选择性遗忘、矛盾处理），并且**知道自己学得健不健康**（认知再生产指标 + 病理监测）。

**English README: [README.en.md](README.en.md)**

---

## 特性速览

| 能力 | 说明 |
|---|---|
| 🧠 检索式认知内核 | 回应由记忆检索产生（非生成），回答可追溯、可证伪 |
| 📚 持续学习 | 受阻（检索 miss/低置信）触发学习，判定反馈驱动沉淀 |
| 🧬 分层记忆 | fast（高频/运动）→ core（沉淀/扬弃）→ archive（归档/可恢复） |
| ⚖️ 矛盾处理 | 相似情境不同答案 → 矛盾对 → 实践裁决（Wilson 置信区间） |
| 🗑 选择性遗忘 | 质量×时间加权淘汰、巩固分降级、权重衰减+验证强度保护 |
| ⚡ 权重记忆 (LAM) | 线性联想记忆：O(1) 预测、96KB、关系型知识泛化 |
| 🔧 工具调用 | 调用模式记忆检索（工具+参数），多步任务情境驱动 |
| 🩺 病理监测 | 范式更新率/沉淀率/区分度/校准形状/工具熵 + 四型病理警示 |
| 🌐 OpenAI 兼容 | 任意 harness 可选 `jepa-1`；浏览器设置界面 `/ui` |

---

## 核心机制概览

| 机制 | 大致描述 |
|---|---|
| **检索式回应与分层记忆** | 回应由记忆检索产生（非生成），回答可追溯可证伪；fast / core / archive 三层记忆，通过分达标晋升 core、巩固分跌破归档（可恢复） |
| **受阻-通过沉淀判据** | 判定正确 +1 / 答错 −2（含频率加权），通过分达标才沉淀——「被实践反复考验且通过的经验才沉淀」，答错的条目不沉淀还会被优先淘汰 |
| **受阻触发学习** | 检索 miss / 低置信 / 矛盾 = 受阻 → 探索查证；学习触发 = 预期失配，不是好奇（好奇会漏学「似曾相识但重要」的知识） |
| **矛盾处理协议** | 相似情境不同答案 → 矛盾对 → 实践反馈裁决（Wilson 置信区间）：互斥分胜负（输方弃权）/ 互补并存（条件差异）/ 证据不足待求证（宽容等待） |
| **可证伪范式** | 工具调用 = (情境, 行动模式, 结果状态)；被证伪的调用不复用（反例驱动积累）；任务→工具偏好从已验证调用统计学出（替代手工词表） |
| **权重记忆 LAM** | 线性联想记忆 W：情境→答案向量，delta 规则在线更新（O(1) 预测、96KB）——关系型知识的泛化通道（事实型知识靠外部检索） |
| **软校准（AdaJEPA）** | 判定正确后把命中条目表征向查询微调（小步长 α=0.1）——相似输入自动受益（泛化增益），错误判定不校准 |
| **分层遗忘** | 遗忘优先级 ∝ 判别价值 × 验证历史：fast 质量×时间加权淘汰 / core 巩固分跌破归档 / W 慢衰减 + 验证强度保护 |
| **认知再生产指标** | status 报告范式更新率 flux、沉淀率、检索区分度 margin、校准形状、工具熵 + 四型病理警示（检索退化/表征失真/认知僵化/价值萎缩） |
| **LeCun 式渐进替换** | 手工阈值逐步替换为数据驱动决策面：二维校准表（决策边界）/ 统计先验（词表）/ Wilson 置信区间（矛盾裁决） |

## 架构

```
body/
  kernel.py           认知体核心：决策链 (调用记忆 → 检索回应 → 探索 → 收尾)
  respond_learner.py  检索式回应学习器：分层记忆 + 受阻-通过 + 矛盾 + LAM + 软校准 + 遗忘
  call_memory.py      调用模式记忆：(情境→工具+参数+结果) + 可证伪门控 + 任务→工具统计先验
  mini_encoder.py     MiniLM 语义编码器（惰性加载、LRU 缓存、默认离线、失败哈希袋兜底）
  plugin_config.py    PluginConfig：22 项可配置 (dataclass, 环境变量 JEPA_* 可覆盖)
  plugin_server.py    OpenAI 兼容 HTTP 服务器 + 设置界面 (/ui)
components/           DCA 本体组件（InfoDrives / Configurator / energy 等）
```

### 决策链（`chat_completion`）

```
完整消息历史 → 情境 z_ctx
  1. call_mem.select_k top-K → 世界模型预测验证排序 (_select_planned)
     → 命中 → tool_calls (via=call_memory+planned)
  2. responder.respond (检索式回应)
     → 命中 → content (via=retrieval)
     → miss → 记录受阻类型 (retrieval_miss / low_confidence / contradiction)
  3. _select_explore 探索 (受阻后, 预算内, 温度衰减)
     → tool_calls (via=explore) → tool_result_step 学习 → 下次检索命中
  4. 默认收尾 (via=default)
```

### 三层记忆 × 三层遗忘（完整蓝图）

```
        进入 (质量×频率)          离开 (证据综合)
fast ───────────────→ core ──→ archive（归档，可恢复）
  │                      ↑
  └── W（衰减 + G 保护）──┘
```

---

## API 文档

服务：`python body/plugin_server.py`（端口 8045，`JEPA_PORT` 可改），OpenAI 兼容。

| 端点 | 方法 | 说明 |
|---|---|---|
| `/v1/chat/completions` | POST | 对话 + 工具调用（OpenAI function-calling 格式，支持 `stream`） |
| `/v1/models` | GET | 模型列表（`jepa-1` / JEPA DCA-4.0）——harness 模型选择用 |
| `/v1/status` | GET | 模型状态 + 开关矩阵 + responder/call_mem 统计 + **cognition 健康板块** |
| `/v1/config` | GET | 当前配置 + schema（设置界面数据源） |
| `/v1/config` | POST | 热更新配置（类型校验，responder 响应式参数即时生效） |
| `/v1/sleep` | POST | 触发睡眠巩固（记忆回放） |
| `/v1/learn_response` | POST | 显式教回应（情境→文本） |
| `/v1/archive` | POST | 存档（save/load） |
| `/ui` | GET | 浏览器设置界面（深色主题，全部配置项可视化） |

### 对话示例

```bash
curl http://127.0.0.1:8045/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "jepa-1",
    "messages": [{"role": "user", "content": "list python files in the project"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "glob",
        "description": "list files matching a glob pattern",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}}
      }
    }]
  }'
```

返回 `tool_calls` 后，harness 执行并把结果以 `role: tool` 消息回传——JEPA 通过 `tool_result_step` 学习完整调用模式，多步任务每步由情境驱动（list → read → done）。

### 配置项（22 项，`/ui` 或 `/v1/config`）

| 配置 | 默认 | 说明 |
|---|---|---|
| `learning` | true | 在线权重梯度 |
| `memory` | true | 记忆写入 |
| `sleep` / `sleep_epochs` / `sleep_lr_scale` / `sleep_prio_mix` | false / 3 / 0.3 / 0.5 | 睡眠巩固（重放固化） |
| `tools` | true | 工具调用 |
| `respond_mode` | retrieval | retrieval=检索式学会说话 / llm=桥接外部 LLM 表达 |
| `archive` | false | 存档（任务级快照） |
| `surprise_thresh` | 0.3 | 惊讶门控（E1 触发工具调用阈值） |
| `max_memory` | 200 | 记忆上限 |
| `explore` / `explore_budget` / `explore_decay` | true / 3 / 0.9 | 自由学习探索（预算+温度衰减） |
| `benchmark_mode` | false | 基准模式（禁探索/网络，纯内化知识） |
| `respond_cap` | 300 | 回应经验上限 |
| `respond_min_sim` | 0.45 | 回应检索阈值 |
| `soft_align` / `soft_align_alpha` | true / 0.1 | AdaJEPA 软校准（步长 α） |

环境变量：`JEPA_EXPLORE` / `JEPA_SOFT_ALIGN` / `JEPA_BENCHMARK` / `JEPA_RESPOND_CAP` / `JEPA_PORT` 等（`PluginConfig.from_env()`）。

---

## 快速开始

```bash
# 1. 依赖 (Python ≥3.10)
pip install -r requirements.txt
# 或分开装 (CPU 版 torch 更小):
#   pip install numpy pyarrow torch --index-url https://download.pytorch.org/whl/cpu
#   pip install transformers sentence-transformers

# 2. 模型：首次运行自动下载 sentence-transformers/all-MiniLM-L6-v2 (~90MB)
#    默认 hf-mirror.com 镜像，缓存到 models/ (JEPA_MODEL_DIR 可覆盖)

# 3. 启动服务器
python body/plugin_server.py

# 4. 对话 / 工具调用（见上例）；设置界面 http://127.0.0.1:8045/ui
```

> 默认强制离线加载（`HF_HUB_OFFLINE`）：模型已本地缓存时不再联网检查，避免断网环境 import 卡死（transformers 联网超时重试问题）。

## 接入 dsh harness（DeepSeek 本地 harness）

dsh 通过 OpenAI 兼容 provider 接入 JEPA，三步：

```bash
# 1. 启动 JEPA 服务器（见快速开始）
python body/plugin_server.py

# 2. 设置 API key 环境变量（JEPA 服务器不校验 key，占位即可）
setx JEPA_API_KEY jepa-local-key
```

3. 在 `~/.dsh/settings.yaml` 注册 provider（如未注册），并设为默认模型：

```yaml
llm-pi-ai:
  providers:
    jepa:
      {
        displayName: JEPA Body (本地模型),
        apiKeyEnv: JEPA_API_KEY,
        api: openai-completions,
        baseURL: http://127.0.0.1:8045/v1/,
        models: [ { id: jepa-1 } ]
      }
agent-default-model:
  provider: jepa
  model: jepa-1
```

完成：dsh 模型选择中出现 **"JEPA Body (本地模型)"**，选择 `jepa-1` 即可对话与工具调用（多步任务由 JEPA 检索调用记忆驱动：list → read → done）。

## 学习与训练

```bash
# 自主查证学习 (MMLU 8 科基准；资料库见 benchmark/library/)
python learner_loop.py --per 12
# → 学后正确率 27.1% (global_facts 83%; 26/96 从不会到会)

# 自动化训练 (教材采集→评估；快照存 benchmark/snapshots/)
python auto_train.py --rounds 3
# → 教材采集: 训练题(问题|正确选项)入库, 正确性由标注保证 (与模型表现解耦)
# → 评估: 每题独立干净 body, 测"教材能否支撑未见题" (global_facts 83.3%)

# 互联网检索学习 (抓取→内化→复用→混叠纠错→泛化 全链路)
python train_web_learning.py
```

**已知基准表现**（诚实声明，2026-08-25）：

| 场景 | 结果 | 说明 |
|---|---|---|
| 自主查证学习（learner_loop 96 题） | 27.1% | 受"同构混叠+资料库覆盖"双瓶颈 |
| 教材泛化（auto_train, global_facts） | 83.3% | 教材支撑未见题（泛化超背题） |
| 内化评估（教材学进模型） | 1.0% | 事实型知识检索式架构的本质边界 |
| econometrics | 0% | 资料库零覆盖（数据边界，非机制问题） |

## 验证脚本（机制级回归）

```bash
python memory_layers_check.py         # 分层记忆：学新不毁旧
python falsification_check.py         # 可证伪性：被证伪的调用不复用
python soft_alignment_check.py        # AdaJEPA 软校准泛化 (0.890→0.927)
python selective_forgetting_check.py  # 分层遗忘五机制
python contradiction_check.py         # 矛盾处理协议四场景
python status_metrics_check.py        # 认知再生产指标 + 病理监测四场景
python multi_step_check.py            # 连续多步工具调用
python dsh_harness_check.py           # 端到端对接 (OpenAI 兼容)
python study_trigger_experiment.py    # 触发源/沉淀判据对照实验 (理论验证)
```

---

## 第三方模型与训练引用说明

| 组件 | 引用 | 许可 |
|---|---|---|
| 语义编码器 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)（HuggingFace，默认经 hf-mirror.com 镜像下载） | Apache-2.0 |
| 运行时 | torch (CPU) / transformers / sentence-transformers / numpy / pyarrow | 各自许可 |

**本项目不训练、不微调任何第三方权重**——MiniLM 仅作**冻结的文本编码器**（情境向量化，384d）。模型内学习（responder 记忆、LAM 权重、矛盾裁决、软校准）全部在推理时在线进行，不产生第三方权重的衍生品。基准测试使用 MMLU 数据集时请遵循其原始许可（**仓库内不含 MMLU 数据文件**）。

---

## 已知边界（诚实声明）

1. **检索式架构的本质边界**：同构统计题的检索区分度（问题形式相似、答案不同）——余弦+词汇闸+W 都无法完全分离，只能精确记忆或外部查证
2. **事实记忆 vs 关系知识**：权重记忆（LAM）只对关系型知识有效；事实型知识（MMLU 类）靠外部检索/资料库
3. **LLM 定位**：本架构中 LLM 是"符号翻译器"（语言↔意图↔符号），不承担事实记忆——事实必须来自认知层（可追溯、防幻觉）
4. **内化评估 1%**：教材直接学进检索模型后泛化≈0——"训练数据库≠训练模型"，模型结构内能力的提升需要生成式/参数化路线（超出本仓库范围）

## 开源说明

- **内部研究文档**（`benchmark/*.md`：理论推进、训练计划、LeCun 对比分析）与**诊断脚本**（`*_check.py`、`jpi*_*.py` 等）公开，供研究者参考完整设计脉络与实验数据
- **项目记忆日志**以脱敏副本公开（`memory_public/`）：个人路径、用户名、工作区 ID 已模糊化（`<repo-root>` / `~/` / `<project-id>`）；原始日志在 `.workbuddy/` 不公开
- 代码中所有本地绝对路径已相对化（`REPO_ROOT` 自动探测），克隆后可直接运行

## License

[MIT](LICENSE) © 2026 huermi
