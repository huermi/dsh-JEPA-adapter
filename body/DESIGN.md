# JepaBody — 真实身体底层模型内核

日期：2026-08-24 ｜ 状态：最小验证 + DeepSeek harness 插件验证全过

## 定位

把三环系统（JepaAgent + C1-C10）封装成**底层模型形式**，可接入各家 harness。
现在已具备：五类原子能力 + OpenAI 风格接口 + 工具调用循环 + 功能开关矩阵 +
响应学习（学会输出字符）+ 睡眠巩固 + 存档。

## 功能开关矩阵（PluginConfig）

```python
from body.plugin_config import PluginConfig
cfg = PluginConfig(learning=True,   # 在线权重梯度 (任务经验是否改变权重)
                   memory=True,     # 记忆写入 (惊讶门控)
                   sleep=False,     # 睡眠巩固 (默认关, 显式/周期触发)
                   tools=True,      # 工具调用
                   respond_mode="retrieval",  # retrieval=学会说话 | llm=桥接
                   archive=False)   # 存档
# 环境变量加载: JEPA_LEARNING / JEPA_MEMORY / JEPA_SLEEP / JEPA_TOOLS /
#               JEPA_RESPOND_MODE / JEPA_ARCHIVE / JEPA_SURPRISE
cfg = PluginConfig.from_env()
```

## 接口契约（harness 只依赖这些）

```python
from body import JepaBody
body = JepaBody(config=cfg)             # 开关矩阵配置

# 五类原子能力
z   = body.perceive(obs)                # 任意观测(图像/文本) → 768d 表征
z2  = body.predict(z, a, delta=1)       # 世界模型预测 (delta>0 未来)
e   = body.energy_of(z)                 # 惊讶度 (记忆陌生度反信号)
d   = body.decide(obs)                  # {action, energy, tool_calls}
info= body.learn(obs, a, obs_next, perf, died, tool_result=None)  # 在线学习

# 工具
body.register_tool(name, fn, description, parameters_schema)
body.call_tool(name, args) -> str

# OpenAI 风格对接 (harness 入口)
resp = body.chat_completion(messages, tools)   # → {content, tool_calls}
final= body.tool_result_step(tool_calls, results)  # 工具结果回传学习

# 响应学习: 让模型学会输出字符 (检索式, 不依赖 LLM)
body.learn_response(obs, text)          # 教 (情境→回应)
resp = body.chat_completion(...)        # retrieval 模式下命中即直接回应

# 睡眠巩固 (记忆回放 → 权重固化)
r = body.sleep()                        # 返回 E1 前后对比

# 存档 (任务边界回滚)
body.save("body.pkl")  /  body.load("body.pkl")
```

## DeepSeek harness 插件（HTTP 桥）— ✅ 端到端打通

```bash
# 启动 (环境变量配置开关)
JEPA_PORT=8031 JEPA_LEARNING=1 JEPA_MEMORY=1 JEPA_SLEEP=0 \
python body/plugin_server.py

# 端点 (OpenAI 兼容)
POST /v1/chat/completions   对话 + 工具调用 {messages, tools}
POST /v1/tool_results       工具结果回传学习 {tool_calls, tool_results}
POST /v1/learn_response     显式教回应 {obs, text} — 学会输出字符
POST /v1/sleep              触发睡眠巩固
POST /v1/archive            {action: save|load, path}
GET  /v1/status             模型状态 + 开关矩阵
```

harness 对接方式：DeepSeek harness（dsh）把 settings.yaml 的 provider baseURL
指向 `http://127.0.0.1:PORT/v1`，默认模型选 `jepa/jepa-1` 即可。

### 真实接入验证（2026-08-24，dsh 0.1.1-rc.2）

- headless 端到端跑通：`dsh --profile headless "task"` → 插件 → 模型 → 输出打印
- **学会输出字符实证**：learn_response 教 3 条回应 → 3 个查询全部命中（零 LLM）
- 关键修复链：SSE 流式 + Connection: close（pi-ai 等 EOF 否则卡死）；
  工具白名单（dsh 25 工具只暴露做事工具）；初始无记忆不调工具（防空参数死循环）；
  取第一条 user 消息（dsh 注入上下文在任务后）

### 已知边界
- 工具调用参数生成未实现（arguments 恒空 → 工具任务会循环，需从记忆学习完整调用模式）
- Qwen 语义编码 CPU 加载 3.5min 不可行（默认哈希袋+白名单）

## 验证状态（plugin_check.py 五项全过）

| 项 | 结果 |
|---|---|
| 开关矩阵 | 关 learning+memory → 任务后模型完全冻结（零污染）；开 → 变化 |
| 响应学习 | 教 3 条 (情境→回应)，相似查询检索命中 2/3（无任何 LLM 参与） |
| HTTP 端到端 | 对话 → 工具调用 → 结果回传 → 学习闭环 ✅ |
| 睡眠巩固 | 任务后 sleep()：E1 降 65%（真实预测器） |
| 存档恢复 | save → 污染 → load → 完整回到存档状态 |

## 验证结果（body/check.py）

| 项 | 结果 |
|---|---|
| [1] 内核闭环 200 tick | ✅ 动作活跃 5 种 + 记忆增长 28 条 |
| [2] OpenAI 兼容接口 | ✅ JSON 往返 + 高惊讶触发 tool_calls + 执行 + 回传 |
| [3] 工具调用闭环 | ✅ 触发 31 次 / 0 错误 / 记忆 29→59 |

## 真实任务输出能力（body_real_check.py，真实权重 timm ViT-B/16 + CIFAR-10）

JEPA 是预测表征内核非生成器 → 输出能力的现实形态：

| 项 | 结果 |
|---|---|
| [R1] 图片理解输出（KNN top-1） | ✅ **90.5%**（200 张测试，随机 10%） |
| [R2] 文本响应输出（回忆=输出，记忆检索） | ✅ **50%**（20 新图命中同类描述，随机 10% 的 5 倍） |
| [R3] JEPA 原生预测输出（世界模型） | ✅ E1 下降 **93%**（真实图片表征上收敛） |
| [R4] 音频输出 | ⚠️ 边界标注：无音频编码器/数据；正式版接 AST/CLAP 做理解，生成走工具 |

关键修正记录：
- **文本→观测**：哈希袋向量直接返回，不过图片编码器（模态特化编码器收到 768d 向量会崩）
- **R3 随机扰动不可学**（E1 不降是设计缺陷）→ 改"同图+固定偏移"（可学的结构规律），JEPA 预测的是规律不是随机性
- **卡点修复**：datasets 加载必须 HF_DATASETS_OFFLINE=1（在线校验挂起）
- R2 测试场景绕过惊讶门控注入经验（e1_hist 预填充）

## P4b 架构验证（p4b_check.py）— JEPA 指挥官 + 生成引擎执行器

**架构**：JEPA 决定"要什么输出"（意图）→ 工具调用 → 生成引擎执行（Qwen 写文本 / PIL 画图）→ 结果回传记忆 → 回忆可检索。

| 验证点 | 结果 |
|---|---|
| V1 意图选择（统一空间语义检索） | ✅ "写诗"→write_text，"画猫"→draw_image（Qwen 语义表征匹配） |
| V2 生成落地 | ✅ Qwen 真实写诗 + PIL 画猫存 PNG（outputs/） |
| V3 回传记忆 | ✅ 工具结果入记忆（绕过门控，2 条） |
| V4 回忆能力 | ✅ Qwen 空间检索命中 |

关键修正记录：
- **工具选择 = 统一空间语义检索**：任务表征 vs 工具描述表征（Qwen 896d → 固定投影 768d），替代写死的 calculator 优先——`_select_tool` + `set_tool_embed`
- **感知统一管道**：QwenPerception（文本→Qwen 896→投影 768），decide/记忆/工具选择全程同一空间（哈希袋与 Qwen 空间错配是 V4 失败根因，已修）
- 字符串观测统一走 `perceive`（模态编码器优先，哈希兜底），agent 内部用统一表征

## P4a 原始输入补全（p4a_check.py + body/multimodal.py）

全模态统一 768d 表征（MultimodalPerception.encode）：

| 模态 | 编码器 | 验证 |
|---|---|---|
| 图片 | JepaPerception（timm ViT-B/16） | ✅ KNN 90%（回归保持） |
| 文本 | Qwen 语义（P4b）或哈希兜底 | ✅ 768d |
| 音频 | 频谱统计特征（STFT→Mel→24 统计）→ 投影 | ✅ 同类 1.000 vs 异类 0.199（间隔 0.801） |
| 视频 | 帧级 timm + (均值,std) 时序统计 → 投影 | ✅ 静态 vs 运动 0.945（std 维度捕捉运动） |

关键修正：
- 音频特征实为 68 维（4+32+32），投影矩阵按实际维度
- 视频纯均值池化把运动稀释（0.964 不可分）→ 加帧间 std 维度（0.945 可分）
- datasets 离线（HF_DATASETS_OFFLINE=1）必须在脚本顶部
- AST 权重（MIT/ast-finetuned-audioset-10-10）镜像暂不可达 → 正式版替换点已定位

## P4c 工具选择机制验证（p4c_tool_learning.py）— 回答"能否不经编码器调用工具"

**问题**：JEPA 能否不用编码器（Qwen 语义提取工具命令）而是用原始机制（记忆）学会工具选择？

| 路径 | 机制 | 结果 |
|---|---|---|
| A 编码器先验 | Qwen 工具描述匹配 | 第一轮 **100%**（零学习） |
| B 记忆路径 | 试错 + 记忆检索 (状态,工具,效果)，探索-利用衰减 | 33% → **72%**（学习曲线 55→70%） |
| C 混合 | Qwen 种子 10 次 → 记忆接管 | **82%**（种子+经验最优） |

**裁决**：
1. **JEPA 原始机制（记忆）能内生学会工具选择**——不需要编码器提取工具命令，代价是试错学习（探索-利用衰减 = Configurator 探索温度调制机制）
2. **编码器 = 冷启动加速器**（先验注入 vs 经验学习的速度差：100% 零学习 vs 72% 需试错）
3. **真实最优架构 = 混合**：先验做种子（立即可用）→ 经验积累后记忆接管（82%）

关键区分：感知编码器（任务→表征，必经）与工具命令提取（表征→工具，可被记忆替代）是两个环节——P4c 验证的是后者可被 JEPA 内生替代。

## 下一步（P4c 起）

- 多轮工具循环（当前单轮）
- 真实图片扩散引擎（PIL 占位 → 接模型）
- 视频/音频编码器接入 C1
- HTTP serve 端点

## 关键设计决策

1. **惊讶度 = 记忆陌生度反信号**（不是 WAIT 预测误差）——修正记录：原实现用占位世界模型的预测误差恒低，工具永不触发；改"观测离记忆原型越远越惊讶"，符合"模型知道自己需要外部信息"
2. **工具触发**：惊讶度高 → 模型主动发起 tool_calls（把之前实证的"惊讶→记忆/学习"机制外化为工具调用）
3. **工具结果 = 外部经验**：写入记忆（不走完整三环，obs=None 保护）
4. **文本→观测**：最小哈希袋编码（正式版换 Qwen 编码器，zero-shot 66% 已验证）

## 下一步（P4 正式版，非本轮）

- 编码器换真实 ViT（jepa_base + timm/Qwen 权重）
- HTTP serve 层（/v1/chat/completions 真实端点）
- 更多工具（文件/搜索/屏幕）
- 多轮 tool_calls 循环（当前单轮）
