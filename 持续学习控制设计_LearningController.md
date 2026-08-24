# 持续学习形式与用户控制设计（LearningController）

日期：2026-08-24 ｜ 依据：P4c 工具选择学习、P4b 蒸馏、JPI 实证链（两级能量/惊讶门控/记忆回放/探索衰减）

## 一、持续学习的形式：三层模式

| 模式 | 时机 | 机制 | 数据源 | 成本 |
|---|---|---|---|---|
| **在线学习**（默认） | 每次交互后 | AdaJEPA 1 步梯度 + 惊讶门控记忆写入 + 工具经验回传 | 实时交互流 | 最低（毫秒级） |
| **批量巩固**（用户触发/定时） | 积累 N 条经验后 | 记忆回放（recent-N/优先级采样）+ 蒸馏巩固（学生逼近教师） | 记忆库 + 外部权重 | 中（分钟级） |
| **离线蒸馏**（用户触发） | 接入新教师权重时 | 多模态蒸馏（multi_modal_distill 已验证）：新权重教师 → 学生巩固 | 对齐数据集 | 高（小时级） |

**核心区别**：在线学习 = 微调当前模型（快、增量）；批量巩固 = 用记忆回放强化（防遗忘，jpi5 零遗忘实证）；离线蒸馏 = 吸收外部大模型知识到小模型（压缩，本轮实证）。

## 二、JEPA 独有的"巩固"含义（蒸馏合并压缩）

```
外部大模型 (教师):  timm 86M + Qwen 0.5B + AST 86M ≈ 0.67B
        ↓ 多模态蒸馏 (multi_modal_distill.py 实证)
内部小模型 (学生):  统一多模态编码器 ≈ 0.5M (压缩 ~600×, 推理快 ~10×)
        ↓ 持续在线巩固 (AdaJEPA 1 步梯度 + 记忆回放)
学生持续精化, 教师可退役
```

**"巩固"= 学生模型用真实经验持续微调**（不是重训）——这正是 JEPA 的"慢快双过程"：快过程（在线 1 步梯度）持续，慢过程（批量蒸馏/回放）定期。

## 三、用户控制接口（LearningController）

```python
class LearningController:
    """持续学习的用户开关与干预"""
    def __init__(self, body):
        self.body = body
        self.mode = "online"          # online | batch | distill | off
        self.online_enabled = True    # 在线学习开关
        self.batch_threshold = 200    # 记忆满 N 条触发批量巩固
        self.schedule = {"batch": False, "distill": False}   # 定时任务

    # ── 用户开关 ──────────────────────────────────────
    def pause(self):    self.online_enabled = False   # 冻结学习 (推理不受影响)
    def resume(self):   self.online_enabled = True
    def set_mode(self, mode): ...                     # 切换在线/批量/离线
    def run_batch_consolidation(self, n=None): ...    # 手动触发记忆回放巩固
    def run_distill(self, teacher_spec): ...          # 手动接入新教师蒸馏

    # ── 用户干预 ──────────────────────────────────────
    def inject_alignment(self, samples):              # 用户提供对齐样本 (纠偏)
        """干预 = 提供 (状态, 期望输出) 对, 立即进入记忆并高权重回放"""
    def correct(self, state, expected):               # 用户纠错 (单点)
        """错误反馈 → 立即学习 + 标记记忆条目高价值"""
    def reset_memory(self):                           # 清空经验 (重新开始)
```

## 四、用户介入的三种语义

| 介入 | 用户动作 | 系统响应 |
|---|---|---|
| 开关 | pause/resume | 冻结/恢复学习（推理不断） |
| 模式 | set_mode | 切换学习粒度（在线/批量/离线） |
| 干预 | correct / inject_alignment | 高权重经验注入，立即生效（比普通经验更快影响） |

**设计原则**：学习与推理解耦（pause 时推理照常）；用户纠错 > 自动经验（纠错样本高权重）；新知识先蒸馏后在线（先接入教师，再持续微调）。

## 五、落地到 JepaBody

```python
body = JepaBody()
body.controller = LearningController(body)   # 默认 online
body.learn(...)                               # 内部: controller.online_enabled 检查
body.controller.run_distill({"image": "timm", "text": "qwen"})   # 离线蒸馏
```

## 六、验证状态

- ✅ 在线学习：AdaJEPA 1 步梯度（P1）+ 惊讶门控记忆（P2/P3）
- ✅ 批量巩固：记忆回放（jpi10 稀疏奖励 +16.7pp；AdaJEPA recent-N）
- ✅ 离线蒸馏：本轮 multi_modal_distill（跨模态检索 + 压缩）
- ⬜ LearningController 完整实现（接口已设计）
