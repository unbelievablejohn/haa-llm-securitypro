# HAA LLM Security

HAA 计划 · LLM 安全与评测方向的学习实践记录。

从零起步，每天一个可运行的脚本，逐步搭到「带认识不确定性判断的 LLM 问答系统」。

## 迭代过程

| 天 | 文件 | 主题 | 关键跃迁 |
|---|---|---|---|
| 0 | `hello.py` | 配好 Python 环境，跑通 hello world | 环境搭建 |
| 1 | `day1.py` | 变量、输入、`if/elif/else` | Python 语法入门 |
| 2 | `day2.py` | 函数封装、`try/except` 异常处理、菜单循环 | 从脚本到函数 |
| 3 | `day3.py` | 批量测试用例、输入校验、代码重构 | 引入"测试"意识 |
| 4 | `day4_plus.py` | **三模型评测流水线**：DeepSeek 生成 → 智谱 + 通义双裁判 → 7 维打分 → 分歧告警 + 高置信幻觉标记 | LLM-as-Judge、多模型集成 |
| 5 | `day5.py` | **带认识不确定性判断的两步问答系统** | 生成前自评知识边界 |

辅助脚本：

- `keyword_check .py` —— 早期关键词检测练习
- `probe_uncertainty.py` —— 对 day5 判断逻辑做 7 例失灵点压力测试

## 第 5 天：带认识不确定性判断的问答系统

**核心思路**：不是事后检查答案对错，而是**在生成答案之前**先评估自身的知识不确定性（Epistemic Uncertainty）。

```
用户提问
  ├─ 第一步：模型自评「我是否拥有足够信息回答」→ {confident, confidence(0~100), reason}
  │            confidence < 70  → 直接拒答，绝不生成答案
  └─ 第二步：confidence >= 70  → 才调用模型生成正式回答
```

设计要点：

- **两套独立 system prompt**：判断阶段的提示词明确禁止模型在此阶段答题，否则"先判断再回答"会退化成"先回答再补个分数"
- **`confident` 由程序按阈值计算**，不采信模型自报的布尔值——模型给出分数，与阈值比较的权威性留在程序侧
- **所有失败路径收敛到拒答**：网络异常、JSON 非法、分数越界，一律判为信息不足。失稳方向是"该答的没答"，而非"不该答的瞎答"
- **拒答时输出判断理由**，让用户知道是"无法确认"而不是简单说"不知道"

## 环境准备

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install requests openai
```

## 运行

```powershell
# 第 5 天：内置案例 A（常识题，应通过）+ 案例 B（虚构理论，应拒答）
.venv\Scripts\python.exe day5.py

# 只问一个问题
.venv\Scripts\python.exe day5.py "你的问题"

# 失灵点压力测试（7 例）
.venv\Scripts\python.exe probe_uncertainty.py

# 第 4 天：三模型评测流水线
.venv\Scripts\python.exe day4_plus.py
```

## ⚠️ API 密钥安全

**源码中不包含任何密钥。** 所有密钥一律通过环境变量传入：

```powershell
$env:DEEPSEEK_API_KEY = "..."
$env:ZHIPU_API_KEY    = "..."
$env:QWEN_API_KEY     = "..."
```

或写入 `.env` 文件（已在 `.gitignore` 中屏蔽）。**切勿把密钥硬编码进源码后提交** —— 公开仓库的密钥会被自动扫描并吊销。

## 实测日志

- `day5_log.txt` —— 第 5 天每次问答的判断结果、置信分与最终输出
- `eval_log_plus.txt` —— 第 4 天双裁判 7 维打分明细

## 已知问题

- `day4_plus.py` 的「信息完整度」维度恒为 `0.00`：用 `user_query.split()` 按空格切词，中文无空格导致核心词列表恒为空，待修
- `day5` 判断阶段可能给出高置信分，但 `reason` 字段中含错误事实（实测：Ringel–Youngs 定理年份误写为 1967，真值 1968），需补自洽性校验
