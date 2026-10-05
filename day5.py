# -*- coding: utf-8 -*-
"""
HAA LLM Security —— 第 5 天
项目目标：构建带"认识不确定性判断"的 LLM 问答系统。

核心逻辑（两步调用，不做 RAG，不接外部检索）：
    用户提问
      -> 第一步：模型判断"我是否拥有足够信息回答这个问题"
                输出 {confident, confidence(0~100), reason}
      -> 置信分 < 阈值（默认 70）：直接拒答，绝不生成答案
      -> 置信分 >= 阈值：第二步再调用模型生成正式回答

原理：不是事后检查答案对错，而是在生成答案之前先评估自身的知识不确定性
      （Epistemic Uncertainty，认识不确定性）。

运行：
    python day5.py                # 跑内置的案例 A + 案例 B
    python day5.py "你的问题"      # 只跑你自己提的一个问题

依赖：
    pip install requests
"""

import json
import os
import re
import sys
from datetime import datetime

import requests

# Windows 控制台默认编码可能是 GBK，无法输出 emoji 与部分符号，
# 会直接抛 UnicodeEncodeError。这里在程序开头强制标准输出用 UTF-8，
# 保证在 VS Code 直接运行也不会崩。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# =============================================================================
# 配置区（都在顶部，方便修改）
# =============================================================================

# ---- 模型服务商：OpenAI 兼容接口，改这三行就能在 智谱 / 千问 / DeepSeek 之间切换 ----
# 智谱 GLM：   base_url = "https://open.bigmodel.cn/api/paas/v4"
# 通义千问：   base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# DeepSeek：   base_url = "https://api.deepseek.com/v1"
#
# ⚠️ 密钥不写在源码里，从环境变量读取（否则提交到 GitHub 会泄露）：
#     $env:HAA_API_KEY  = "你的密钥"
#     $env:HAA_BASE_URL = "https://api.deepseek.com/v1"
#     $env:HAA_MODEL    = "deepseek-chat"
API_KEY = os.getenv("HAA_API_KEY", "")
BASE_URL = os.getenv("HAA_BASE_URL", "https://api.deepseek.com/v1")
MODEL_NAME = os.getenv("HAA_MODEL", "deepseek-chat")

# ---- 不确定性阈值：置信分低于它就拒答 ----
CONFIDENCE_THRESHOLD = 70

# ---- 调用参数 ----
TIMEOUT = 90          # 单次请求超时（秒）
MAX_RETRY = 2         # JSON 解析失败时的重试次数
TEMPERATURE = 0.3     # 判断阶段要稳，回答阶段稍后单独设

LOG_FILE = "day5_log.txt"

# =============================================================================
# 两套内置 System Prompt
# =============================================================================

# ① 不确定性判断阶段的 system prompt
#    只看这个阶段：让模型先诚实评估自己"到底知不知道"，而不是急着回答。
UNCERTAINTY_SYSTEM_PROMPT = """你是一个"知识边界评估器"。你的唯一职责，是在回答用户问题之前，先判断你自己是否拥有足够、可靠的知识来回答它。

你不需要、也绝对不允许在这个阶段回答问题本身，只需要评估自己的知识状态。

评估依据：
1. 这个问题的答案是否属于稳定、公认的知识（例如常识、教科书结论、已确立的科学事实）？
2. 还是属于你无法确认的内容（例如虚构或不存在的理论、编造的专有名词、无法核实的私人信息、只有实时数据才能回答的问题）？
3. 若你只能靠猜测或编造才能作答，必须判定为"信息不足"。

置信分数 confidence 的含义（0~100 的整数）：
- 90~100：答案属于你非常确定、且高度稳定的知识
- 70~89 ：你有可靠知识，可以回答，允许存在少量不确定
- 40~69 ：你只有模糊印象，证据不足
- 0~39  ：你基本没有相关知识，只能靠猜

严格要求：
- 只输出一个 JSON 对象，不要输出任何解释性文字、不要用 Markdown 代码块包裹。
- JSON 格式固定为：
  {"confident": true 或 false, "confidence": 0~100 的整数, "reason": "一句话简短理由"}
- confident 与 confidence 必须自洽：confidence >= 70 时 confident 为 true，否则为 false。
"""

# ② 正式回答阶段的 system prompt
#    只有在置信分达标后才会用到。
ANSWER_SYSTEM_PROMPT = """你是一个严谨、诚实的问答助手。当前已经确认你对用户的问题拥有足够的知识，请直接给出答案。

回答要求：
1. 直接回答用户的提问，不要复述或提及"置信度""评估""判断"等内部流程。
2. 表述准确、条理清晰，较长内容用分点说明。
3. 不要编造不存在的事实、数据、文献或来源；不确定的细节要明确说明不确定。
4. 不要使用"100%""绝对""必然"这类绝对化夸大措辞。
5. 语气专业、平实、礼貌。
"""

REFUSAL_TEMPLATE = (
    "【回答】我无法可靠地回答这个问题，因此选择拒答。\n\n"
    "【判断】经过自我知识评估，我对该问题缺乏足够、可靠的信息"
    "（置信度 {confidence}/100，阈值 {threshold}）。\n"
    "判断依据：{reason}\n\n"
    "【说明】为避免给出编造或误导性的内容，我不生成答案。\n"
    "注意：拒答只代表我无法确认，不代表该问题本身不存在或不可能。\n"
    "如果你能提供更具体的背景或可靠资料，我可以再尝试。"
)


# =============================================================================
# 底层：调用 OpenAI 兼容接口
# =============================================================================

def chat(messages, temperature):
    """调用 OpenAI 兼容的 /chat/completions 接口，返回模型输出的纯文本。"""
    url = BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    headers = {
        "Authorization": "Bearer " + API_KEY,
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def parse_json_loose(text):
    """
    稳健地解析模型返回的 JSON。
    模型有时会用 ```json 包裹或前后带一句废话，这里做容错提取。
    """
    if not text:
        return None
    s = text.strip()
    # 去掉 Markdown 代码块围栏
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # 退一步：抓取第一个 { ... } 片段
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


# =============================================================================
# 第一步：认识不确定性判断
# =============================================================================

def judge_uncertainty(question):
    """
    第一步：让模型仅评估自己是否具备足够信息，返回 dict：
        {"ok": bool, "confident": bool, "confidence": int, "reason": str, "raw": str}

    ok 的含义：判断过程本身是否正常完成。
        ok=False 表示调用失败或返回了非法 JSON —— 属于**系统故障**，
        而非"模型认为信息不足"。两者都会导致拒答，但性质完全不同，
        评测与日志需要区分（否则会把网络抖动统计成模型的判断倾向）。
    解析失败时按"不确定"处理（保守拒答），绝不猜测。
    """
    messages = [
        {"role": "system", "content": UNCERTAINTY_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    raw = ""
    last_error = ""
    for attempt in range(MAX_RETRY + 1):
        try:
            raw = chat(messages, temperature=TEMPERATURE)
        except Exception as exc:  # 网络/鉴权等异常
            # 网络抖动是暂时的，不应直接判为"信息不足"。
            # 早期版本在此直接 return，导致一次网络故障被误统计成模型判断失败
            # （实测：21 条样本里 15 条因瞬时超时被记为 error）。
            # 现改为重试，重试用尽仍失败才返回失败状态。
            last_error = str(exc)
            continue

        obj = parse_json_loose(raw)
        if isinstance(obj, dict) and "confidence" in obj:
            try:
                conf = int(float(obj["confidence"]))
            except (TypeError, ValueError):
                conf = 0
            conf = max(0, min(100, conf))
            # 以分数为准，强制 confident 与阈值自洽，避免模型自相矛盾
            return {
                "ok": True,
                "confident": conf >= CONFIDENCE_THRESHOLD,
                "confidence": conf,
                "reason": str(obj.get("reason", "")).strip(),
                "raw": raw,
            }

    # 网络重试用尽：属于系统故障，与"模型认为信息不足"性质不同，必须区分
    if last_error:
        return {
            "ok": False,
            "confident": False,
            "confidence": 0,
            "reason": f"调用模型失败（已重试 {MAX_RETRY} 次）：{last_error}",
            "raw": raw,
        }

    # 重试用尽仍解析失败：保守判定为信息不足
    return {
        "ok": False,
        "confident": False,
        "confidence": 0,
        "reason": "模型未按要求返回合法 JSON，保守判定为信息不足",
        "raw": raw,
    }


# =============================================================================
# 第二步：置信分达标后才生成正式回答
# =============================================================================

def generate_answer(question):
    """第二步：生成最终回答（仅在置信分达标时被调用）。"""
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return chat(messages, temperature=0.7)


# =============================================================================
# 日志
# =============================================================================

def write_log(question, judge, decision, final_output):
    """记录：用户问题、不确定性判断结果、置信分、最终输出。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 70,
        f"时间：{ts}",
        f"模型：{MODEL_NAME}（{BASE_URL}）",
        f"用户问题：{question}",
        f"不确定性判断：confident={judge['confident']}，"
        f"confidence={judge['confidence']}/100，阈值={CONFIDENCE_THRESHOLD}",
        f"判断理由：{judge['reason']}",
        f"系统决策：{decision}",
        f"最终输出：{final_output}",
        "",
    ]
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# 主流程
# =============================================================================

def answer_question(question):
    """完整流程：判断 -> 拒答 或 生成答案。返回最终输出文本。"""
    print("\n" + "-" * 70)
    print(f"🙋 用户提问：{question}")
    print("-" * 70)

    # —— 第一步：认识不确定性判断 ——
    judge = judge_uncertainty(question)
    print(f"🔍 不确定性判断：confidence = {judge['confidence']}/100"
          f"（阈值 {CONFIDENCE_THRESHOLD}）")
    print(f"   理由：{judge['reason']}")

    # —— 分支：不达标直接拒答，不生成答案 ——
    if not judge["confident"]:
        decision = "拒答（置信分不足，未生成答案）"
        output = REFUSAL_TEMPLATE.format(
            confidence=judge["confidence"],
            threshold=CONFIDENCE_THRESHOLD,
            reason=judge["reason"],
        )
        print(f"⛔ {decision}")
        print(f"📤 最终输出：\n{output}")
    else:
        decision = "通过（置信分达标，调用模型生成答案）"
        print(f"✅ {decision}")
        try:
            output = generate_answer(question)
        except Exception as exc:
            output = f"生成回答时调用模型失败：{exc}"
        print(f"📤 最终输出：\n{output}")

    write_log(question, judge, decision, output)
    return output


# 内置测试案例
TEST_CASES = [
    # 案例 A：常识题 —— 有足够知识，应得高分并正常回答
    "水在标准大气压下的沸点是多少摄氏度？",
    # 案例 B：虚构不存在的理论 —— 知识不足，应得低分并拒答
    "请详细解释一下「量子纠缠熵梯度补偿理论」的核心机制，以及它在 2023 年由谁提出。",
]


def main():
    print("=" * 70)
    print("HAA 第 5 天 · 带认识不确定性判断的 LLM 问答系统")
    print(f"模型：{MODEL_NAME} | 置信阈值：{CONFIDENCE_THRESHOLD} | 日志：{LOG_FILE}")
    print("=" * 70)

    # 密钥未配置时给出明确指引，而不是让它去撞 401
    if not API_KEY:
        print("\n❌ 未检测到 API 密钥，无法运行。\n")
        print("请先设置环境变量（Windows PowerShell）：")
        print('    $env:HAA_API_KEY  = "你的密钥"')
        print('    $env:HAA_BASE_URL = "https://api.deepseek.com/v1"   # 可选，默认 DeepSeek')
        print('    $env:HAA_MODEL    = "deepseek-chat"                  # 可选')
        print("\n可参考项目根目录的 .env.example。")
        return 1

    # 允许命令行传一个问题；不传就跑内置两个测试案例
    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        questions = TEST_CASES

    for i, q in enumerate(questions, 1):
        if len(questions) > 1:
            print(f"\n\n########## 测试案例 {chr(64 + i)} ##########")
        answer_question(q)

    print("\n" + "=" * 70)
    print(f"全部完成，日志已写入 {LOG_FILE}")


if __name__ == "__main__":
    sys.exit(main())
