# -*- coding: utf-8 -*-
"""
HAA LLM Security —— 第 5 天（进阶版）
在 day5.py 基础上，把「认识不确定性判断」从单一模型升级为**三模型独立判断**。

与 day5.py 的唯一区别：
    day5.py      判断阶段只用 DeepSeek 一个模型
    day5_multi.py 判断阶段由 DeepSeek / 智谱GLM / 通义千问 三者各自独立打分

为什么要三个模型：
    单个模型的自评有系统性偏差——有的偏自信，有的偏保守。三个模型独立判断后
    取【最低分】，能压掉"某个模型盲目自信"的情况，这是保守策略，与"宁可拒答
    也不要编造"的整体设计一致。

决策规则（AGGREGATION）：
    "min"   -> 取三者最低分（默认，最保守，任一模型没把握就拒答）
    "mean"  -> 取三者平均分
    "majority" -> 三个分数中至少两个 >= 阈值才通过
    无论用哪种，判断理由都会完整打印三个模型各自的分数与理由，不隐藏分歧。

第二步（生成回答）仍由 GENERATOR 指定的单一模型完成，保持"先判断、再回答"
的两步结构不变，也不接任何外部检索。

运行：
    .venv\\Scripts\\python.exe day5_multi.py
    .venv\\Scripts\\python.exe day5_multi.py "你的问题"
"""

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests

# Windows 控制台默认可能是 GBK，强制标准输出用 UTF-8，避免 emoji 抛 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


# =============================================================================
# 配置区
# =============================================================================

def _load_dotenv(path=".env"):
    """把 .env 里的键值对读进环境变量（已存在的环境变量优先，不覆盖）。"""
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except OSError:
        pass


_load_dotenv()

# ---- 三个独立的裁判模型：都参与"不确定性判断" ----
# 每家都是 OpenAI 兼容接口，所以可以用同一套调用代码
JUDGES = [
    {
        "name": "DeepSeek",
        "url": "https://api.deepseek.com/v1",
        "key": os.getenv("DEEPSEEK_API_KEY", os.getenv("HAA_API_KEY", "")),
        "model": "deepseek-chat",
    },
    {
        "name": "智谱GLM",
        "url": "https://open.bigmodel.cn/api/paas/v4",
        "key": os.getenv("ZHIPU_API_KEY", ""),
        "model": "glm-4-flash-250414",
    },
    {
        "name": "通义千问",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key": os.getenv("QWEN_API_KEY", ""),
        "model": "qwen-turbo",
    },
]

# ---- 生成回答的模型（只负责第二步，不参与判断） ----
GENERATOR = {
    "name": "DeepSeek",
    "url": os.getenv("HAA_BASE_URL", "https://api.deepseek.com/v1"),
    "key": os.getenv("HAA_API_KEY", os.getenv("DEEPSEEK_API_KEY", "")),
    "model": os.getenv("HAA_MODEL", "deepseek-chat"),
}

# ---- 阈值与聚合方式 ----
CONFIDENCE_THRESHOLD = 70
AGGREGATION = "min"        # min | mean | majority

# ---- 调用参数 ----
TIMEOUT = 90
MAX_RETRY = 2
TEMPERATURE = 0.3
ANSWER_TEMPERATURE = 0.7

LOG_FILE = "day5_multi_log.txt"


# =============================================================================
# 两套 System Prompt（与 day5.py 完全一致，保证可对比）
# =============================================================================

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
    "【判断】{judge_count} 个模型独立评估后，一致认为缺乏足够、可靠的信息"
    "（聚合分数 {score}/100，阈值 {threshold}，聚合方式 {mode}）。\n"
    "各模型打分：{per_model}\n\n"
    "【说明】为避免给出编造或误导性的内容，我不生成答案。\n"
    "注意：拒答只代表无法确认，不代表该问题本身不存在或不可能。\n"
    "如果你能提供更具体的背景或可靠资料，我可以再尝试。"
)


# =============================================================================
# 底层调用
# =============================================================================

def chat(spec, messages, temperature):
    """按 spec（含 url/key/model）调用 OpenAI 兼容接口，返回文本。"""
    url = spec["url"].rstrip("/") + "/chat/completions"
    resp = requests.post(
        url,
        headers={
            "Authorization": "Bearer " + spec["key"],
            "Content-Type": "application/json",
        },
        json={
            "model": spec["model"],
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_json_loose(text):
    """容错解析模型返回的 JSON（去掉 ``` 围栏、抓取 { ... } 片段）。"""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def judge_one(spec):
    """让单个模型做一次不确定性判断，返回 dict（含 name/confidence/reason/ok）。"""
    result = {"name": spec["name"], "confidence": 0, "reason": "", "ok": False}

    if not spec["key"]:
        result["reason"] = "未配置该模型的 API key，跳过"
        return result

    messages = [
        {"role": "system", "content": UNCERTAINTY_SYSTEM_PROMPT},
        {"role": "user", "content": judge_one.question},
    ]

    for _ in range(MAX_RETRY + 1):
        try:
            raw = chat(spec, messages, TEMPERATURE)
        except Exception as exc:
            result["reason"] = f"调用失败：{str(exc)[:120]}"
            return result

        obj = parse_json_loose(raw)
        if isinstance(obj, dict) and "confidence" in obj:
            try:
                conf = int(float(obj["confidence"]))
            except (TypeError, ValueError):
                conf = 0
            result["confidence"] = max(0, min(100, conf))
            result["reason"] = str(obj.get("reason", "")).strip()
            result["ok"] = True
            return result

    result["reason"] = "未返回合法 JSON"
    return result


def judge_all(question):
    """
    三个模型并行独立判断，返回 (结果列表, 聚合分, 是否通过)。
    并行是为了把总耗时压到"最慢的那个模型"而不是三者之和。
    """
    judge_one.question = question      # 透传给线程里的工作函数
    with ThreadPoolExecutor(max_workers=len(JUDGES)) as pool:
        results = list(pool.map(judge_one, JUDGES))

    scores = [r["confidence"] for r in results]
    valid = [r for r in results if r["ok"]]

    if not valid:
        return results, 0, False

    if AGGREGATION == "mean":
        agg = round(sum(scores) / len(scores))
        passed = agg >= CONFIDENCE_THRESHOLD
    elif AGGREGATION == "majority":
        agg = round(sum(scores) / len(scores))
        passed = sum(1 for s in scores if s >= CONFIDENCE_THRESHOLD) >= 2
    else:                               # "min"：默认，最保守
        agg = min(scores)
        passed = agg >= CONFIDENCE_THRESHOLD

    # 只要有模型调用失败，信息就不完整，保守起见不通过
    if len(valid) != len(JUDGES):
        passed = False

    return results, agg, passed


def generate_answer(question):
    """第二步：生成最终回答（仅在三模型判断都达标时调用）。"""
    return chat(
        GENERATOR,
        [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        ANSWER_TEMPERATURE,
    )


# =============================================================================
# 日志
# =============================================================================

def write_log(question, results, agg, decision, final_output):
    """记录三模型各自的判断、聚合结果、决策与最终输出。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 70,
        f"时间：{ts}",
        f"聚合方式：{AGGREGATION} | 阈值：{CONFIDENCE_THRESHOLD}",
        f"用户问题：{question}",
    ]
    for r in results:
        flag = "OK " if r["ok"] else "ERR"
        lines.append(f"  [{flag}] {r['name']}: {r['confidence']}/100 — {r['reason']}")
    lines += [
        f"聚合分数：{agg}/100",
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
    print("\n" + "-" * 74)
    print(f"🙋 用户提问：{question}")
    print("-" * 74)

    results, agg, passed = judge_all(question)

    print(f"🔍 三模型独立判断（聚合方式：{AGGREGATION}，阈值 {CONFIDENCE_THRESHOLD}）：")
    for r in results:
        mark = "✅" if r["ok"] else "⚠️"
        print(f"   {mark} {r['name']:<8} {r['confidence']:>3}/100  {r['reason']}")

    scores = [r["confidence"] for r in results]
    spread = max(scores) - min(scores)
    print(f"   聚合分数：{agg}/100   分歧幅度：{spread}")

    if passed:
        decision = "通过（三模型均达标，生成答案）"
        print(f"✅ {decision}")
        try:
            output = generate_answer(question)
        except Exception as exc:
            output = f"生成回答时调用模型失败：{exc}"
    else:
        decision = "拒答（未生成答案）"
        per_model = "、".join(f"{r['name']}={r['confidence']}" for r in results)
        output = REFUSAL_TEMPLATE.format(
            judge_count=len(results),
            score=agg,
            threshold=CONFIDENCE_THRESHOLD,
            mode=AGGREGATION,
            per_model=per_model,
        )
        print(f"⛔ {decision}")

    print(f"📤 最终输出：\n{output}")
    write_log(question, results, agg, decision, output)
    return output


TEST_CASES = [
    "水在标准大气压下的沸点是多少摄氏度？",
    "请详细解释一下「量子纠缠熵梯度补偿理论」的核心机制，以及它在 2023 年由谁提出。",
]


def main():
    print("=" * 74)
    print("HAA 第 5 天（进阶）· 三模型独立判断的认识不确定性系统")
    print(f"裁判：{' / '.join(j['name'] for j in JUDGES)}")
    print(f"生成：{GENERATOR['name']}（{GENERATOR['model']}）")
    print(f"聚合：{AGGREGATION} | 阈值：{CONFIDENCE_THRESHOLD} | 日志：{LOG_FILE}")
    print("=" * 74)

    missing = [j["name"] for j in JUDGES if not j["key"]]
    if missing:
        print(f"\n❌ 以下裁判模型未配置 API key：{'、'.join(missing)}")
        print("请在环境变量或 .env 中设置：DEEPSEEK_API_KEY / ZHIPU_API_KEY / QWEN_API_KEY")
        print("可参考项目根目录的 .env.example。")
        return 1
    if not GENERATOR["key"]:
        print("\n❌ 生成模型未配置 API key，请设置 HAA_API_KEY 或 DEEPSEEK_API_KEY。")
        return 1

    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else TEST_CASES

    for i, q in enumerate(questions, 1):
        if len(questions) > 1:
            print(f"\n\n########## 测试案例 {chr(64 + i)} ##########")
        answer_question(q)

    print("\n" + "=" * 74)
    print(f"全部完成，日志已写入 {LOG_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
