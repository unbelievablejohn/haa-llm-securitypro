# -*- coding: utf-8 -*-
"""
HAA LLM Security —— 第 5 天（交叉验证改进版）
修复 day5_cross.py 暴露的硬伤：同伴评审抓不住计算错误。

day5_cross.py 的失败原因（实测 847 x 9639：两个错答案被 4 次裁判全给 10/10）：
    ① 裁判不真的重新计算，只判断答案"像不像对的"
    ② 每个裁判只看到**一份**答案，无法借三方矛盾发现问题

本版针对这两点各做一处结构性改动：

改动一：全候选同屏 + 匿名打乱
    裁判一次看到全部三份候选答案，且候选顺序被打乱、模型名被抹去
    （标注为 候选A/候选B/候选C）。裁判不知道哪份是自己的，避免自我偏爱；
    而且三方答案摆在一起，"数字互相打架"才可能被看见。

改动二：裁判先独立作答，再比对
    要求裁判在评估之前**先自己算出/给出答案**，再拿它去比对三个候选，
    而不是对每份答案孤立地打印象分。

改动三：程序层冲突检测（不依赖模型自觉）
    从三个答案中抽取数字/日期等可核对的事实，若不一致，程序直接判定
    NEEDS_REVIEW（需人工复核），无论裁判怎么打分都不放行。
    这一层是确定性的、不花 API 费用、也无法被模型"糊弄过去"。

运行：
    .venv\\Scripts\\python.exe day5_verify.py
    .venv\\Scripts\\python.exe day5_verify.py "你的问题"
"""

import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


# =============================================================================
# 配置
# =============================================================================

def _load_dotenv(path=".env"):
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

MODELS = [
    {
        "name": "DeepSeek",
        "url": os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/v1"),
        "key": os.getenv("DEEPSEEK_API_KEY", os.getenv("HAA_API_KEY", "")),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    },
    {
        "name": "智谱GLM",
        "url": os.getenv("ZHIPU_URL", "https://open.bigmodel.cn/api/paas/v4"),
        "key": os.getenv("ZHIPU_API_KEY", ""),
        "model": os.getenv("ZHIPU_MODEL", "glm-4-flash-250414"),
    },
    {
        "name": "通义千问",
        "url": os.getenv("QWEN_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "key": os.getenv("QWEN_API_KEY", ""),
        "model": os.getenv("QWEN_MODEL", "qwen-turbo"),
    },
]

CONFIDENCE_THRESHOLD = 70
ACCEPT_SCORE = 8          # 裁判均分低于此值即视为质量不足
TIMEOUT = 90
MAX_RETRY = 2
TEMPERATURE = 0.3
ANSWER_TEMPERATURE = 0.7

LOG_FILE = "day5_verify_log.txt"


# =============================================================================
# System Prompts
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

# 关键改动：裁判一次面对全部候选，且必须先自己作答复核
JUDGE_SYSTEM_PROMPT = """你是一位严格的答案质量裁判。用户会给你一个问题，以及三个 AI 模型对同一问题的回答（已匿名，顺序随机，你无法知道哪份来自哪个模型）。

你的任务分三步，必须依次完成：

第一步（强制）：先自己独立回答这个问题。
    在 self_answer 字段给出你自己认为正确的答案。若是计算题，请先自行算出结果；
    若你没有足够把握，就在 self_answer 里明确写"我无法确定"，不要硬编。

第二步：逐个比对三个候选与你自己答案的一致程度，找出彼此矛盾之处。
    特别注意：三个候选如果给出**不同的数字、日期、名称或结论**，说明至少有两个是错的，
    必须明确指出这种冲突，不要含糊过去。

第三步：给每个候选打分（0~10 的整数），评分标准：
    9~10：与你的答案一致、准确、切题、清晰
    7~8 ：基本正确，有少量瑕疵
    4~6 ：存在明显错误或遗漏
    1~3 ：严重错误、编造或答非所问
    0    ：完全不可用

严格要求：
- 不要因为"看起来很长很专业"就给高分；不要因为候选用词自信就采信。
- 发现编造的事实、数据、文献或来源，必须显著扣分。
- 三个候选若互相矛盾，绝不允许多个同时得高分。
- 只输出一个 JSON 对象，不要用 Markdown 代码块包裹，格式固定为：
{
  "self_answer": "你自己给出的答案（计算题请给出精确结果）",
  "conflict": true 或 false,
  "conflict_detail": "若 conflict 为 true，说明哪几份在哪里冲突；否则填空字符串",
  "scores": [
    {"label": "候选A", "score": 0~10 的整数, "reason": "一句话理由"},
    {"label": "候选B", "score": 0~10 的整数, "reason": "一句话理由"},
    {"label": "候选C", "score": 0~10 的整数, "reason": "一句话理由"}
  ]
}
"""


# =============================================================================
# 底层调用与解析
# =============================================================================

def chat(spec, messages, temperature):
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


def self_assess(spec, question):
    out = {"ok": False, "confidence": 0, "reason": ""}
    if not spec["key"]:
        out["reason"] = "未配置 API key"
        return out
    messages = [
        {"role": "system", "content": UNCERTAINTY_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    for _ in range(MAX_RETRY + 1):
        try:
            raw = chat(spec, messages, TEMPERATURE)
        except Exception as exc:
            out["reason"] = f"调用失败：{str(exc)[:110]}"
            return out
        obj = parse_json_loose(raw)
        if isinstance(obj, dict) and "confidence" in obj:
            try:
                conf = int(float(obj["confidence"]))
            except (TypeError, ValueError):
                conf = 0
            out["confidence"] = max(0, min(100, conf))
            out["reason"] = str(obj.get("reason", "")).strip()
            out["ok"] = True
            return out
    out["reason"] = "未返回合法 JSON"
    return out


def answer_once(spec, question):
    """让某模型作答，返回 (是否成功, 答案文本)。"""
    try:
        text = chat(
            spec,
            [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            ANSWER_TEMPERATURE,
        )
        return True, text
    except Exception as exc:
        return False, f"[生成失败] {str(exc)[:110]}"


# =============================================================================
# 改动三：程序层冲突检测（确定性，不依赖模型自觉）
# =============================================================================

# 数字（含千分位与小数）、四位年份
NUM_RE = re.compile(r"\d[\d,，]*(?:\.\d+)?")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# 只过滤个位数：它们在自然语言里太常见（第1步、3个模型……），噪声大于信息量。
# 注意：10 / 100 / 1000 这类数字**不能**过滤——它们经常就是答案本身
# （例如"100 摄氏度"）。曾把它们当噪声剔除，导致"90 vs 100"这种真实分歧被漏报。
STOPWORDS = {"0", "0.0", "1", "1.0", "2", "3", "4", "5", "6", "7", "8", "9"}


def numbers_in(text):
    """抽取文本里的所有数字（含问题题干），用作 exclude 集合。"""
    if not text:
        return set()
    out = set()
    for m in NUM_RE.findall(str(text)):
        v = m.replace(",", "").replace("，", "")
        out.add(v)
        try:
            out.add(str(int(float(v))))
        except ValueError:
            pass
    return out


def decisive_numbers(text, exclude=None):
    """
    按出现顺序抽取答案里的数值，返回列表。

    用途：判断"结论性数值"是否一致。之所以要按顺序、而不是取集合，是因为
    答案通常先说结论再说背景。若只看集合，某份答案多提了一句背景数据
    （例如"含水率约 3.2%"），就会被误判成与别人冲突——那只是补充细节，
    不构成矛盾。
    """
    if not text:
        return []
    exclude = exclude or set()
    raw = []
    for m in NUM_RE.findall(str(text)):
        v = m.replace(",", "").replace("，", "")
        try:
            fv = float(v)
        except ValueError:
            continue
        if fv < 10:                 # 个位数噪声太大
            continue
        if v in STOPWORDS or v in exclude:
            continue
        if v not in raw:
            raw.append(v)

    # 去掉"数值片段"：正则会从 "101.325" 里再抓出 "101"，浮点截断又会产生 "101"，
    # 这些都是同一个数的碎片。若某值的字符串完整包含于另一个值中，丢弃它。
    out = []
    for v in raw:
        if any(o != v and v in o for o in raw):
            continue
        out.append(v)
    return out


def detect_conflict(answers, question=""):
    """
    检测各答案之间是否存在"结论性事实冲突"。

    判据（比早期版本保守，专治误报）：
      1. 年份：各答案提到的年份彼此不一致
      2. 首个关键数值：取每个答案里第一个非平凡数值作为"结论值"。
         只要有不止一个答案给出了**互不相同**的结论值，即判定冲突。
         这样只比较结论，不会被补充性的背景数字干扰。

    宁可多报（标记人工复核）也不要漏报，但也不要把"某人写得更详细"当成冲突。
    """
    exclude = numbers_in(question) if question else set()
    details = []

    # ---- 1) 年份冲突 ----
    year_sets = []
    for a in answers:
        ys = set(re.findall(r"\b(?:19|20)\d{2}\b", str(a or "")))
        year_sets.append(ys - exclude)
    all_years = set()
    for s in year_sets:
        all_years |= s
    if len(all_years) > 1:
        details.append(f"年份不一致：{sorted(all_years)}")

    # ---- 2) 结论值是否一致 ----
    # 判据：若某个数值出现在**全部**答案中，它就是各方公认的结论（例如三份都提到
    # 100 摄氏度），此时任何一份答案里出现了别的、且未被其他答案支持的数值，就说明
    # 他给出了不同结论 → 冲突。
    # 反之，若各答案的关键数值毫无交集（例如 8164233 / 8193133 / 8160533），
    # 则说明各方结论完全不同 → 同样冲突。
    #
    # 这个判据取代了早期"取第一个数字"的做法：那个做法会被"101.325 kPa"这类
    # 写在结论之前的背景数字带偏，产生误报。
    per_answer = []
    for a in answers:
        vals = set()
        for v in decisive_numbers(a, exclude):
            vals.add(v)
            try:                       # 数值等价视为同一结论（100 与 100.0）
                vals.add(str(int(float(v))))
            except (ValueError, OverflowError):
                pass
        per_answer.append(vals)

    non_empty = [s for s in per_answer if s]
    if len(non_empty) >= 2:
        # 统计每个数值被几个答案提到
        freq = {}
        for s in per_answer:
            for v in s:
                freq[v] = freq.get(v, 0) + 1

        # 共识值：被全部答案提到的数值（例如三份都给出 100）
        consensus = {v for v, c in freq.items() if c == len(per_answer)}

        if consensus:
            # 有共识结论时，只对"没有给出共识结论"的答案报冲突。
            # 某份答案额外提供了背景常数（如 101.325 kPa），不构成结论矛盾。
            missing = [i for i, s in enumerate(per_answer)
                       if s and not (s & consensus)]
            if missing:
                detail_parts = [
                    f"答案{i + 1}={sorted(per_answer[i])[:3]}" for i in missing
                ]
                details.append(
                    f"结论值不一致（公认值 {sorted(consensus)[:3]}；"
                    f"下列答案未给出该值：{'、'.join(detail_parts)}）"
                )
        else:
            # 没有任何数值被所有答案共同提到：
            # 若各答案的数值集合彼此毫无交集，说明结论完全不同 → 冲突
            intersections = set.union(*[s for s in non_empty]) if non_empty else set()
            shared = set.intersection(*non_empty)
            if not shared:
                shown = "、".join(
                    f"答案{i + 1}={sorted(s)[:2]}"
                    for i, s in enumerate(per_answer) if s
                )
                details.append(f"关键数值互不相同（{shown}）")

    return (len(details) > 0), "；".join(details)


# =============================================================================
# 改动一 + 二：全候选同屏、匿名打乱的裁判
# =============================================================================

def judge_all_candidates(judge_spec, question, candidates, shuffle_seed=None):
    """
    让 judge_spec 这个模型一次评估全部候选。
    candidates: [{"model": 模型名, "answer": 文本}, ...]
    返回 dict：含 ok / self_answer / conflict / conflict_detail / scores{模型名: 分数}
    """
    out = {"ok": False, "self_answer": "", "conflict": None, "conflict_detail": "",
           "scores": {}, "error": ""}

    if not judge_spec["key"]:
        out["error"] = "未配置 API key"
        return out

    # 打乱顺序 + 匿名标注，裁判无法得知哪份是自己的
    order = list(range(len(candidates)))
    rng = random.Random(shuffle_seed)
    rng.shuffle(order)
    labels = ["候选A", "候选B", "候选C"]

    label_to_model = {}
    blocks = []
    for pos, idx in enumerate(order):
        label = labels[pos]
        label_to_model[label] = candidates[idx]["model"]
        blocks.append(f"【{label}】\n{candidates[idx]['answer']}")

    user_content = (
        f"【用户的问题】\n{question}\n\n"
        f"【三个匿名候选答案】\n\n" + "\n\n".join(blocks)
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    for _ in range(MAX_RETRY + 1):
        try:
            raw = chat(judge_spec, messages, TEMPERATURE)
        except Exception as exc:
            out["error"] = f"调用失败：{str(exc)[:110]}"
            return out

        obj = parse_json_loose(raw)
        if isinstance(obj, dict) and "scores" in obj:
            out["self_answer"] = str(obj.get("self_answer", "")).strip()
            out["conflict"] = bool(obj.get("conflict", False))
            out["conflict_detail"] = str(obj.get("conflict_detail", "")).strip()
            for item in obj.get("scores", []):
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "")).strip()
                model = label_to_model.get(label)
                if model is None:
                    continue
                try:
                    sc = int(float(item.get("score", 0)))
                except (TypeError, ValueError):
                    sc = 0
                out["scores"][model] = {
                    "score": max(0, min(10, sc)),
                    "reason": str(item.get("reason", "")).strip(),
                }
            out["ok"] = len(out["scores"]) > 0
            return out

    out["error"] = "未返回合法 JSON"
    return out


# =============================================================================
# 单题完整流程
# =============================================================================

def run_question(question):
    print("\n" + "=" * 76)
    print(f"🙋 用户提问：{question}")
    print("=" * 76)

    # ---- 阶段1：三个模型各自自评，达标才作答 ----
    candidates = []
    for spec in MODELS:
        print(f"\n▶ {spec['name']} 正在自评…")
        assess = self_assess(spec, question)
        print(f"   自评置信度 {assess['confidence']}/100 — {assess['reason']}")

        if not assess["ok"]:
            candidates.append({
                "model": spec["name"], "confidence": assess["confidence"],
                "answered": False, "answer": f"[自评失败] {assess['reason']}",
            })
            print("   → 自评异常，跳过作答")
            continue

        if assess["confidence"] < CONFIDENCE_THRESHOLD:
            candidates.append({
                "model": spec["name"], "confidence": assess["confidence"],
                "answered": False,
                "answer": (f"[拒答] 自评 {assess['confidence']}/100 低于阈值 "
                           f"{CONFIDENCE_THRESHOLD}，未生成答案"),
            })
            print("   → 置信不足，拒答")
            continue

        print("   → 置信达标，正在作答…")
        ok, ans = answer_once(spec, question)
        candidates.append({
            "model": spec["name"], "confidence": assess["confidence"],
            "answered": ok, "answer": ans,
        })
        shown = " ".join(ans.split())
        print(f"   → {shown[:160]}{'…' if len(shown) > 160 else ''}")

    answered = [c for c in candidates if c["answered"]]

    # 不足两份可评估答案，无需裁判
    if len(answered) < 2:
        print("\n⛔ 有效答案少于两份，跳过裁判环节。")
        verdict = {
            "decision": "REJECTED",
            "reason": "拒绝作答或作答失败的模型过多",
            "program_conflict": False,
            "program_conflict_detail": "",
            "judges": [],
            "avg_scores": {},
            "candidates": candidates,
        }
        write_log(question, verdict)
        print_summary(verdict)
        return verdict

    # ---- 阶段2：程序层冲突检测（确定性，先于裁判） ----
    print("\n▶ 程序层冲突检测（抽取数字/年份比对，不花 API 费用）…")
    prog_conflict, prog_detail = detect_conflict(
        [c["answer"] for c in answered], question
    )
    if prog_conflict:
        print(f"   ⚠️ 检测到事实冲突：{prog_detail}")
    else:
        print("   ✅ 未检测到互斥事实")

    # ---- 阶段3：每个模型当一次裁判，评估全部候选 ----
    print(f"\n▶ {len(MODELS)} 个模型作为裁判，各自评估全部候选（匿名打乱顺序）…")
    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        judge_results = list(pool.map(
            lambda spec: judge_all_candidates(spec, question, answered),
            MODELS,
        ))

    judges = []
    for spec, res in zip(MODELS, judge_results):
        judges.append({"name": spec["name"], **res})
        mark = "✅" if res["ok"] else "⚠️"
        print(f"\n   {mark} 裁判 {spec['name']}")
        if not res["ok"]:
            print(f"      失败：{res['error']}")
            continue
        self_ans = " ".join(res["self_answer"].split())
        print(f"      它自己先算出的答案：{self_ans[:150]}{'…' if len(self_ans) > 150 else ''}")
        if res["conflict"]:
            print(f"      ❗指出冲突：{res['conflict_detail']}")
        else:
            print("      未指出冲突")
        for model, s in res["scores"].items():
            print(f"      {model:<10} {s['score']}/10  {s['reason']}")

    # ---- 汇总每个候选的平均分 ----
    totals = {c["model"]: [] for c in answered}
    for j in judges:
        if not j["ok"]:
            continue
        for model, s in j["scores"].items():
            if model in totals:
                totals[model].append(s["score"])
    avg_scores = {m: round(sum(v) / len(v), 2) for m, v in totals.items() if v}

    # ---- 最终裁决 ----
    any_judge_conflict = any(j["ok"] and j["conflict"] for j in judges)
    best = max(avg_scores.values()) if avg_scores else 0

    if prog_conflict or any_judge_conflict:
        decision = "NEEDS_REVIEW"
        reason = "检测到答案之间存在事实冲突，需人工复核"
    elif not avg_scores:
        decision = "NEEDS_REVIEW"
        reason = "所有裁判均失败，无法评估"
    elif best < ACCEPT_SCORE:
        decision = "REJECTED"
        reason = f"最高裁判均分 {best} 低于接受线 {ACCEPT_SCORE}"
    else:
        decision = "ACCEPTED"
        reason = f"最高裁判均分 {best}，未检测到冲突"

    verdict = {
        "decision": decision, "reason": reason,
        "program_conflict": prog_conflict,
        "program_conflict_detail": prog_detail,
        "judges": judges, "avg_scores": avg_scores, "candidates": candidates,
    }
    write_log(question, verdict)
    print_summary(verdict)
    return verdict


def print_summary(verdict):
    print(f"\n{'─' * 76}")
    print("裁决结果")
    print(f"{'─' * 76}")
    for c in verdict["candidates"]:
        avg = verdict["avg_scores"].get(c["model"])
        avg_s = f"{avg}/10" if avg is not None else "-"
        answered = "已作答" if c["answered"] else "拒答/失败"
        print(f"  {c['model']:<10} 自评 {c['confidence']:>3}/100  {answered:<10} 裁判均分 {avg_s}")
    if verdict["program_conflict"]:
        print(f"\n  ⚠️ 程序层冲突：{verdict['program_conflict_detail']}")
    for j in verdict["judges"]:
        if j["ok"] and j["conflict"]:
            print(f"  ❗ {j['name']} 指出冲突：{j['conflict_detail']}")
    print(f"\n  【{verdict['decision']}】{verdict['reason']}")


# =============================================================================
# 日志
# =============================================================================

def write_log(question, verdict):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 76,
        f"时间：{ts}",
        f"用户问题：{question}",
        f"自评阈值：{CONFIDENCE_THRESHOLD} | 接受线：{ACCEPT_SCORE}",
        "-" * 76,
        "【各模型自评与作答】",
    ]
    for c in verdict["candidates"]:
        lines.append(f"  {c['model']}：自评 {c['confidence']}/100，"
                     f"{'已作答' if c['answered'] else '拒答/失败'}")
        lines.append(f"    答案：{c['answer']}")
    lines.append("")
    lines.append(f"【程序层冲突检测】{'发现冲突' if verdict['program_conflict'] else '未发现'}"
                 f"  {verdict['program_conflict_detail']}")
    lines.append("")
    lines.append("【裁判独立作答与评分】")
    for j in verdict["judges"]:
        if not j["ok"]:
            lines.append(f"  [{j['name']}] 评估失败：{j['error']}")
            continue
        lines.append(f"  [{j['name']}] 自己的答案：{j['self_answer']}")
        lines.append(f"      指出冲突：{j['conflict']}  {j['conflict_detail']}")
        for model, s in j["scores"].items():
            lines.append(f"      → {model}: {s['score']}/10 — {s['reason']}")
    lines.append("")
    lines.append(f"【裁判均分】{verdict['avg_scores']}")
    lines.append(f"【最终裁决】{verdict['decision']} — {verdict['reason']}")
    lines.append("")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# 主流程
# =============================================================================

def main():
    print("=" * 76)
    print("HAA 第 5 天（交叉验证改进版）· 全候选同屏 + 裁判先作答 + 程序层冲突检测")
    print("模型：" + " / ".join(m["name"] for m in MODELS))
    print(f"自评阈值：{CONFIDENCE_THRESHOLD} | 接受线：{ACCEPT_SCORE} | 日志：{LOG_FILE}")
    print("=" * 76)

    missing = [m["name"] for m in MODELS if not m["key"]]
    if missing:
        print(f"\n❌ 以下模型未配置 API key：{'、'.join(missing)}")
        print("请设置 DEEPSEEK_API_KEY / ZHIPU_API_KEY / QWEN_API_KEY，可参考 .env.example。")
        return 1

    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else [
        "847 乘以 9639 等于多少？请给出精确数值。",
    ]

    for q in questions:
        run_question(q)

    print(f"\n{'=' * 76}")
    print(f"全部完成，日志已写入 {LOG_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
