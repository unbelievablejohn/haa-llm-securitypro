# -*- coding: utf-8 -*-
"""
consistency_guard.py —— 扩展的程序层校验（实体比对 + 自洽性检查）

定位
----
这是**不依赖模型自觉**的校验层：纯程序、零 API 费用、确定性。
它与 `injection_guard.py`（输入层）互补，工作在输出层：
    输入层：这道题是不是在试图操纵我？
    输出层：这份输出自己有没有前后矛盾？多份输出之间有没有冲突？

为什么需要它 —— 两个实证依据
-----------------------------
1. **自洽性失效**：曾出现置信分 **96** 的判断，其 reason 里写"该定理由
   Ringel 和 Youngs 于 **1967** 年证明"，而最终 answer 里写"**1968** 年证明"。
   同一个响应内部，理由与答案互相矛盾，**而当时没有任何机制发现它**。

2. **跨答案冲突**：用 `847 × 9639` 测试时，三个模型给出三个不同数字，
   其中两个是错的，而**裁判没发现**。若在程序层直接比对数字，这个矛盾
   一秒就能暴露。

本模块提供两类检查
------------------
    check_self_consistency(reason, answer)   —— 单一响应内部：理由 vs 答案
    check_cross_entities(items, question)    —— 多份候选之间：实体级比对

设计原则
--------
· **宁可多报，不可漏报**：漏报会让错误答案被放行；多报的代价只是多一次人工确认。
· **区分严重度**：`conflict` 是可确定判定的矛盾（如两个年份互斥）；
  `suspicious` 是值得看一眼的迹象（如理由里的数字未在答案中出现）。
· **说明依据**：每一处判定都给出具体命中的文本，便于人工复核。
"""

import re

# ---------------------------------------------------------------------------
# 事实片段的抽取规则
# ---------------------------------------------------------------------------

YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
NUM_RE = re.compile(r"\d[\d,，]*(?:\.\d+)?")
# 中文书名号或引号包裹的标题
TITLE_RE = re.compile(r"《([^》]{2,80})》|「([^」]{2,80})」")
# 机构名：以常见后缀结尾的连续中文串
ORG_RE = re.compile(
    r"[\u4e00-\u9fff]{2,12}(?:大学|学院|研究院|研究所|实验室|公司|集团|委员会|协会|学会|基金会)")
# 拉丁文专名：两个及以上首字母大写的词（人名、机构名、论文名）
LATIN_RE = re.compile(r"\b([A-Z][a-zA-Z\.\-]{1,}(?:\s+[A-Z][a-zA-Z\.\-]{1,}){1,5})\b")

# 参与数字比对时忽略的短数字与常见噪声
NUM_STOP = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}


def extract_facts(text, exclude_numbers=None):
    """
    从一段文本中抽取可核对的事实片段。

    exclude_numbers：题干中出现过的数字。它们在被引述时属于正常复现，
    不应算作冲突证据（例如把题干的 847 与 9639 抄一遍）。

    返回 dict，每类是一个集合：
        years   四位年份
        numbers 3 位以上的数值（已去千分位）
        titles  书名号/引号内的标题
        orgs    机构名
        latin   拉丁文专名
    """
    t = str(text or "")
    exclude = exclude_numbers or set()

    years = set(YEAR_RE.findall(t))

    numbers = set()
    for m in NUM_RE.findall(t):
        v = m.replace(",", "").replace("，", "")
        try:
            fv = float(v)
        except ValueError:
            continue
        if fv < 100:                 # 两位数以下噪声太大
            continue
        if v in NUM_STOP or v in exclude:
            continue
        numbers.add(v)

    titles = set()
    for m in TITLE_RE.finditer(t):
        title = (m.group(1) or m.group(2) or "").strip()
        if title:
            titles.add(title)

    orgs = set(ORG_RE.findall(t)) if hasattr(ORG_RE, "findall") else set()
    orgs = set(ORG_RE.findall(t)) if False else set(
        m.group(0) for m in ORG_RE.finditer(t))

    latin = set()
    for m in LATIN_RE.finditer(t):
        name = m.group(1).strip()
        # 过滤掉常见的英文短语噪声
        if len(name) >= 5 and name not in ("The", "This", "That"):
            latin.add(name)

    return {
        "years": years,
        "numbers": numbers,
        "titles": titles,
        "orgs": orgs,
        "latin": latin,
    }


def numbers_in(text):
    """抽取文本里的所有数字（含题干），用作 exclude 集合。"""
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


# ---------------------------------------------------------------------------
# 检查一：单一响应内部的自洽性（理由 vs 答案）
# ---------------------------------------------------------------------------

def check_self_consistency(reason, answer, question=""):
    """
    检查同一响应内，判断理由与最终答案是否互相矛盾。

    最典型的情形（实测发生过）：理由里写"1967 年证明"，答案里写"1968 年证明"。

    判定规则：
      · 年份：两边都有年份，且**完全没有交集** → conflict（可确定判定）
              部分重叠 → suspicious
      · 标题：两边都提到不同标题 → suspicious（可能是引用了不同文献）
      · 机构：同上

    返回 dict：
        severity  "none" / "suspicious" / "conflict"
        details   list[str]  每处问题的具体说明
    """
    if not reason or not answer:
        return {"severity": "none", "details": []}

    rf = extract_facts(reason, exclude_numbers=numbers_in(question))
    af = extract_facts(answer, exclude_numbers=numbers_in(question))
    details = []
    severity = "none"

    # ---- 年份 ----
    ry, ay = rf["years"], af["years"]
    if ry and ay:
        if not (ry & ay):
            severity = "conflict"
            details.append(
                f"年份互相矛盾：理由中说 {'、'.join(sorted(ry))}，"
                f"答案中说 {'、'.join(sorted(ay))}")
        elif ry - ay:
            severity = max(severity, "suspicious", key=["none", "suspicious", "conflict"].index)
            details.append(
                f"理由提到未在答案中出现的年份：{'、'.join(sorted(ry - ay))}")

    # ---- 标题 ----
    rt, at = rf["titles"], af["titles"]
    if rt and at:
        common = rt & at
        if not common:
            severity = max(severity, "suspicious", key=["none", "suspicious", "conflict"].index)
            details.append(
                f"理由与答案提到不同标题：理由《{'》《'.join(sorted(rt))}》，"
                f"答案《{'》《'.join(sorted(at))}》")

    # ---- 机构 ----
    ro, ao = rf["orgs"], af["orgs"]
    if ro and ao and not (ro & ao):
        severity = max(severity, "suspicious", key=["none", "suspicious", "conflict"].index)
        details.append(
            f"理由与答案提到不同机构：理由 {sorted(ro)}，答案 {sorted(ao)}")

    return {"severity": severity, "details": details}


# ---------------------------------------------------------------------------
# 检查二：多份候选之间的实体比对
# ---------------------------------------------------------------------------

def check_cross_entities(answers, question=""):
    """
    对多份候选答案做**实体级**比对，找出互斥的事实。

    与 day5_verify.detect_conflict 的关系：
        那个函数只比对"结论性数值"；本函数把比对面扩展到
        年份、标题、机构、拉丁文专名，覆盖"数字相同但人名不同"这类
        前者抓不到的情况（例如三份答案都写 2021 年，但作者分别是
        Zhang / Li / Wang）。

    判定规则（与数值判据同源，避免误报）：
        · 若某个实体出现在**全部**答案中 → 它是公认值，不构成冲突
        · 若各答案的实体集合**毫无交集**且都非空 → 结论完全不同 → conflict
        · 若只有个别答案提出别人没有的实体 → suspicious（可能是补充细节）

    返回 dict：
        severity  "none" / "suspicious" / "conflict"
        details   list[str]
        facts     list[dict]  各答案抽取到的事实，便于人工查看
    """
    if not answers or len(answers) < 2:
        return {"severity": "none", "details": [], "facts": []}

    exclude = numbers_in(question) if question else set()
    facts = [extract_facts(a, exclude_numbers=exclude) for a in answers]
    details = []
    severity = "none"

    # 比对字段：(字段名, 中文标签, 是否按"数值"处理)
    #
    # 数值单独处理，原因：答案常包含背景数字（如"标准大气压 101.325 kPa"），
    # 若按年份那套"无交集即冲突"的规则处理会大量误报。数值采用更保守的判据：
    #   各答案的数值集合**毫无交集** → 结论完全不同 → conflict
    #   有交集但个别答案提出他人没有的值 → suspicious（可能是补充细节）
    FIELDS = (
        ("numbers", "数值", True),
        ("years", "年份", False),
        ("titles", "标题", False),
        ("orgs", "机构", False),
        ("latin", "专名", False),
    )

    for field, label, is_number in FIELDS:
        sets = [f[field] for f in facts]
        non_empty = [s for s in sets if s]
        if len(non_empty) < 2:
            continue

        shared = set.intersection(*non_empty)
        if shared:
            lone = set()
            for s in non_empty:
                lone |= (s - shared)
            if lone:
                severity = max(severity, "suspicious",
                               key=["none", "suspicious", "conflict"].index)
                details.append(f"{label}存在分歧：公认 {sorted(shared)}，"
                               f"另有 {sorted(lone)} 只出现在部分答案中")
        else:
            # 毫无交集：各方结论完全不同 → 这是可确定判定的矛盾，
            # 例如 847×9639 的三个答案 8164233 / 8193133 / 8160533。
            severity = "conflict"
            shown = " | ".join(
                f"答案{i+1}={sorted(s)[:3]}" for i, s in enumerate(sets) if s)
            details.append(f"{label}完全不一致：{shown}")

    return {"severity": severity, "details": details, "facts": facts}


def summarize(result):
    """把检查结果压成一行，便于打进日志与报告。"""
    sev = result.get("severity", "none")
    if sev == "none":
        return "自洽性检查通过"
    tag = "矛盾" if sev == "conflict" else "存疑"
    return f"[{tag}] " + "；".join(result.get("details", []))


if __name__ == "__main__":
    # 自测：用实测中真正出现过的矛盾，以及正常样本做对照
    print("=" * 78)
    print("检查一：单一响应内部自洽性（reason vs answer）")
    print("=" * 78)
    cases = [
        ("实测发生过的矛盾（P7 案例）",
         "Ringel–Youngs 定理是图论中关于完全图边分解的经典定理，于 1967 年由 "
         "Gerhard Ringel 和 J. W. T. Youngs 证明，属于稳定的数学史知识。",
         "Ringel–Youngs 定理（关于完全图 K_n 的亏格公式）于 1968 年由 "
         "Gerhard Ringel 和 J. W. T. Youngs 证明。",
         "conflict"),
        ("正常：理由与答案年份一致",
         "该定理于 1968 年证明，属于稳定数学史知识。",
         "该定理于 1968 年由 Ringel 和 Youngs 证明。",
         "none"),
        ("正常：理由是常识描述、不含年份",
         "这是基础物理常识。",
         "水在标准大气压下的沸点是 100 摄氏度。",
         "none"),
        ("存疑：理由与答案提到不同标题",
         "我无法确认《Sodium Interphase Mechanisms》这篇论文是否存在。",
         "该结论出自《Ion Transport in Layered Oxides》一文。",
         "suspicious"),
        ("正常：理由与答案提到同一标题",
         "我无法确认《Global Carbon Policy Review》这本期刊是否存在。",
         "关于《Global Carbon Policy Review》，我没有可靠信息。",
         "none"),
    ]
    for label, reason, answer, expect in cases:
        r = check_self_consistency(reason, answer)
        ok = "OK" if r["severity"] == expect else "不符预期"
        print(f"\n  [{ok}] {label}")
        print(f"        期望={expect}  实际={r['severity']}")
        for d in r["details"]:
            print(f"        · {d}")

    print()
    print("=" * 78)
    print("检查二：多份候选之间的实体比对")
    print("=" * 78)
    cross_cases = [
        ("三份数字互不相同（847×9639 的实测情形）",
         ["847 × 9639 = 8,164,233", "847 乘以 9639 得 8,193,133",
          "847 × 9639 = 8,160,533"],
         "847 乘以 9639 等于多少？", "conflict"),
        ("三份年份相同（正常一致）",
         ["该论文发表于 2023 年。", "2023 年发表。", "发表于 2023 年。"],
         "这篇论文哪年发表？", "none"),
        ("年份相同但作者名不同（数值判据抓不到）",
         ["该论文由 Zhang Wei 于 2023 年发表。",
          "该论文由 Li Ming 于 2023 年发表。",
          "该论文由 Wang Fang 于 2023 年发表。"],
         "这篇论文哪年发表，作者是谁？", "conflict"),
    ]
    for label, answers, q, expect in cross_cases:
        r = check_cross_entities(answers, q)
        ok = "OK" if r["severity"] == expect else "不符预期"
        print(f"\n  [{ok}] {label}")
        print(f"        期望={expect}  实际={r['severity']}")
        for d in r["details"]:
            print(f"        · {d}")
