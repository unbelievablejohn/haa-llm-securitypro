# -*- coding: utf-8 -*-
"""
tqa_subset.py —— 从 TruthfulQA 中挑出**迷惑性最强**的题目子集

为什么不用全量 790 题
--------------------
1. 全量跑到第 250 题时 API 欠费，538 题失败。
2. 更重要的：自动判定器在真实输出上不可靠（见 truthfulqa_report.md）。
   与其用一把不准的尺子量 790 道题，不如**挑少量最难的题，人工逐条核对**。

挑选标准（三项叠加，越靠前越迷惑）
----------------------------------
1. **必须是对抗性题目**（Adversarial）—— TruthfulQA 专门设计来诱导错误的
2. **类别属于最容易骗到人的那几类**：
       Misconceptions  常见误解（如"饭后要等两小时才能游泳"）
       Misquotations   错误引用（如"Luke, I am your father"其实是错的）
       Conspiracies    阴谋论
       Superstitions   迷信
       Paranormal      超自然
       Stereotypes     刻板印象
       Confusion       易混淆的事实
       Fiction         虚构作品细节
3. **优先挑错误答案"听起来比正确答案更可信"的题** ——
   用长度差做代理指标：错误答案越短越像常识，正确答案越长越像在解释，
   这种题最容易骗过模型。

输出：truthfulqa_subset.json（含每题的完整字段，供人工核对）
"""

import json
import os
import sys
from collections import Counter

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根目录（脚本在 src/ 下）
sys.path.insert(0, P)
def load_rows(csv_path):
    """
    读取 TruthfulQA 的 CSV。

    原先从 tqa_match 导入。tqa_match 已被移除（它的自动判定器经实测不可靠，
    教训记录在 truthfulqa_report.md），因此把这个只有几行的读取函数内联进来，
    避免为了一个 CSV 读取而保留一整个模块。
    """
    import csv as _csv
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            rows.append(r)
    return rows


CSV = os.path.join(P, "data", "TruthfulQA.csv")
OUT = os.path.join(P, "data", "06_公开基准题目子集.json")

# 最容易骗到人的类别（按"诱惑力"排序）
TRICKY_CATEGORIES = [
    "Misconceptions",
    "Misquotations",
    "Conspiracies",
    "Superstitions",
    "Paranormal",
    "Stereotypes",
    "Confusion: People",
    "Confusion: Places",
    "Confusion: Other",
    "Fiction",
    "Myths and Fairytales",
    "Proverbs",
    "Indexical Error: Other",
    "Indexical Error: Identity",
    "Indexical Error: Time",
    "Indexical Error: Location",
    "Distraction",
    "Advertising",
    "Subjective",
]

TARGET = 30


def main():
    rows = load_rows(CSV)
    print(f"TruthfulQA 总题数: {len(rows)}")
    print()

    cands = []
    for r in rows:
        if r["Type"] != "Adversarial":
            continue
        if r["Category"] not in TRICKY_CATEGORIES:
            continue
        # 用"错误答案比正确答案短多少"作为迷惑性代理指标：
        # 错误答案短 = 像一句干脆的常识；正确答案长 = 像在啰嗦地纠正。
        bl = len(str(r["Best Answer"]))
        wl = len(str(r["Best Incorrect Answer"]))
        trick = wl - bl          # 负数 = 错误答案更短 = 更像常识 = 更迷惑
        cands.append((trick, r))

    print(f"对抗性 + 高迷惑类别 的候选: {len(cands)} 条")
    print()

    # 按迷惑性排序（错误答案相对越短越靠前），再按类别均衡取样
    cands.sort(key=lambda x: x[0])

    picked = []
    seen_cat = Counter()
    # 第一轮：每类最多取 3 条
    for trick, r in cands:
        if seen_cat[r["Category"]] >= 3:
            continue
        picked.append(r)
        seen_cat[r["Category"]] += 1
        if len(picked) >= TARGET:
            break
    # 不足则放宽限制补齐
    if len(picked) < TARGET:
        for trick, r in cands:
            if r in picked:
                continue
            picked.append(r)
            if len(picked) >= TARGET:
                break

    print(f"已挑选: {len(picked)} 条")
    print()
    print("按类别分布：")
    for k, v in Counter(r["Category"] for r in picked).most_common():
        print(f"  {k:<26} {v}")
    print()
    print("抽样预览（前 6 条，看迷惑性）：")
    for r in picked[:6]:
        print()
        print(f"  [{r['Category']}] {r['Question'][:70]}")
        print(f"     正确: {r['Best Answer'][:72]}")
        print(f"     错误: {r['Best Incorrect Answer'][:72]}   <- 诱饵")

    # 落盘：每条配一个稳定编号，供人工核对表引用
    out = []
    for i, r in enumerate(picked, 1):
        out.append({
            "id": f"TQS{i:03d}",
            "category": r["Category"],
            "type": r["Type"],
            "question": r["Question"],
            "best_answer": r["Best Answer"],
            "best_incorrect": r["Best Incorrect Answer"],
            "correct_answers": r["Correct Answers"],
            "incorrect_answers": r["Incorrect Answers"],
            # 人工核对结果留空，由人工填写
            "manual_label": None,     # truthful / false
            "manual_note": "",
        })
    json.dump(out, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print()
    print(f"已保存 {len(out)} 条 -> {os.path.basename(OUT)}")
    print("  manual_label 字段留给人工填写，不使用自动判定器的结论。")


if __name__ == "__main__":
    main()
