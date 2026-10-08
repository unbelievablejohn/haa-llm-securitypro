# -*- coding: utf-8 -*-
"""
truthfulqa_eval.py —— 在公开基准 TruthfulQA 上评测认识不确定性机制

为什么必须换公开基准
--------------------
本项目原有 54 条样本全部由作者自行设计。这是评审最容易攻击的一点：

    "你自己出题、自己判分，那当然全对。"

TruthfulQA（790 题）是学术界公认的基准，题目不是你出的，因此结论可与他人的
工作横向比较。它的设计恰好切中本项目的要害 —— **每道题都配有"人类常见的
错误答案"**（imitative falsehood），也就是说这些题**天生容易答错**。

核心实验设计：测出"拒答究竟救了我们多少次"
-------------------------------------------
校准分析已经证明，置信分本身不能保证内容正确。那么拒答机制的价值到底有多大？
本脚本用**强制作答**回答这个问题：

    第一阶段：正常判断。记录模型"想答还是想拒"。
    第二阶段：无论判断结果，一律生成答案（强制作答）。
    第三阶段：把生成的答案与 TruthfulQA 的标准答案比对，判定对错。

于是可以得到一个别的做法拿不到的交叉表：

    想答 且 答对   —— 机制起了作用
    想答 但 答错   —— **机制失效（confident-wrong），最危险**
    想拒 但 本可答对 —— **拒答的代价（过度拒答）**
    想拒 且 本会答错 —— **拒答的价值（成功避险）**

最后一项是所有指标里最有说服力的：它直接量化"如果去掉拒答机制，
会多出多少个错误答案"。

运行
----
    .venv\\Scripts\\python.exe truthfulqa_eval.py --limit 30     # 小样本验证
    .venv\\Scripts\\python.exe truthfulqa_eval.py               # 全量 790 题
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import day5
from tqa_match import classify, load_rows

CSV_NAME = "TruthfulQA.csv"
OUT_DIR = "tqa_results"


def find_csv():
    """定位 TruthfulQA.csv：先看项目目录，再看临时目录。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, CSV_NAME),
                 os.path.join(os.environ.get("TEMP", ""), "haa_msgs", CSV_NAME),
                 CSV_NAME):
        if os.path.isfile(cand):
            return cand
    return None


def evaluate_one(row, force=True):
    """对单条 TruthfulQA 题目跑完整流程。"""
    q = row["Question"]
    day5.reset_call_stats()

    judge = day5.judge_uncertainty(q)
    rec = {
        "question": q,
        "type": row["Type"],
        "category": row["Category"],
        "best_answer": row["Best Answer"],
        "best_incorrect": row["Best Incorrect Answer"],
        "confidence": judge["confidence"],
        "reason_type": judge.get("reason_type", "?"),
        "reason": judge["reason"],
        "judge_ok": judge["ok"],
        "decision": "answer" if judge["confident"] else "refuse",
        "answer": "",
        "forced": False,
        "label": None,
        "correct_score": None,
        "incorrect_score": None,
        "basis": "",
    }

    if not judge["ok"]:
        rec["decision"] = "error"
        rec["api_calls"] = day5.get_call_stats()["calls"]
        return rec

    # 强制模式：即使判为拒答也生成答案，才能知道"本来会答成什么"
    if judge["confident"] or force:
        rec["forced"] = not judge["confident"]
        try:
            rec["answer"] = day5.generate_answer(q)
        except Exception as exc:
            rec["answer"] = f"[生成失败] {exc}"
        if rec["answer"].lstrip().startswith(("[生成失败]", "生成回答时调用模型失败")):
            rec["decision"] = "error"

    # 判定答案对错（只要有答案文本就判，包括被强制作答的）
    if rec["answer"] and not rec["answer"].lstrip().startswith("[生成失败]"):
        c = classify(rec["answer"], row)
        rec["label"] = c["label"]
        rec["correct_score"] = round(c["correct_score"], 3)
        rec["incorrect_score"] = round(c["incorrect_score"], 3)
        rec["basis"] = c["basis"]

    rec["api_calls"] = day5.get_call_stats()["calls"]
    return rec


def compute(records):
    """计算核心交叉表与指标。"""
    valid = [r for r in records if r["decision"] in ("answer", "refuse")
             and r["label"] in ("truthful", "false")]
    unclear = [r for r in records if r["label"] == "unclear"]
    errors = [r for r in records if r["decision"] == "error"]

    # 交叉表：模型"想不想答" × 答案"对不对"
    cells = defaultdict(list)
    for r in valid:
        key = ("想答" if r["decision"] == "answer" else "想拒",
               "答对" if r["label"] == "truthful" else "答错")
        cells[key].append(r["id"] if "id" in r else r["question"][:40])

    n_want_answer = sum(1 for r in valid if r["decision"] == "answer")
    n_want_refuse = sum(1 for r in valid if r["decision"] == "refuse")

    n_ans_right = len(cells[("想答", "答对")])
    n_ans_wrong = len(cells[("想答", "答错")])
    n_ref_right = len(cells[("想拒", "答对")])
    n_ref_wrong = len(cells[("想拒", "答错")])

    m = {
        "total": len(records),
        "valid": len(valid),
        "unclear": len(unclear),
        "errors": len(errors),
        "want_answer": n_want_answer,
        "want_refuse": n_want_refuse,
        "answered_right": n_ans_right,
        "answered_wrong": n_ans_wrong,
        "refused_would_right": n_ref_right,
        "refused_would_wrong": n_ref_wrong,
    }
    m["precision_when_answered"] = (n_ans_right / n_want_answer
                                    if n_want_answer else None)
    # 拒答的价值：被拒的题里，本来会答错的比例
    m["refusal_value"] = (n_ref_wrong / n_want_refuse
                          if n_want_refuse else None)
    # 拒答的代价：被拒的题里，本来能答对的比例
    m["refusal_cost"] = (n_ref_right / n_want_refuse
                         if n_want_refuse else None)
    # 若不拒答，总错误数；拒答后剩下的错误数
    m["errors_without_abstention"] = n_ans_wrong + n_ref_wrong
    m["errors_with_abstention"] = n_ans_wrong
    m["errors_prevented"] = n_ref_wrong
    m["abstention_effectiveness"] = (
        n_ref_wrong / (n_ans_wrong + n_ref_wrong)
        if (n_ans_wrong + n_ref_wrong) else None)
    m["api_calls_total"] = sum(r.get("api_calls", 0) for r in records)
    m["by_type"] = {}
    for t in sorted(set(r["type"] for r in valid)):
        sub = [r for r in valid if r["type"] == t]
        right = sum(1 for r in sub if r["label"] == "truthful")
        m["by_type"][t] = {"n": len(sub), "truthful": right,
                           "truthful_rate": right / len(sub) if sub else None}
    return m


def build_report(m, records, model_name, csv_path):
    L = []
    L.append("# TruthfulQA 公开基准评测报告")
    L.append("")
    L.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("")
    L.append("| 项目 | 值 |")
    L.append("|---|---|")
    L.append(f"| 使用模型 | `{model_name}` |")
    L.append(f"| 数据集 | TruthfulQA（公开基准，非自建） |")
    L.append(f"| 数据来源 | {os.path.basename(csv_path)} |")
    L.append(f"| 样本量 | {m['total']} 题 |")
    L.append(f"| 可判定 | {m['valid']} 题 |")
    L.append(f"| 判定不明确 | {m['unclear']} 题（未计入指标） |")
    L.append(f"| 异常 | {m['errors']} 题 |")
    L.append(f"| 总 API 调用 | {m['api_calls_total']} 次 |")
    L.append("")
    L.append("## 为什么用这个基准")
    L.append("")
    L.append("TruthfulQA 的每道题都配有**人类常见的错误答案**（imitative falsehood），")
    L.append("也就是说这些题**天生容易答错**。用它来测认识不确定性机制，")
    L.append("可以回答一个自建数据集回答不了的问题：")
    L.append("")
    L.append("> **拒答机制究竟救了我们多少次？**")
    L.append("")
    L.append("## 核心交叉表（模型「想不想答」× 答案「对不对」）")
    L.append("")
    L.append("采用**强制作答**：无论模型是否想拒答，一律让它生成答案，")
    L.append("这样才能知道「如果当时答了，会答成什么」。")
    L.append("")
    L.append("|  | 答对 | 答错 | 小计 |")
    L.append("|---|---|---|---|")
    L.append(f"| **想作答** | {m['answered_right']} | **{m['answered_wrong']}** | "
             f"{m['want_answer']} |")
    L.append(f"| **想拒答** | {m['refused_would_right']} | "
             f"**{m['refused_would_wrong']}** | {m['want_refuse']} |")
    L.append("")
    L.append("四格的含义：")
    L.append("")
    L.append("| 格子 | 含义 |")
    L.append("|---|---|")
    L.append(f"| 想答 · 答对（{m['answered_right']}） | 机制起了作用 |")
    L.append(f"| 想答 · 答错（{m['answered_wrong']}） | **机制失效，最危险的情况** |")
    L.append(f"| 想拒 · 本可答对（{m['refused_would_right']}） | **拒答的代价**（效用损失） |")
    L.append(f"| 想拒 · 本会答错（{m['refused_would_wrong']}） | **拒答的价值**（成功避险） |")
    L.append("")
    L.append("## 核心指标")
    L.append("")
    L.append("| 指标 | 数值 | 含义 |")
    L.append("|---|---|---|")
    if m["precision_when_answered"] is not None:
        L.append(f"| 作答时的正确率 | **{m['precision_when_answered']:.4f}** | "
                 f"想作答的题里，实际答对的比例 |")
    if m["refusal_value"] is not None:
        L.append(f"| **拒答价值** | **{m['refusal_value']:.4f}** | "
                 f"被拒答的题里，模型本来会答错的比例（越高说明拒得越对） |")
    if m["refusal_cost"] is not None:
        L.append(f"| 拒答代价 | **{m['refusal_cost']:.4f}** | "
                 f"被拒答的题里，模型本来能答对的比例（效用损失） |")
    L.append(f"| 若不拒答的总错误数 | {m['errors_without_abstention']} | "
             f"想答答错 + 想拒但本会答错 |")
    L.append(f"| 拒答后的实际错误数 | {m['errors_with_abstention']} | "
             f"只有想答的题会产生答案 |")
    L.append(f"| **被拒答挡下的错误** | **{m['errors_prevented']}** | "
             f"拒答机制避免的错误答案数量 |")
    if m["abstention_effectiveness"] is not None:
        L.append(f"| **拒答贡献率** | **{m['abstention_effectiveness']:.4f}** | "
                 f"所有潜在错误中，被拒答挡下的比例 |")
    L.append("")
    L.append("## 按题目类型")
    L.append("")
    L.append("| 类型 | 题数 | 答对 | 正确率 |")
    L.append("|---|---|---|---|")
    for t, v in m["by_type"].items():
        L.append(f"| {t} | {v['n']} | {v['truthful']} | "
                 f"{v['truthful_rate']:.4f} |")
    L.append("")
    L.append("## 诚实说明")
    L.append("")
    L.append(f"1. **判定不明确的样本有 {m['unclear']} 题**，已排除在指标之外。")
    L.append("   自动判定自由文本答案必然有误差，这些样本需要人工复核才能定性。")
    L.append("   把它们排除而非硬猜，是为了不让指标虚高。")
    L.append("2. 强制作答会迫使模型回答它本来会拒答的问题，")
    L.append("   因此**不能**把本报告的数字当作系统真实部署时的表现 ——")
    L.append("   它衡量的是「拒答机制的价值」，而不是「系统的准确率」。")
    L.append("3. 单次运行，未做重复测量，指标存在运行间波动的可能。")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="TruthfulQA 公开基准评测")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全量）")
    ap.add_argument("--no-force", action="store_true",
                    help="关闭强制作答（则无法测出拒答的价值）")
    args = ap.parse_args()

    csv_path = find_csv()
    if not csv_path:
        print("找不到 TruthfulQA.csv")
        print("请先下载：")
        print('  curl.exe -sL -o TruthfulQA.csv "https://raw.githubusercontent.com/'
              'sylinrl/TruthfulQA/main/TruthfulQA.csv"')
        return 1

    rows = load_rows(csv_path)
    if args.limit:
        rows = rows[:args.limit]

    if not day5.API_KEY:
        print("未设置 HAA_API_KEY")
        return 1

    print("=" * 78)
    print("TruthfulQA 公开基准评测")
    print("=" * 78)
    print(f"模型      : {day5.MODEL_NAME}")
    print(f"数据集    : {csv_path}")
    print(f"样本量    : {len(rows)} 题")
    print(f"强制作答  : {'否' if args.no_force else '是'}"
          + ("（关闭后无法测出拒答价值）" if args.no_force else ""))
    print("=" * 78)
    print()

    records = []
    for i, row in enumerate(rows, 1):
        rec = evaluate_one(row, force=not args.no_force)
        rec["id"] = f"TQA{i:04d}"
        records.append(rec)
        mark = {"truthful": "对", "false": "错", "unclear": "?"}.get(rec["label"], "!")
        forced = "强制" if rec["forced"] else "    "
        print(f"[{i:>3}/{len(rows)}] {rec['id']} 置信{rec['confidence']:>3} "
              f"{rec['decision']:<6} {forced} 判定={mark} "
              f"{rec['category'][:18]:<18} {rec['question'][:40]}", flush=True)

    m = compute(records)
    report = build_report(m, records, day5.MODEL_NAME, csv_path)

    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(OUT_DIR, f"tqa_{ts}.json")
    json.dump({"model_name": day5.MODEL_NAME, "base_url": day5.BASE_URL,
               "metrics": m, "records": records},
              open(jpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    rpath = "truthfulqa_report.md"
    open(rpath, "w", encoding="utf-8").write(report)

    print()
    print("=" * 78)
    print("核心交叉表")
    print("=" * 78)
    print(f"                   答对      答错")
    print(f"  想作答           {m['answered_right']:>4}      {m['answered_wrong']:>4}")
    print(f"  想拒答           {m['refused_would_right']:>4}      "
          f"{m['refused_would_wrong']:>4}")
    print()
    if m["precision_when_answered"] is not None:
        print(f"  作答时正确率   : {m['precision_when_answered']:.4f}")
    if m["refusal_value"] is not None:
        print(f"  拒答价值       : {m['refusal_value']:.4f}"
              f"  ← 被拒的题里本会答错的比例")
    if m["refusal_cost"] is not None:
        print(f"  拒答代价       : {m['refusal_cost']:.4f}"
              f"  ← 被拒的题里本可答对的比例")
    print(f"  挡下的错误     : {m['errors_prevented']}")
    if m["abstention_effectiveness"] is not None:
        print(f"  拒答贡献率     : {m['abstention_effectiveness']:.4f}")
    print()
    print(f"  结果 JSON : {jpath}")
    print(f"  报告      : {rpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
