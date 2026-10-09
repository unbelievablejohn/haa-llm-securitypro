# -*- coding: utf-8 -*-
"""
tqa_manual.py —— 在精选子集上跑，并输出**人工核对表**

为什么人工核对
--------------
自动判定器在真实模型输出上不可靠（见 truthfulqa_report.md 的完整复盘）。
与其用一把不准的尺子量很多题，不如**少跑一些、逐条看**。

因此本脚本做两件事：
    1. 跑系统（不确定性判断 + 强制作答），把结果存进 JSON
    2. 输出一份**核对表**：每题一行，把模型的回答与标准答案/诱饵并列，
       供人工填写 manual_label

实验设计（与全量版一致，只是规模小）
------------------------------------
强制作答，从而知道"如果当时答了，会答成什么"。于是能得到交叉表：
    想作答·答对 / 想作答·答错 / 想拒答·本会答对 / 想拒答·本会答错
最后一项就是**拒答的价值**。

用法
----
    .venv\\Scripts\\python.exe tqa_manual.py             # 跑全部 30 条
    .venv\\Scripts\\python.exe tqa_manual.py --limit 8   # 先跑前 8 条
    .venv\\Scripts\\python.exe tqa_manual.py --sheet     # 只打印核对表，不调 API
"""

import argparse
import json
import os
import sys
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根目录（脚本在 src/ 下）
sys.path.insert(0, P)

import day5

SUBSET = os.path.join(P, "data", "06_公开基准题目子集.json")
OUT = os.path.join(P, "data", "04_公开基准", "tqa_manual.json")


def build_sheet(records):
    """生成人工核对表。"""
    L = []
    L.append("=" * 100)
    L.append("TruthfulQA 精选子集 · 人工核对表")
    L.append("=" * 100)
    L.append("")
    L.append("读法：每题下面列出【标准答案】与【诱饵】，然后列模型的实际回答。")
    L.append("请判断模型回答属于哪一类，填进 manual_label 字段（truthful / false）。")
    L.append("")
    for r in records:
        L.append("=" * 100)
        L.append(f"[{r['id']}] {r['category']}   置信={r['confidence']}  "
                 f"决策={r['decision']}  原因类别={r.get('reason_type')}")
        L.append("")
        L.append(f"  问    ：{r['question']}")
        L.append(f"  标准答：{r['best_answer']}")
        L.append(f"  诱饵  ：{r['best_incorrect']}")
        L.append("  模型答：")
        ans = r.get("answer") or "(未生成)"
        for line in " ".join(str(ans).split()).split(". "):
            pass
        wrapped = " ".join(str(ans).split())
        for i in range(0, len(wrapped), 92):
            L.append(f"          {wrapped[i:i+92]}")
        L.append("")
        L.append(f"  人工判定：{r.get('manual_label') or '（待填）'}"
                 f"   {r.get('manual_note') or ''}")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="精选子集人工核对")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sheet", action="store_true",
                    help="只打印核对表，不调用 API（读已有结果）")
    args = ap.parse_args()

    subset = json.load(open(SUBSET, encoding="utf-8"))
    if args.limit:
        subset = subset[:args.limit]

    if args.sheet:
        if not os.path.isfile(OUT):
            print("还没有结果文件，请先跑一次。")
            return 1
        d = json.load(open(OUT, encoding="utf-8"))
        print(build_sheet(d["records"]))
        return 0

    if not day5.API_KEY:
        print("未设置 HAA_API_KEY")
        return 1

    print("=" * 88)
    print("TruthfulQA 精选子集 · 强制作答")
    print("=" * 88)
    print(f"模型   : {day5.MODEL_NAME}")
    print(f"题数   : {len(subset)}")
    print("=" * 88)
    print()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    records = []
    for i, item in enumerate(subset, 1):
        q = item["question"]
        judge = day5.judge_uncertainty(q)
        rec = {
            "id": item["id"],
            "category": item["category"],
            "question": q,
            "best_answer": item["best_answer"],
            "best_incorrect": item["best_incorrect"],
            "correct_answers": item["correct_answers"],
            "incorrect_answers": item["incorrect_answers"],
            "confidence": judge["confidence"],
            "reason_type": judge.get("reason_type", "?"),
            "reason": judge["reason"],
            "decision": "answer" if judge["confident"] else "refuse",
            "forced": False,
            "answer": "",
            "manual_label": None,
            "manual_note": "",
        }
        if judge["ok"] and judge["confident"]:
            try:
                rec["answer"] = day5.generate_answer(q)
            except Exception as exc:
                rec["answer"] = f"[生成失败] {exc}"
        elif judge["ok"]:
            # 强制作答：知道"如果答了会说什么"，才能算出拒答的价值
            rec["forced"] = True
            try:
                rec["answer"] = day5.generate_answer(q)
            except Exception as exc:
                rec["answer"] = f"[生成失败] {exc}"

        records.append(rec)
        tag = "强制" if rec["forced"] else "    "
        print(f"[{i:>2}/{len(subset)}] {rec['id']} 置信{rec['confidence']:>3} "
              f"{rec['decision']:<6} {tag}  {rec['category'][:22]:<22} "
              f"{q[:34]}", flush=True)

        # 增量落盘
        json.dump({"model_name": day5.MODEL_NAME,
                   "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "records": records},
                  open(OUT, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    print()
    print(f"结果已保存: {os.path.relpath(OUT, P)}")
    print()
    print("=" * 88)
    print("汇总（未经人工核对，仅供参考）")
    print("=" * 88)
    n_ans = sum(1 for r in records if r["decision"] == "answer")
    n_ref = sum(1 for r in records if r["decision"] == "refuse")
    print(f"  模型想作答 : {n_ans} 条")
    print(f"  模型想拒答 : {n_ref} 条")
    print(f"  其中被强制作答 : {sum(1 for r in records if r['forced'])} 条")
    print()
    print("  下一步：逐条人工核对，填写 manual_label 字段。")
    print("  核对表： .venv\\Scripts\\python.exe tqa_manual.py --sheet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
