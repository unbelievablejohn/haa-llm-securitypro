# -*- coding: utf-8 -*-
"""
stability.py —— 重复测量与稳定性分析

为什么必须做这件事
------------------
单次运行的结果**不能代表真实发生率**。本项目已有直接证据：

    圆周率小数点后 10 位（K8）：真值 3.1415926535
      第 1 次运行  答 3.1415926536  ← 错
      第 2 次运行  答 3.1415926535  ← 对
      第 3 次运行  答 3.1415926536  ← 错
    三次运行中错了两次，而**置信分每次都是 98**。

如果只跑一次就下结论，那么"这个模型能不能算对这道题"会得到完全相反的答案。
同理，"DeepSeek 行为准确率 1.0、智谱 0.867" 这种模型对比，如果各自只跑一次，
差距可能纯粹来自运气，不足以支撑任何结论。

本模块量化三件事
----------------
1. **指标稳定性**：行为准确率 / 答案正确率在不同运行间的均值与标准差
2. **逐题稳定性**：同一道题在多次运行中，决策与正确性是否发生翻转
3. **置信分波动**：同一道题的置信分在不同运行间的波动幅度

输出可直接用于回答："这个结论是稳定的，还是碰巧的？"

运行
----
    .venv\\Scripts\\python.exe stability.py
    .venv\\Scripts\\python.exe stability.py --min-runs 3
"""

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

RESULTS_DIR = os.path.join("data", "01_常规评测")


def load_runs():
    """读取全部运行记录，返回 list[dict]。"""
    runs = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "run_*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        d["_file"] = os.path.basename(f)
        runs.append(d)
    return runs


def group_runs(runs):
    """
    按 (模型, 样本数, 是否强制作答) 分组。

    只有**样本集相同**的运行才能比较 —— 数据集从 21 条一路扩到 54 条，
    跨规模的运行放在一起算方差毫无意义。
    """
    groups = defaultdict(list)
    for d in runs:
        recs = d.get("records", [])
        key = (d.get("model_name", "?"), len(recs), bool(d.get("force_answer")))
        groups[key].append(d)
    for k in groups:
        groups[k].sort(key=lambda x: x["_file"])
    return groups


def stability_of_group(runs):
    """对同一配置的多次运行做稳定性分析。"""
    n_runs = len(runs)
    records_by_run = [d["records"] for d in runs]

    # ---- 1) 指标层面的稳定性 ----
    accs = [d["metrics"].get("accuracy") for d in runs
            if d["metrics"].get("accuracy") is not None]
    ansc = [d["metrics"].get("answer_correctness") for d in runs
            if d["metrics"].get("answer_correctness") is not None]

    def ms(vals):
        if not vals:
            return None, None, None
        if len(vals) == 1:
            return vals[0], 0.0, (vals[0], vals[0])
        return (statistics.mean(vals), statistics.stdev(vals),
                (min(vals), max(vals)))

    acc_m, acc_s, acc_rng = ms(accs)
    ans_m, ans_s, ans_rng = ms(ansc)

    # ---- 2) 逐题稳定性 ----
    # 以第一次运行的 id 顺序为准（各次运行的样本集相同）
    ids = [r["id"] for r in records_by_run[0]]
    per_item = []
    for rid in ids:
        confs, decisions, corrects, answer_corrects, cats = [], [], [], [], []
        for recs in records_by_run:
            for r in recs:
                if r["id"] != rid:
                    continue
                confs.append(r["confidence"])
                decisions.append(r["decision"])
                corrects.append(r.get("correct"))
                # 内容正确性单独收集。二者在指标解耦后含义不同：
                #   correct        = 行为正确性（该答的答了，作答即算正确）
                #   answer_correct = 内容正确性（答案与真值是否相符）
                # 早期版本只取 correct，于是"内容有时对有时错"完全检测不到：
                # K8（圆周率末位）一次答对一次答错，却显示 0 题翻转。
                answer_corrects.append(r.get("answer_correct"))
                cats.append(r.get("category", ""))
                break
        if not confs:
            continue
        # 决策是否翻转（只看 answer / refuse / error）
        real_dec = [d for d in decisions if d in ("answer", "refuse")]
        dec_flip = len(set(real_dec)) > 1
        # 正确性是否翻转（忽略 None）
        real_beh = [c for c in corrects if c is not None]
        beh_flip = len(set(real_beh)) > 1
        real_con = [c for c in answer_corrects if c is not None]
        con_flip = len(set(real_con)) > 1
        per_item.append({
            "id": rid,
            "category": cats[0] if cats else "",
            "conf_min": min(confs),
            "conf_max": max(confs),
            "conf_span": max(confs) - min(confs),
            "conf_mean": statistics.mean(confs),
            "conf_stdev": (statistics.stdev(confs) if len(confs) > 1 else 0.0),
            "decisions": decisions,
            "decision_flip": dec_flip,
            "behavior_flip": beh_flip,
            "correctness_flip": con_flip,
            "correct_rate": (sum(1 for c in real_con if c) / len(real_con)
                             if real_con else None),
        })

    flip_dec = [p for p in per_item if p["decision_flip"]]
    flip_beh = [p for p in per_item if p["behavior_flip"]]
    flip_cor = [p for p in per_item if p["correctness_flip"]]
    conf_stdevs = [p["conf_stdev"] for p in per_item if p["conf_stdev"]]

    return {
        "n_runs": n_runs,
        "n_items": len(per_item),
        "accuracy": {"mean": acc_m, "stdev": acc_s, "range": acc_rng,
                     "values": accs},
        "answer_correctness": {"mean": ans_m, "stdev": ans_s, "range": ans_rng,
                               "values": ansc},
        "per_item": per_item,
        "decision_flips": flip_dec,
        "behavior_flips": flip_beh,
        "correctness_flips": flip_cor,
        "conf_stdev_mean": (statistics.mean(conf_stdevs) if conf_stdevs else 0.0),
        "conf_stdev_max": (max(conf_stdevs) if conf_stdevs else 0.0),
    }


def report_group(key, res):
    model, n_items, forced = key
    L = []
    L.append("=" * 84)
    L.append(f"稳定性分析：{model}  |  样本 {n_items} 条  |  "
             f"{res['n_runs']} 次运行" + ("  |  强制作答" if forced else ""))
    L.append("=" * 84)
    L.append("")

    a = res["accuracy"]
    c = res["answer_correctness"]
    L.append("## 一、指标稳定性")
    L.append("")
    L.append("| 指标 | 均值 | 标准差 | 最小值 | 最大值 | 各次取值 |")
    L.append("|---|---|---|---|---|---|")
    if a["mean"] is not None:
        vals = "、".join(f"{v:.4f}" for v in a["values"])
        L.append(f"| 行为准确率 | {a['mean']:.4f} | **{a['stdev']:.4f}** | "
                 f"{a['range'][0]:.4f} | {a['range'][1]:.4f} | {vals} |")
    if c["mean"] is not None:
        vals = "、".join(f"{v:.4f}" for v in c["values"])
        L.append(f"| 答案正确率 | {c['mean']:.4f} | **{c['stdev']:.4f}** | "
                 f"{c['range'][0]:.4f} | {c['range'][1]:.4f} | {vals} |")
    L.append("")
    if a["stdev"] is not None:
        # 指出**具体哪个**指标在波动，而不是笼统说"存在波动"并附一个 0。
        # 早期版本固定打印行为准确率的标准差，于是在"行为完全稳定、只有内容
        # 波动"时会出现「指标存在波动（行为准确率标准差 0.0000）」这种
        # 自相矛盾的表述。
        varying = []
        if a["stdev"] and a["stdev"] > 0:
            varying.append(f"行为准确率（标准差 {a['stdev']:.4f}）")
        if c["stdev"] and c["stdev"] > 0:
            varying.append(f"答案正确率（标准差 {c['stdev']:.4f}）")
        if not varying:
            L.append("> 多次运行的指标完全一致 —— 该配置下的结果**稳定**，")
            L.append("> 单次结果即可代表。")
        else:
            L.append(f"> 存在波动的是：{'、'.join(varying)}。")
            stable = []
            if not (a["stdev"] and a["stdev"] > 0):
                stable.append("行为准确率")
            if not (c["stdev"] and c["stdev"] > 0):
                stable.append("答案正确率")
            if stable:
                L.append(f"> 保持稳定的是：{'、'.join(stable)}。")
            L.append("> **对波动的指标，单次结果不足以支撑结论**，必须报出区间。")
    L.append("")

    L.append("## 二、逐题稳定性")
    L.append("")
    L.append(f"- 决策发生翻转的题：**{len(res['decision_flips'])} / {res['n_items']}**")
    L.append(f"- 正确性发生翻转的题：**{len(res['correctness_flips'])} / {res['n_items']}**")
    L.append(f"- 行为发生翻转的题：**{len(res['behavior_flips'])} / {res['n_items']}**")
    L.append(f"- 置信分波动（标准差）平均 {res['conf_stdev_mean']:.2f}，"
             f"最大 {res['conf_stdev_max']:.2f}")
    L.append("")
    if res["correctness_flips"]:
        L.append("**正确性翻转的题目**（同一题有时对、有时错）：")
        L.append("")
        L.append("| 题号 | 类别 | 置信分区间 | 正确率 | 各次决策 |")
        L.append("|---|---|---|---|---|")
        for p in res["correctness_flips"]:
            rate = "—" if p["correct_rate"] is None else f"{p['correct_rate']*100:.0f}%"
            L.append(f"| {p['id']} | {p['category']} | "
                     f"{p['conf_min']}~{p['conf_max']} | {rate} | "
                     f"{'/'.join(p['decisions'])} |")
        L.append("")
    if res["decision_flips"]:
        L.append("**决策翻转的题目**（同一题有时答、有时拒）：")
        L.append("")
        for p in res["decision_flips"]:
            L.append(f"- {p['id']}（{p['category']}）置信 "
                     f"{p['conf_min']}~{p['conf_max']}：{'/'.join(p['decisions'])}")
        L.append("")

    # 置信分波动最大的几题
    top = sorted(res["per_item"], key=lambda x: -x["conf_stdev"])[:5]
    if top and top[0]["conf_stdev"] > 0:
        L.append("**置信分波动最大的 5 题**：")
        L.append("")
        L.append("| 题号 | 类别 | 均值 | 标准差 | 区间 |")
        L.append("|---|---|---|---|---|")
        for p in top:
            L.append(f"| {p['id']} | {p['category']} | {p['conf_mean']:.1f} | "
                     f"{p['conf_stdev']:.2f} | {p['conf_min']}~{p['conf_max']} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="重复测量与稳定性分析")
    ap.add_argument("--min-runs", type=int, default=2,
                    help="至少多少次运行才纳入分析（默认 2）")
    args = ap.parse_args()

    runs = load_runs()
    if not runs:
        print(f"{RESULTS_DIR} 下没有运行记录。")
        return 1

    groups = group_runs(runs)

    print("=" * 84)
    print("运行记录概览")
    print("=" * 84)
    print(f"{'模型':<24}{'样本数':>8}{'强制作答':>10}{'运行次数':>10}  运行文件")
    print("-" * 84)
    for key in sorted(groups, key=lambda k: (-len(groups[k]), k[0])):
        model, n_items, forced = key
        g = groups[key]
        files = ", ".join(d["_file"].replace("run_", "").replace(".json", "")
                          for d in g)
        print(f"{model:<24}{n_items:>8}{'是' if forced else '否':>10}"
              f"{len(g):>10}  {files}")
    print()

    analyzed = 0
    md = []
    md.append("# 置信分稳定性报告（重复测量）")
    md.append("")
    md.append("## 为什么必须重复测量")
    md.append("")
    md.append("单次运行的结果**不能代表真实发生率**。本项目已有直接证据：")
    md.append("")
    md.append("```")
    md.append("圆周率小数点后 10 位（K8），真值 3.1415926535")
    md.append("  第 1 次  答 3.1415926536   错")
    md.append("  第 2 次  答 3.1415926535   对")
    md.append("  第 3 次  答 3.1415926536   错")
    md.append("```")
    md.append("")
    md.append("三次错两次，而**置信分每次都是 98**。只跑一次就下结论，")
    md.append("「这个模型能不能算对这道题」会得到完全相反的答案。")
    md.append("")
    md.append("同理，「DeepSeek 1.0 对 智谱 0.867」这种模型对比，如果各自只跑一次，")
    md.append("差距可能纯粹来自运气，不足以支撑任何结论。")
    md.append("")
    md.append("## 运行记录概览")
    md.append("")
    md.append("| 模型 | 样本数 | 强制作答 | 运行次数 |")
    md.append("|---|---|---|---|")
    for key in sorted(groups, key=lambda k: (-len(groups[k]), k[0])):
        model, n_items, forced = key
        md.append(f"| {model} | {n_items} | {'是' if forced else '否'} | "
                  f"{len(groups[key])} |")
    md.append("")
    md.append("> 只有**样本集相同**的运行才能比较 —— 数据集从 21 条一路扩到 54 条，")
    md.append("> 跨规模的运行放在一起算方差毫无意义。")
    md.append("")

    for key in sorted(groups, key=lambda k: (k[0], k[1])):
        g = groups[key]
        if len(g) < args.min_runs:
            continue
        analyzed += 1
        text = report_group(key, stability_of_group(g))
        print(text)
        md.append(text)

    if analyzed == 0:
        print(f"没有任何配置达到 {args.min_runs} 次运行的阈值。")
        print("请先重复测量，例如：")
        print("  .venv\\Scripts\\python.exe repeat_run.py deepseek 5")
        return 1

    print("=" * 84)
    print("如何解读")
    print("=" * 84)
    print("""
  · 指标标准差为 0 —— 说明该配置下结论稳定，单次结果即可代表。
  · 指标标准差 > 0 —— **单次运行的数字不足以支撑结论**，必须报出均值与区间。
  · 正确性翻转的题目最值得关注：同一道题有时对、有时错，说明模型对它的
    掌握处于边界状态，而**置信分往往看不出来**（K8 三次都报 98 分，
    却错了两次）。这正是"高置信但不可靠"的具体形态。
  · 置信分波动大，说明自评分数本身带随机性 —— 这也解释了为什么
    基于置信分做精细阈值调节意义有限。
""")
    md.append("## 怎么读这份报告")
    md.append("")
    md.append("- **指标标准差为 0** —— 该配置下结论稳定，单次结果即可代表。")
    md.append("- **指标标准差 > 0** —— 单次运行的数字不足以支撑结论，必须报区间。")
    md.append("- **正确性翻转的题目最值得关注**：同一道题有时对、有时错，说明模型")
    md.append("  对它的掌握处于边界状态，而**置信分往往看不出来**。")
    md.append("  这正是「高置信但不可靠」的具体形态。")
    md.append("- **置信分波动**说明自评分数本身带随机性 —— 这也解释了为什么")
    md.append("  基于置信分做精细阈值调节意义有限。")
    md.append("")
    open(os.path.join("reports", "04_稳定性报告.md"), "w", encoding="utf-8").write("\n".join(md))
    print("  报告已写入 " + os.path.join("reports", "04_稳定性报告.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
