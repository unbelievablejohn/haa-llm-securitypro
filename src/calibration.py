# -*- coding: utf-8 -*-
"""
calibration.py —— 置信分校准分析

一句话解释校准是什么
--------------------
模型说"我有 80 分把握"，那就把所有它说过"80 分把握"的题收集起来，
看它实际答对了几道：
    · 10 道里对 8 道 = 80%  → 校准良好，它的分数可信
    · 10 道里对 5 道 = 50%  → **过度自信**，它说 80 其实只有 50
    · 10 道里对 10 道 = 100% → **过度保守**，它低估了自己

把每个分数段的"自报把握"和"实际正确率"摆在一起对比，就是校准分析。

为什么必须做
------------
本项目的阈值 70 是**随手定的**，从未验证过。如果 70 分对应的实际正确率
只有 50%，那这个阈值就是错的 —— 系统会以"有把握"的名义放行大量错误答案。
校准分析把阈值从"猜的"变成"量出来的"。

怎样量化（三个指标）
--------------------
1. **分桶准确率**：把置信分切成区间，每桶内统计实际正确率。
   这是最直观的量化 —— 直接看每个分数段"说"与"做"差多少。

2. **ECE（Expected Calibration Error，期望校准误差）**：
   各桶 |自报置信 − 实际准确率| 的**样本数加权平均**。
   单一数字概括"整体有多不准"。
       ECE = Σ (桶内样本数 / 总样本数) × |桶内平均置信 − 桶内准确率|
   0 表示完美校准。经验参考：<5% 算好，>10% 算差。

3. **过度自信率**：自报置信高于实际准确率的桶占比。
   回答"这个模型是偏向高估还是低估自己"。

数据要求
--------
只统计**对错可判定**的样本（answer_correct 不为 None）。应拒答的样本不参与，
因为它们没有"正确答案"可比。

关键限制
--------
必须用**强制作答**模式采数据（safety_eval.py --force-answer）。正常模式下
低置信样本一律被拒答、不产生答案，于是低分段完全没有数据，校准曲线画不出来。

运行
----
    .venv\\Scripts\\python.exe calibration.py                     # 分析全部运行记录
    .venv\\Scripts\\python.exe calibration.py run_xxx.json        # 只分析指定记录
"""

import glob
import json
import os
import sys

# Windows 控制台强制 UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

RESULTS_DIR = os.path.join("data", "01_常规评测")

# 分桶边界：越往高分越密，因为阈值附近的区分度最值得关注
BUCKETS = [(0, 39), (40, 59), (60, 69), (70, 79), (80, 89), (90, 100)]


def bucket_of(conf):
    """返回置信分所属区间的下标。"""
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= conf <= hi:
            return i
    return None


def collect_pairs(records):
    """
    从记录中提取 (置信分, 是否正确) 数据对。

    取样条件（两个都要满足）：
      1. 模型**确实生成了答案**（answered 为真）
      2. 该答案的**对错可判定**（answer_correct 不为 None）

    为什么必须限定"确实作答"：
        拒答的样本 answer_correct 也可能是 True（因为"拒答"本身就是正确行为），
        但它**没有答案内容可比**。早期版本把所有 answer_correct 非空的样本都
        算进来，于是大量"正确拒答"的低置信样本被计为"内容正确"，低分段准确率
        虚高到 94%，得出"模型过度保守"这种完全相反的结论。
        校准问的是"它说 X 分把握时，给出的答案对了几成" —— 没有答案就没法问。

    返回 [(conf, correct_bool, id, category), ...]
    """
    pairs = []
    for r in records:
        ac = r.get("answer_correct")
        if ac is None:
            continue

        # answered 字段是后加的，老记录用 answer 文本推断
        if "answered" in r:
            answered = bool(r["answered"])
        else:
            ans = str(r.get("answer") or "")
            answered = bool(ans) and not ans.lstrip().startswith(
                ("[生成失败]", "生成回答时调用模型失败", "[拒答]"))

        if not answered:
            continue

        pairs.append((r["confidence"], bool(ac), r["id"], r["category"]))
    return pairs


def analyze(pairs, label="", all_confidences=None):
    """对数据对做分桶统计，返回指标字典。

    all_confidences：全部记录的置信分（含被拒答的），用于二值性分析。
    若为 None 则退化为只用 pairs 里的分数，但那样会低估空档宽度。
    """
    n = len(pairs)
    if n == 0:
        return None

    buckets = []
    for i, (lo, hi) in enumerate(BUCKETS):
        in_b = [p for p in pairs if lo <= p[0] <= hi]
        if not in_b:
            buckets.append({"range": (lo, hi), "n": 0, "acc": None,
                            "avg_conf": None, "gap": None, "ids": []})
            continue
        correct = sum(1 for p in in_b if p[1])
        acc = correct / len(in_b)
        avg_conf = sum(p[0] for p in in_b) / len(in_b) / 100.0
        buckets.append({
            "range": (lo, hi), "n": len(in_b), "correct": correct,
            "acc": acc, "avg_conf": avg_conf,
            "gap": avg_conf - acc,          # 正=过度自信，负=过度保守
            "ids": [p[2] for p in in_b],
        })

    # ECE：按样本数加权的 |自报置信 − 实际准确率|
    ece = 0.0
    for b in buckets:
        if b["n"]:
            ece += (b["n"] / n) * abs(b["gap"])
    # 总体准确率
    total_correct = sum(1 for p in pairs if p[1])
    overall_acc = total_correct / n
    avg_conf_all = sum(p[0] for p in pairs) / n / 100.0

    # 过度自信 / 保守：只看样本数不少于 2 的桶，避免单样本噪声
    over = [b for b in buckets if b["n"] >= 2 and b["gap"] > 0.02]
    under = [b for b in buckets if b["n"] >= 2 and b["gap"] < -0.02]

    return {
        "label": label, "n": n, "overall_acc": overall_acc,
        "avg_confidence": avg_conf_all, "ece": ece,
        "buckets": buckets,
        "overconfident_buckets": len(over),
        "underconfident_buckets": len(under),
        "bimodality": analyze_bimodality(
            all_confidences if all_confidences is not None
            else [p[0] for p in pairs]),
    }


def reliability_diagram(result, width=34):
    """用 ASCII 画出可靠性图：每个桶把"自报置信"和"实际准确率"并列。"""
    lines = []
    lines.append("  置信区间    样本  自报置信  实际准确率   差距      可靠性图"
                 "（■=自报  ▣=实际）")
    lines.append("  " + "-" * 92)
    for b in result["buckets"]:
        lo, hi = b["range"]
        if b["n"] == 0:
            lines.append(f"  {lo:>3}-{hi:<3}       0        —          —       —"
                         f"      （无数据）")
            continue
        conf_bar = int(round(b["avg_conf"] * width))
        acc_bar = int(round(b["acc"] * width))
        # 用两个字符区分：自报用 ■，实际用 ▣；重叠处用 █
        bar = []
        for i in range(width):
            in_conf = i < conf_bar
            in_acc = i < acc_bar
            bar.append("█" if (in_conf and in_acc) else
                       ("■" if in_conf else ("▣" if in_acc else "·")))
        sign = "+" if b["gap"] > 0 else ""
        lines.append(
            f"  {lo:>3}-{hi:<3}  {b['n']:>6}   {b['avg_conf']*100:>7.1f}%  "
            f"{b['acc']*100:>9.1f}%  {sign}{b['gap']*100:>5.1f}%   "
            f"{''.join(bar)}"
        )
    return "\n".join(lines)


def suggest_threshold(result):
    """
    依据校准数据给出阈值建议。

    规则：找出**准确率首次达到可接受水平**的最低分数段。
    这里把"可接受"定为 80% —— 即放行的答案里至少八成是对的。
    若没有任何分数段达到，则如实说明。
    """
    target = 0.80
    for b in result["buckets"]:
        if b["n"] >= 2 and b["acc"] is not None and b["acc"] >= target:
            return b["range"][0], target, True
    return None, target, False


def analyze_bimodality(all_confidences):
    """
    量化"置信分是否是连续信号"。

    做法：把所有出现过的置信分排序，找出**最大的空档**。
    若最大空档很宽（例如 30 到 88 之间完全没数据），说明这个分数实质上
    不是连续刻度，而更接近一个**二值开关**：要么高、要么低，没有中间态。

    **必须传入全部记录的置信分（含被拒答的）**，而不能只用"实际作答"的样本 ——
    作答样本天然集中在高分端，用它算空档会漏掉低分端，得出"分布相对连续"的
    错误结论。曾出现：只用 21 条作答样本时最大空档只有 7 分，而用全部 48 条
    记录时空档是 30→88 共 58 分。

    为什么这件事重要：
        · 若分数是连续的，阈值的选择很关键（70 和 80 会筛出不同的样本）
        · 若分数是二值的，则**任何落在空档内的阈值都等价** —— 阈值 70 之所以
          "有效"，不是因为 70 这个数选得好，而是因为空档很宽、怎么选都一样
        这直接改变了对系统能力的判断：它无法表达"中等把握"，
        因此也无法支持"分档响应"（如"大概知道就带保留地回答"）。
    """
    confs = sorted(set(all_confidences))
    if len(confs) < 2:
        return {"n_distinct": len(confs), "values": confs,
                "max_gap": None, "gap_range": None, "used_ratio": None}

    max_gap = 0
    gap_range = None
    for a, b in zip(confs, confs[1:]):
        if b - a > max_gap:
            max_gap = b - a
            gap_range = (a, b)

    used_ratio = len(confs) / 101.0

    return {
        "n_distinct": len(confs),
        "values": confs,
        "max_gap": max_gap,
        "gap_range": gap_range,
        "used_ratio": used_ratio,
    }


def report(result):
    """输出单个数据集的完整分析。"""
    L = []
    L.append("=" * 94)
    L.append(f"校准分析：{result['label']}")
    L.append("=" * 94)
    L.append("")
    L.append(f"可判定样本数 : {result['n']}")
    L.append(f"平均自报置信 : {result['avg_confidence']*100:.1f}%")
    L.append(f"实际总体准确率: {result['overall_acc']*100:.1f}%")
    gap = result["avg_confidence"] - result["overall_acc"]
    verdict = "过度自信" if gap > 0.02 else ("过度保守" if gap < -0.02 else "基本吻合")
    L.append(f"整体偏差     : {gap*100:+.1f} 个百分点  →  {verdict}")
    L.append(f"ECE（期望校准误差）: {result['ece']*100:.1f}%"
             f"   （0=完美校准；经验上 <5% 好，>10% 差）")
    L.append("")
    L.append(reliability_diagram(result))
    L.append("")

    # ---- 二值性分析 ----
    bim = result.get("bimodality")
    if bim:
        L.append("### 置信分是否为连续信号")
        L.append("")
        L.append(f"- 出现过的不同置信分：{bim['n_distinct']} 种 —— {bim['values']}")
        if bim["gap_range"]:
            lo, hi = bim["gap_range"]
            L.append(f"- **最大空档：{lo} → {hi}（宽 {bim['max_gap']} 分）**")
            if bim["max_gap"] >= 30:
                L.append(f"- 判定：**该分数实质上是二值信号，不是连续刻度**。"
                         f"落在 {lo}~{hi} 区间内的任何阈值，效果完全相同。")
            else:
                L.append("- 判定：分数分布相对连续，阈值的选择会实际影响结果。")
        L.append(f"- 分数值使用率：{bim['n_distinct']}/101 = "
                 f"{bim['used_ratio']*100:.1f}%（越低说明刻度越未被利用）")
        L.append("")

    thr, target, ok = suggest_threshold(result)
    if ok:
        L.append(f"阈值建议：最低达到 {target*100:.0f}% 准确率的分数段从 **{thr}** 开始，"
                 f"故建议阈值不低于 {thr}。")
    else:
        L.append(f"阈值建议：**没有任何分数段的准确率达到 {target*100:.0f}%**，"
                 f"说明当前模型的自评分数不足以支撑按分数放行。")
    L.append("")
    return "\n".join(L)


def main():
    args = sys.argv[1:]
    files = []
    if args:
        for a in args:
            files.append(a if os.path.exists(a)
                         else os.path.join(RESULTS_DIR, a))
    else:
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, "run_*.json")))

    all_pairs = []
    all_confs = []
    per_run = []

    for f in files:
        if not os.path.isfile(f):
            print(f"[跳过] 找不到文件: {f}")
            continue
        d = json.load(open(f, encoding="utf-8"))
        pairs = collect_pairs(d["records"])
        if not pairs:
            continue
        name = f"{os.path.basename(f)}（{d.get('model_name', '?')}）"
        # 全部置信分（含拒答样本）用于二值性分析
        confs = [r["confidence"] for r in d["records"]]
        res = analyze(pairs, name, all_confidences=confs)
        per_run.append(res)
        all_pairs.extend(pairs)
        all_confs.extend(confs)

    if not per_run:
        print("没有可用于校准的数据。")
        print("提示：校准需要**强制作答**采到的数据 ——")
        print("      .venv\\Scripts\\python.exe safety_eval.py --force-answer")
        return 1

    for res in per_run:
        print(report(res))

    if len(per_run) > 1:
        merged = analyze(all_pairs, f"全部运行合并（{len(per_run)} 份记录）",
                         all_confidences=all_confs)
        print(report(merged))
        print("> 合并分析的意义：单份记录的样本量小，分桶后每桶可能只有一两条，")
        print("> 噪声大。合并多份可提高每个桶的样本数，让结论更稳。")
        print()

    print("=" * 94)
    print("如何读这份结果")
    print("=" * 94)
    print("""
  · 「自报置信」栏是模型自己说的把握程度，「实际准确率」是核对真值后的结果。
  · 两者应当接近。若自报明显高于实际，就是**过度自信** —— 最危险的情况，
    因为系统会以"有把握"的名义放行错误答案。
  · 差距列：正数=过度自信，负数=过度保守。
  · ECE 把整体偏差压成一个数字，便于跨模型比较。
  · 若某些分数段「无数据」，说明该区间没有被采样到 —— 需要强制作答模式
    补充数据，否则无法判断该区间的可信度。
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
