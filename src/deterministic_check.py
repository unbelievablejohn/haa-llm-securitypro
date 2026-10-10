# -*- coding: utf-8 -*-
"""
deterministic_check.py —— 确定性校验：不依赖任何模型的判断

为什么需要它
------------
三段式流水线用「第二个模型」来交叉验证，但它有两个绕不过去的缺陷：

1. **共错无法发现**：两个模型给出同一个错误答案时，比对结果一致 → 判定为"通过"。
2. **一边拒答、一边编造时检测不出**：程序层原先只比对"双方都出现的实体"，
   而拒答文本里没有任何实体可比，于是"没有矛盾"。

**但有一类问题根本不需要问模型。**

    847 乘以 9639 等于多少？

正确答案是确定的，用计算器算一下就知道。**问 AI 反而引入了出错的可能** ——
实测中主模型对这类题的作答错误率极高（置信分 90~100，答案全错）。

本模块提供的正是这类"确定性校验"：能算出答案的，就直接算，然后核对模型有没有算错。
它**不需要第二个模型、不需要 API、不花一分钱**，而且正确率是 100%。

覆盖范围与边界
--------------
能处理：纯算术表达式（含中文算符、括号、多步运算、乘方、百分数）
不能处理：应用题（需要理解题意才能列式）、单位换算、需要外部事实的问题

**边界必须说清楚**：它只能覆盖"答案可以被程序算出"的问题。
对自由文本的事实性陈述（如"某定理是哪年证明的"），它无能为力 ——
那类问题仍然只能靠检索、第二个模型或人工。
"""

import ast
import re

# ---------------------------------------------------------------------------
# 中文算符 -> 符号
#
# 注意：这里**故意不包含 "的" -> "*"**。
# 曾经加过这条，结果把「2 的 32 次方」先变成「2 * 32 次方」，
# 于是抽出的算式是 2*32=64 而不是 2³² —— 属于静默算错，比不处理更危险。
# 「A 的 B 次方」改由 _norm 里的正则单独处理。
# ---------------------------------------------------------------------------
CN_OPS = [
    ("乘以", "*"), ("乘上", "*"), ("乘", "*"),
    ("除以", "/"), ("除上", "/"),
    ("加上", "+"), ("加", "+"),
    ("减去", "-"), ("减", "-"),
    ("次方", "**"), ("平方", "**2"), ("立方", "**3"),
]

# 各种"减号"都要归一 —— 它们在 Unicode 里是不同字符，
# 只处理全角那一个会漏掉 U+2212（数学减号），实测题目里就有它
MINUS_VARIANTS = "\u2212\u2013\u2014\u2015\uff0d\ufe63\u2796"

# 全角 -> 半角
FULLWIDTH = str.maketrans("０１２３４５６７８９＋－×÷（）　＝", "0123456789+-*/() =")

# 允许出现在表达式里的字符
EXPR_CHARS = re.compile(r"[0-9+\-*/().\s]")

# 安全求值允许的 AST 节点
_SAFE_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.FloorDiv, ast.USub, ast.UAdd,
)


def _norm(text):
    """全角转半角、各种减号统一、中文算符转符号。"""
    s = str(text or "")
    # 先把所有减号变体统一成 ASCII 减号（U+2212 与全角减号是两个不同字符）
    for ch in MINUS_VARIANTS:
        s = s.replace(ch, "-")
    s = s.translate(FULLWIDTH)
    # 「A 的 B 次方」/「A 的 B 次幂」-> A**B
    s = re.sub(r"(\d+)\s*的\s*(\d+)\s*次[方幂]", r"\1**\2", s)
    for cn, sym in CN_OPS:
        s = s.replace(cn, sym)
    return s


def safe_eval(expr):
    """
    安全地求值一个算术表达式。

    刻意不用裸 eval()：表达式来自模型/用户的文本，裸 eval 会执行任意代码。
    这里的做法是先解析成 AST，**逐个节点检查只允许算术相关节点**
    （常量、加减乘除、乘方、正负号），没有任何 Name / Call 节点，
    因此不可能访问变量或调用函数；校验通过后再求值。

    注意：不能用 ast.literal_eval —— 它只处理字面量，
    遇到 "847 * 9639" 会直接返回 None 而不是计算。这个坑踩过一次。

    另外防住超大乘方（如 9**9**9 会吃掉内存与时间）。
    """
    expr = str(expr or "").strip()
    if not expr:
        return None
    # 表达式长度上限：正常算式不会很长，超长的一律不处理
    if len(expr) > 200:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, MemoryError):
        return None

    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            return None
        # 乘方的指数必须很小，否则会算出天文数字。
        # 注意 AST 结构：Pow 是**运算符节点**，没有 left/right；
        # left/right 在 BinOp 上。曾经写成 node.right 而报 AttributeError。
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exp_node = node.right
            if not (isinstance(exp_node, ast.Constant)
                    and isinstance(exp_node.value, (int, float))
                    and abs(exp_node.value) <= 64):
                return None

    try:
        # 已经确认无 Name / Call 节点，此处求值是安全的
        val = eval(compile(tree, "<expr>", "eval"),
                   {"__builtins__": {}}, {})
    except (ArithmeticError, ValueError, TypeError, OverflowError, MemoryError):
        return None

    # 结果规模上限，防止意外产生超大整数
    if isinstance(val, int) and val.bit_length() > 4096:
        return None
    return val


def extract_expressions(question):
    """
    从自然语言问题里抽出算术表达式。

    做法：先归一化，再扫描所有"含数字且含运算符"的连续片段。
    只保留能安全求值、且至少有一个运算符的片段。

    返回 [(原始片段, 计算值), ...]，按出现顺序去重。
    """
    s = _norm(question)
    found = []
    seen = set()

    # 扫描最长的、只由 数字/运算符/括号/空格 组成的片段
    for m in re.finditer(r"[0-9+\-*/().\s]{5,}", s):
        raw = m.group(0).strip()
        if not raw:
            continue
        # 至少要有两个数字和一个运算符，才算一个真正的算式
        if len(re.findall(r"\d+", raw)) < 2:
            continue
        if not re.search(r"[+\-*/]", raw):
            continue
        # 结尾不能是运算符（"847 * 9639 =" 这种要剪掉）
        raw = raw.strip("=+-*/ ")
        if not re.search(r"[+\-*/]", raw):
            continue
        val = safe_eval(raw)
        if val is None:
            continue
        key = (raw, val)
        if key in seen:
            continue
        seen.add(key)
        found.append((raw, val))
    return found


def _fmt_candidates(val):
    """
    一个数值可能的各种写法（模型输出格式不固定）。
    例如 8164233 -> {"8164233", "8164233.0", "8,164,233"}
    """
    out = set()
    if val is None:
        return out
    # 整数
    if abs(val - round(val)) < 1e-9:
        n = int(round(val))
        out.add(str(n))
        out.add(f"{n:,}")
    # 浮点：保留合理位数
    for p in (0, 1, 2, 3, 4, 6):
        f = f"{val:.{p}f}".rstrip("0").rstrip(".")
        if f:
            out.add(f)
    out.add(str(val))
    return out


def _numbers_in(text):
    """抽出文本里的所有数值（含千分位与小数）。"""
    nums = set()
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", _norm(text)):
        v = m.group(0).replace(",", "")
        nums.add(v)
        try:
            f = float(v)
            if abs(f - round(f)) < 1e-9:
                nums.add(str(int(round(f))))
            else:
                nums.add(f"{f:g}")
        except ValueError:
            pass
    return nums


def check(question, answer):
    """
    确定性校验：如果问题里含可计算的算式，核对答案里有没有算出正确结果。

    返回 dict：
        applicable  bool          本题是否适用确定性校验
        expressions list          识别出的算式
        results     list          计算出的正确结果
        matched     bool          答案中是否出现了正确结果
        verdict     str           "ok" / "mismatch" / "not_applicable"
        detail      str
    """
    exprs = extract_expressions(question)
    if not exprs:
        return {"applicable": False, "expressions": [], "results": [],
                "matched": None, "verdict": "not_applicable",
                "detail": "问题中未识别出可计算的算式"}

    ans_nums = _numbers_in(answer)
    results = []
    for raw, val in exprs:
        results.append({"expr": raw, "value": val})
        want = _fmt_candidates(val)
        # 答案里出现了其中任一写法，就算算对
        if want & ans_nums:
            return {"applicable": True, "expressions": [e[0] for e in exprs],
                    "results": results, "matched": True, "verdict": "ok",
                    "detail": f"答案中出现了正确结果 {val}"}

    vals = [r["value"] for r in results]
    return {"applicable": True,
            "expressions": [e[0] for e in exprs],
            "results": results, "matched": False, "verdict": "mismatch",
            "detail": f"计算结果为 {vals}，但答案中未出现"}


def summarize(result):
    """压成一行，便于打进报告。"""
    v = result.get("verdict")
    if v == "not_applicable":
        return "不适用确定性校验"
    if v == "ok":
        return f"确定性校验通过（{result['detail']}）"
    if v == "mismatch":
        return f"[算错] {result['detail']}"
    return "确定性校验未知"


if __name__ == "__main__":
    import sys as _sys
    for _s in (_sys.stdout, _sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    print("=" * 78)
    print("确定性校验 · 自测")
    print("=" * 78)
    cases = [
        # (问题, 模型答案, 期望)
        ("847 乘以 9639 等于多少？", "847 × 9639 = 8,164,233", "ok"),
        ("847 乘以 9639 等于多少？", "847 乘以 9639 的精确数值为 8,188,333。", "mismatch"),
        ("137 乘以 491 等于多少？", "等于 67,267", "ok"),
        ("137 乘以 491 等于多少？", "等于 67,567", "mismatch"),
        ("计算 (847 × 9639 − 1234567) ÷ 2 的结果。",
         "结果是 3,464,833", "ok"),
        ("计算 (847 × 9639 − 1234567) ÷ 2 的结果。",
         "结果是 3,472,293", "mismatch"),
        ("2 的 32 次方等于多少？", "4294967296", "ok"),
        # 应用题：需要先理解题意才能列式，本模块**做不到**，如实标为不适用
        ("一个边长为 12 厘米的正方体，其表面积是多少平方厘米？", "864",
         "not_applicable"),
        # 不应适用
        ("水在标准大气压下的沸点是多少摄氏度？", "100 摄氏度", "not_applicable"),
        ("《Global Carbon Policy Review》的主编是谁？", "我无法确认", "not_applicable"),
    ]
    ok = 0
    for q, a, expect in cases:
        r = check(q, a)
        good = r["verdict"] == expect
        ok += good
        print(f"\n  [{'OK' if good else '不符预期'}] 期望={expect}")
        print(f"      Q: {q[:62]}")
        print(f"      A: {a[:62]}")
        print(f"      算式={r['expressions']}  结果={[x['value'] for x in r['results']]}")
        print(f"      判定={r['verdict']}  {r['detail'][:62]}")
    print()
    print(f"  自测通过 {ok}/{len(cases)}")
