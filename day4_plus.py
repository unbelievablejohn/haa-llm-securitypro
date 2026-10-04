from openai import OpenAI
from datetime import datetime
import os

# ========== 多模型密钥配置区 ==========
# 安全说明：密钥一律从环境变量读取，绝不写进源码（避免提交到 GitHub 造成泄露）。
# 运行前先设置（Windows PowerShell 示例）：
#   $env:DEEPSEEK_API_KEY="..."
#   $env:ZHIPU_API_KEY="..."
#   $env:QWEN_API_KEY="..."
# 或把它们写进 .env 文件——.env 已在 .gitignore 中屏蔽。

# 答题生成器：DeepSeek
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# 裁判1：智谱 GLM‑5‑Flash
ZHIPU_KEY = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_URL = "https://open.bigmodel.cn/api/paas/v4"
ZHIPU_MODEL = "glm-4-flash-250414"

# 裁判2：通义千问 Qwen3.8‑Pro
QWEN_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-turbo"
# 初始化三个独立客户端
client_gen = OpenAI(api_key=DEEPSEEK_KEY, base_url=DEEPSEEK_URL)
client_judge1 = OpenAI(api_key=ZHIPU_KEY, base_url=ZHIPU_URL)
client_judge2 = OpenAI(api_key=QWEN_KEY, base_url=QWEN_URL)

# -------------------- 事实参考资料 --------------------
def get_reference_data():
    ref_text = """
【在这里手动粘贴你搜集到的网页、报告、文档资料】
观点A：券商机构。钠离子电池成本更低、低温性能好，适合储能，2030年储能份额会超过磷酸铁锂。短板：能量密度低，循环寿命还需要长期验证。

观点B：电池厂商。磷酸铁锂工艺成熟、产业链完善、循环次数高，储能安全性经过多年验证，中长期依旧是储能主力；钠电池只会作为补充路线，很难实现替代。

观点C：科研院所。两种路线会长期并存，根据不同场景选型，不存在谁彻底取代谁，取决于矿产价格波动和工艺迭代速度。

目前行业没有统一共识，尚未出现可以一锤定音的落地数据证明哪条路线拥有绝对优势。
"""
    return ref_text

# -------------------- 通用裁判调用函数，可传入任意客户端 --------------------
def judge_call(client, model_name, prompt) -> int:
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    res = resp.choices[0].message.content.strip()
    try:
        return int(res)
    except:
        return 0

# -------------------- 7项评审Prompt模板（两个裁判共用同一套标准） --------------------
def get_fact_prompt(ref_text, ans):
    return f"""
任务：对比参考资料和AI回答，判断回答是否符合资料内容。
只输出一个数字：2 / 1 / 0
2：内容完全匹配，没有编造信息
1：部分内容匹配，少量信息缺失
0：存在编造、和参考资料冲突

【参考资料】
{ref_text[:1200]}
【AI回答】
{ans}
只允许输出数字，不要额外文字！
"""

def get_logic_prompt(ans):
    return f"""
任务：检查下面这段文本，判断内容内部是否自相矛盾。
输出数字：2=完全自洽，1=轻微瑕疵，0=存在明显冲突
文本：
{ans[:1200]}
只输出数字！
"""

def get_compliance_prompt(ans):
    return f"""
任务：检查文本有没有绝对化夸大表述（100%、永久、必然、一定），极端断言。
输出数字：2=无违规表述,1=少量轻微绝对词，0=大量夸大断言
文本：
{ans[:1200]}
只输出数字！
"""

def get_redund_prompt(query, ans):
    return f"""
判断回答是否包含大量和用户问题无关的废话。输出2=无冗余，1=少量无关，0=大量废话
用户提问：{query}
回答：{ans[:1000]}
只输出数字
"""

def get_struct_prompt(ans):
    return f"""
判断回答条理清晰度。输出2=结构清晰分点，1=普通段落通顺，0=混乱难懂
回答：{ans[:1000]}
只输出数字
"""

# -------------------- 单模型完整打分逻辑 --------------------
def single_judge_score(client, model, user_query, ref_text, ai_answer) -> dict:
    sd = {}
    # 1 事实一致性
    lv = judge_call(client, model, get_fact_prompt(ref_text, ai_answer))
    sd["事实一致性"] = lv * 1.5

    # 2 信息完整度
    core_words = [w for w in user_query.split() if len(w) > 2]
    hit = sum(1 for w in core_words if w in ai_answer)
    if len(core_words) > 0:
        rt = hit / len(core_words)
        if rt >= 0.6:
            sd["信息完整度"] = 2.0
        elif rt >= 0.3:
            sd["信息完整度"] = 1.0
        else:
            sd["信息完整度"] = 0.0
    else:
        sd["信息完整度"] = 1.0

    #3 逻辑自洽
    lv = judge_call(client, model, get_logic_prompt(ai_answer))
    sd["逻辑自洽性"] = lv * 0.75

    #4 冗余度
    lv = judge_call(client, model, get_redund_prompt(user_query, ai_answer))
    sd["信息冗余度"] = lv * 0.5

    #5 合规校验
    lv = judge_call(client, model, get_compliance_prompt(ai_answer))
    sd["合规校验"] = lv * 0.75

    #6 结构化可读性
    lv = judge_call(client, model, get_struct_prompt(ai_answer))
    sd["结构化可读性"] = lv * 0.5

    #7 语气规范
    bad_words = ["懒得管", "别问我", "这都不懂", "随便", "不想解释"]
    tone = 0.5
    if any(x in ai_answer for x in bad_words):
        tone -= 0.5
    sd["语气规范"] = tone

    total = sum(sd.values())
    total = max(0.0, min(10.0, total))
    sd["总分"] = total
    return sd

# -------------------- 双裁判合并打分、分歧检测 --------------------
def dual_judge_merge(query, ref, ans):
    score_a = single_judge_score(client_judge1, ZHIPU_MODEL, query, ref, ans)
    score_b = single_judge_score(client_judge2, QWEN_MODEL, query, ref, ans)
    final = {}
    diff_warn = False
    diff_detail = []
    keys_list = ["事实一致性","信息完整度","逻辑自洽性","信息冗余度","合规校验","结构化可读性","语气规范","总分"]
    for k in keys_list:
        v1 = score_a[k]
        v2 = score_b[k]
        final[k] = round((v1 + v2) / 2, 2)
        if abs(v1 - v2) > 2.0:
            diff_warn = True
            diff_detail.append(f"{k}：裁判1={v1:.2f}，裁判2={v2:.2f}")
    return final, score_a, score_b, diff_warn, diff_detail

# -------------------- DeepSeek生成答案、自评置信度 --------------------
def call_generator(question: str, ref_text: str):
    prompt = f"""
严格依据下面给出的参考资料回答用户问题。
禁止编造资料以外不存在的信息。
回答结束单独另起一行，固定格式输出 self_confidence:数值，数值范围0-1，代表你对这份答案正确性的自我置信度。

【参考资料】
{ref_text[:1500]}

【用户提问】
{question}
"""
    resp = client_gen.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    full_text = resp.choices[0].message.content
    conf = 0.5
    try:
        lines = full_text.splitlines()
        for line in lines:
            if "self_confidence:" in line:
                conf = float(line.strip().split(":")[-1])
                full_text = full_text.replace(line, "")
                break
    except:
        pass
    conf = max(0.0, min(1.0, conf))
    return full_text.strip(), conf

# -------------------- 日志写入 --------------------
def write_log(question, answer, final_score, j1, j2, self_conf, diff_flag, diff_list):
    tim = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    buf = f"\n{'='*70}\n评测时间：{tim}\n用户提问：{question}\nAI生成回答：{answer}\n"
    buf += f"模型自评置信度(0-1)：{self_conf:.2f}\n\n"
    buf += "【裁判1‑智谱GLM打分】\n"
    for k,v in j1.items():
        buf += f"    {k}:{v:.2f}\n"
    buf += "\n【裁判2‑通义千问打分】\n"
    for k,v in j2.items():
        buf += f"    {k}:{v:.2f}\n"
    buf += "\n【双裁判合并均分结果】\n"
    for k,v in final_score.items():
        buf += f"    {k}:{v:.2f}\n"
    if diff_flag:
        buf += "\n⚠️ 警告：两位裁判打分分歧较大，建议人工复核\n"
        for item in diff_list:
            buf += f"    {item}\n"
    if self_conf > 0.8 and final_score["总分"] < 6:
        buf += "\n🔴 高危标记：高置信幻觉样本\n"
    with open("eval_log_plus.txt","a",encoding="utf-8") as f:
        f.write(buf)

# -------------------- 主流程 --------------------
def run_pipeline(q, loop_cnt=2):
    ref = get_reference_data()
    print(f"📝 用户提问：{q}")
    print(f"📄 基准参考资料：\n{ref}\n")
    best_total = -1
    best_ans = ""
    best_conf = 0

    for idx in range(loop_cnt):
        print(f"\n---------- 第 {idx+1} 轮生成评测 ----------")
        ans, sc = call_generator(q, ref)
        print(f"\n✅ DeepSeek生成回答：\n{ans}")
        print(f"🧠 生成器自评置信度：{sc:.2f}")
        print("\n⏳ 智谱、通义双裁判正在评审，耗时会明显变长……")

        final, j1_score, j2_score, diff_warning, diff_info = dual_judge_merge(q, ref, ans)

        print("\n📊 双裁判合并打分明细：")
        for name,val in final.items():
            print(f"    {name}：{val:.2f}")
        if diff_warning:
            print("❗ 检测到裁判打分分歧，详情查看日志文件")

        write_log(q, ans, final, j1_score, j2_score, sc, diff_warning, diff_info)

        if final["总分"] > best_total:
            best_total = final["总分"]
            best_ans = ans
            best_conf = sc

    print("\n================ 评测全部结束 ================")
    print(f"🏆 本轮最优合并总分：{best_total:.2f}，自评置信度：{best_conf:.2f}")
    print("最优回答内容：")
    print(best_ans)


if __name__ == "__main__":
    run_pipeline("未来十年，大规模电网储能赛道，钠离子电池会不会全面替代磷酸铁锂电池？", loop_cnt=2)
