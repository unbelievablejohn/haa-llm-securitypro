# -*- coding: utf-8 -*-
"""重复测量驱动：对指定模型连续跑 N 次完整评测。

用 Python 而不是 PowerShell 脚本，因为 Windows PowerShell 5.1 会把 UTF-8
脚本按 GBK 解析，导致中文字符串报错（已踩过）。

【密钥约定】
    本文件**不含任何密钥**。密钥必须通过环境变量提供，由父进程传入：
        DEEPSEEK_API_KEY / ZHIPU_API_KEY
    这样文件本身可以安全地提交进版本库。
    早期版本曾把密钥直接写在这里 —— 那种做法一旦提交就等于泄露，
    必须杜绝。

用法：
    python repeat_run.py deepseek 5
    python repeat_run.py glm 5
"""
import os
import subprocess
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

PROJ = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(PROJ, ".venv", "Scripts", "python.exe")

# 每个目标模型：从哪个环境变量取密钥，以及可选的 base_url / model 覆盖
MODELS = {
    "deepseek": {
        "key_env": "DEEPSEEK_API_KEY",
        "HAA_BASE_URL": None,
        "HAA_MODEL": None,
        "label": "deepseek-chat",
    },
    "glm": {
        "key_env": "ZHIPU_API_KEY",
        "HAA_BASE_URL": "https://open.bigmodel.cn/api/paas/v4",
        "HAA_MODEL": "glm-4-flash-250414",
        "label": "glm-4-flash-250414",
    },
}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    name, times = sys.argv[1], int(sys.argv[2])
    cfg = MODELS.get(name)
    if not cfg:
        print(f"未知模型: {name}（可选 {'/'.join(MODELS)}）")
        return 1

    key = os.environ.get(cfg["key_env"], "").strip()
    if not key:
        print(f"未设置环境变量 {cfg['key_env']}，无法运行。")
        print(f'  $env:{cfg["key_env"]} = "你的密钥"')
        return 1

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["HAA_API_KEY"] = key
    for k in ("HAA_BASE_URL", "HAA_MODEL"):
        v = cfg[k]
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v

    print("=" * 74)
    print(f"重复测量：{cfg['label']}  共 {times} 次")
    print("=" * 74)

    for i in range(1, times + 1):
        print(f"\n---------- 第 {i} / {times} 次 ----------", flush=True)
        r = subprocess.run([PY, os.path.join(PROJ, "safety_eval.py")],
                           cwd=PROJ, env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        for line in (r.stdout or "").splitlines():
            key_words = ("行为准确率", "答案正确率", "confident-wrong",
                         "答案错误", "过度拒答", "结果 JSON")
            if any(kw in line for kw in key_words):
                print("  " + line.strip(), flush=True)
        if r.returncode != 0:
            print(f"  [警告] 第 {i} 次退出码 {r.returncode}")
            tail = (r.stderr or "")[-300:]
            if tail:
                print("  " + tail.replace("\n", "\n  "))

    print(f"\n完成：{cfg['label']} 共 {times} 次")
    return 0


if __name__ == "__main__":
    sys.exit(main())
