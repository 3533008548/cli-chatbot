"""
调用 DeepSeek API 示例
从 .env 读取 API Key，发送消息并打印完整响应结构
"""

import os
import json
from pathlib import Path

# ─── 优先尝试 python-dotenv，不存在则手动解析 ───
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 手动解析 .env 文件（轻量兜底）
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

# ─── 1. 从环境变量读取 API Key ───
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError(
        "未找到 DEEPSEEK_API_KEY。\n"
        "请将 .env.example 复制为 .env，然后把你的 Key 填进去。\n"
        "参考命令: copy .env.example .env"
    )

# ─── 2. 构造请求 ───
import requests  # noqa: E402 (延迟导入，让用户先看到上面的错误信息)

url = "https://api.deepseek.com/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

messages = [
    {"role": "system", "content": "你是一个乐于助人的助手"},
    {"role": "user",   "content": "北京邮电大学是什么样的大学？请用中文回答"},
]

# ─── 3. Temperature 实验：遍历不同温度值 ───
temperatures = [0, 0.5, 1.0, 1.5]

print("=" * 60)
print("【Temperature 实验】开始对比测试")
print("=" * 60)

for temp in temperatures:
    print(f"\n{'─' * 60}")
    print(f">>> temperature = {temp}")
    print(f"{'─' * 60}")

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temp,
        "stream": False,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data["usage"]

        print(f"  【回答】{content}")
        print(f"  【Token 用量】prompt={usage['prompt_tokens']}, "
              f"completion={usage['completion_tokens']}, "
              f"total={usage['total_tokens']}")
    except Exception as e:
        print(f"  ❌ 请求失败: {e}")

print(f"\n{'=' * 60}")
print("✓ Temperature 实验全部执行完毕，退出码 0")
