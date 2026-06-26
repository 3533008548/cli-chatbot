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

payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "你是一个乐于助人的助手"},
        {"role": "user",   "content": "北京邮电大学的英文缩写是什么？请用中文回答"},
    ],
    "stream": False,
}

# ─── 3. 发送请求 ───
print(">>> 正在调用 DeepSeek API ...\n")

resp = requests.post(url, headers=headers, json=payload, timeout=30)
resp.raise_for_status()  # 非 2xx 直接抛异常

data = resp.json()

# ─── 4. 打印原始 JSON（方便理解 API 返回结构） ───
print("=" * 60)
print("【步骤 3】完整原始 JSON 响应：")
print("=" * 60)
print(json.dumps(data, ensure_ascii=False, indent=2))
print()

# ─── 5. 提取并打印模型回答 ───
print("=" * 60)
print("【步骤 4】模型回答内容：")
print("=" * 60)
content = data["choices"][0]["message"]["content"]
print(content)
print()

# ─── 6. 打印 token 用量 ───
print("=" * 60)
print("【步骤 5】Token 用量统计：")
print("=" * 60)
usage = data["usage"]
print(f"  prompt_tokens      = {usage['prompt_tokens']}")
print(f"  completion_tokens  = {usage['completion_tokens']}")
print(f"  total_tokens       = {usage['total_tokens']}")
print()

print("✓ 全部执行完毕，退出码 0")
