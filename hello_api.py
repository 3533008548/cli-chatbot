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

# ─── 2. 多轮流式对话 ───
import requests  # noqa: E402 (延迟导入，让用户先看到上面的错误信息)

url = "https://api.deepseek.com/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

# 初始化对话：system + 第一轮 user 消息
messages = [
    {"role": "system", "content": "你是一个乐于助人的助手"},
    {"role": "user",   "content": "北京邮电大学是什么样的大学？请用中文回答"},
]

print("=" * 60)
print("【多轮流式对话】开始（输入 q 退出）")
print("=" * 60)

turn = 0
while True:
    turn += 1
    role_label = "助手" if turn > 1 else "助手（第一轮）"

    print(f"\n{'─' * 60}")
    print(f">>> 第 {turn} 轮 — {role_label} 流式回复中 ...")
    print(f"{'─' * 60}")

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "stream": True,
    }

    full_content = ""
    try:
        with requests.post(url, headers=headers, json=payload,
                           stream=True, timeout=60) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                raw = line.removeprefix("data: ")
                if raw.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    print(token, end="", flush=True)
                    full_content += token
        print()  # 换行
    except Exception as e:
        print(f"\n  ❌ 请求失败: {e}")
        break

    # 追加本轮 assistant 回答到消息历史
    messages.append({"role": "assistant", "content": full_content})

    print(f"\n  [消息历史已累积 {len(messages)} 条（含 system）]")

    # 获取下一轮用户输入，追加后继续
    print(f"\n{'─' * 60}")
    user_input = input("请输入你的下一句话（输入 q 退出）: ").strip()
    if user_input.lower() == "q":
        print("\n✓ 对话结束")
        break
    messages.append({"role": "user", "content": user_input})
