"""
调用 DeepSeek API 示例（增强版）

功能：
  - 多轮流式对话（stream=True）
  - 网络超时 / 限流自动重试（指数退避）
  - 每轮 Token 计费显示 + 累积统计
"""

import os
import json
import time
from pathlib import Path

# ─── 优先尝试 python-dotenv，不存在则手动解析 ───
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
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

import requests  # noqa: E402

# ─── API 请求配置 ───
URL = "https://api.deepseek.com/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

# ─── 重试配置 ───
MAX_RETRIES = 3
INITIAL_BACKOFF = 1  # 秒，首次重试等待

# ─── 计费单价（参考 DeepSeek Chat 官方定价） ───
# 输入 ¥0.001 / 1K tokens，输出 ¥0.002 / 1K tokens
PRICE_PROMPT_PER_1K = 0.001
PRICE_COMPLETION_PER_1K = 0.002


# ═══════════════════════════════════════════════════════════
#  重试 + 流式请求
# ═══════════════════════════════════════════════════════════

def stream_with_retry(payload):
    """
    带指数退避重试的流式请求。
    返回 (full_content, usage_dict_or_None)
    遇不可恢复错误则抛出异常。
    """
    last_error = None

    for attempt in range(1, MAX_RETRIES + 2):  # 第 1 次是原始请求
        try:
            resp = requests.post(
                URL, headers=HEADERS, json=payload,
                stream=True, timeout=60,
            )

            # ── 限流 429：解析 Retry-After 后重试 ──
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 2 ** attempt))
                print(f"\n  ⚠️ 触发限流 (429)，等待 {retry_after}s 后自动重试 ...")
                time.sleep(retry_after)
                continue

            resp.raise_for_status()  # 其他非 2xx 直接抛异常

            # ── 正常处理流式响应 ──
            full_content = ""
            usage = None

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

                # 流式最后一段可能携带 usage（需 stream_options）
                if "usage" in chunk:
                    usage = chunk["usage"]

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    print(token, end="", flush=True)
                    full_content += token

            print()  # 流式结束换行
            return full_content, usage

        except requests.exceptions.Timeout:
            last_error = "请求超时"
            if attempt > MAX_RETRIES:
                raise
            backoff = INITIAL_BACKOFF * (2 ** (attempt - 1))
            print(f"\n  ⚠️ 网络超时，{backoff}s 后自动重试 (第 {attempt}/{MAX_RETRIES} 次)...")
            time.sleep(backoff)

        except requests.exceptions.ConnectionError:
            last_error = "连接断开"
            if attempt > MAX_RETRIES:
                raise
            backoff = INITIAL_BACKOFF * (2 ** (attempt - 1))
            print(f"\n  ⚠️ 连接断开，{backoff}s 后自动重试 (第 {attempt}/{MAX_RETRIES} 次)...")
            time.sleep(backoff)

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            if 500 <= status < 600:
                if attempt > MAX_RETRIES:
                    raise
                backoff = INITIAL_BACKOFF * (2 ** (attempt - 1))
                print(f"\n  ⚠️ 服务端错误 ({status})，{backoff}s 后自动重试 ...")
                time.sleep(backoff)
            else:
                raise  # 非 5xx 的 HTTP 错误直接抛出

    # 所有重试耗尽
    raise RuntimeError(f"请求失败，已重试 {MAX_RETRIES} 次: {last_error}")


# ═══════════════════════════════════════════════════════════
#  计费计算与展示
# ═══════════════════════════════════════════════════════════

def format_cost(prompt_tokens, completion_tokens):
    """计算并返回费用说明字符串"""
    prompt_cost = prompt_tokens / 1000 * PRICE_PROMPT_PER_1K
    completion_cost = completion_tokens / 1000 * PRICE_COMPLETION_PER_1K
    total_cost = prompt_cost + completion_cost
    return (f"输入 {prompt_tokens:>6} tokens  ≈ ¥{prompt_cost:.6f}\n"
            f"输出 {completion_tokens:>6} tokens  ≈ ¥{completion_cost:.6f}\n"
            f"合计 {prompt_tokens + completion_tokens:>6} tokens  ≈ ¥{total_cost:.6f}")


# ═══════════════════════════════════════════════════════════
#  主对话循环
# ═══════════════════════════════════════════════════════════

# 初始化对话：system + 第一轮 user 消息
messages = [
    {"role": "system", "content": "你是一个乐于助人的助手"},
    {"role": "user",   "content": "北京邮电大学是什么样的大学？请用中文回答"},
]

# 累积计费
cumulative_prompt = 0
cumulative_completion = 0

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
        "stream_options": {"include_usage": True},  # 获取流式用量
    }

    try:
        content, usage = stream_with_retry(payload)
    except Exception as e:
        print(f"\n  ❌ {e}")
        break

    # 追加本轮 assistant 回答到消息历史
    messages.append({"role": "assistant", "content": content})

    # ── 计费统计 ──
    if usage:
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
    else:
        # 降级：按字符数粗略估算（~2 chars ≈ 1 token for Chinese）
        pt_est = sum(len(m["content"]) for m in messages) // 2
        ct_est = len(content) // 2
        pt, ct = pt_est, ct_est
        print("  ℹ️ 未获取到用量信息，已按字符数估算")

    cumulative_prompt += pt
    cumulative_completion += ct

    print(f"\n{'─' * 40}")
    print(f"  【本轮 Token 计费】")
    print(f"  {format_cost(pt, ct)}")
    print(f"{'─' * 40}")
    print(f"  【累积计费】")
    print(f"  输入 {cumulative_prompt:>6}  +  输出 {cumulative_completion:>6}"
          f"  =  {cumulative_prompt + cumulative_completion:>6}  tokens")
    total_cost = (cumulative_prompt / 1000 * PRICE_PROMPT_PER_1K
                  + cumulative_completion / 1000 * PRICE_COMPLETION_PER_1K)
    print(f"  总费用 ≈ ¥{total_cost:.6f}")
    print(f"{'─' * 40}")
    print(f"  [消息历史已累积 {len(messages)} 条（含 system）]")

    # 获取下一轮用户输入
    print(f"\n{'─' * 60}")
    user_input = input("请输入你的下一句话（输入 q 退出）: ").strip()
    if user_input.lower() == "q":
        # 退出时汇总
        print(f"\n{'=' * 60}")
        print("📊 对话结束 — 最终计费汇总")
        print(f"{'=' * 60}")
        final_cost = (cumulative_prompt / 1000 * PRICE_PROMPT_PER_1K
                      + cumulative_completion / 1000 * PRICE_COMPLETION_PER_1K)
        print(f"  总轮次        {turn}")
        print(f"  输入 tokens   {cumulative_prompt}")
        print(f"  输出 tokens   {cumulative_completion}")
        print(f"  总计 tokens   {cumulative_prompt + cumulative_completion}")
        print(f"  总费用        ¥{final_cost:.6f}")
        print(f"{'=' * 60}")
        print("✓ 对话结束")
        break
    messages.append({"role": "user", "content": user_input})
