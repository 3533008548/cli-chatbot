"""
agent_tools_demo.py — Agent 工具调用演示

核心流程（完整 Agent 单步）：
  1. 用户提问
  2. LLM 返回 tool_calls（决定调哪个工具 + 参数）
  3. 把 tool_calls 加入消息历史
  4. 执行工具 → 得到结果
  5. 把工具结果（tool role）加入消息历史
  6. 再次调 LLM → 模型把工具结果组织成自然语言回答
  7. 重复 2-6，直到 LLM 不再请求工具

功能：加、减、乘、除四则运算
"""

import os
import json
import time
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════════════════════
#  复用 hello_api.py 的 API Key 加载逻辑
# ═══════════════════════════════════════════════════════════
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

API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    raise RuntimeError("未找到 DEEPSEEK_API_KEY，请检查 .env 文件")

import requests  # noqa: E402

URL = "https://api.deepseek.com/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


# ═══════════════════════════════════════════════════════════
#  第一步：Tool 描述（JSON Schema，供 LLM 识别）
# ═══════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "加法运算，返回 a + b 的结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "加数（第一个数）"},
                    "b": {"type": "number", "description": "加数（第二个数）"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subtract",
            "description": "减法运算，返回 a - b 的结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "被减数"},
                    "b": {"type": "number", "description": "减数"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "乘法运算，返回 a × b 的结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "乘数（第一个数）"},
                    "b": {"type": "number", "description": "乘数（第二个数）"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "divide",
            "description": "除法运算，返回 a ÷ b 的结果，除数为零时返回错误信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "被除数"},
                    "b": {"type": "number", "description": "除数（不能为 0）"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════
#  第二步：Tool 执行器（实际跑工具代码）
# ═══════════════════════════════════════════════════════════

TOOL_REGISTRY: dict[str, callable] = {}


def register_tool(func):
    """装饰器：将函数注册到工具注册表"""
    TOOL_REGISTRY[func.__name__] = func
    return func


@register_tool
def add(a: float, b: float) -> dict[str, Any]:
    result = a + b
    return {"operation": f"{a} + {b}", "result": result}


@register_tool
def subtract(a: float, b: float) -> dict[str, Any]:
    result = a - b
    return {"operation": f"{a} - {b}", "result": result}


@register_tool
def multiply(a: float, b: float) -> dict[str, Any]:
    result = a * b
    return {"operation": f"{a} × {b}", "result": result}


@register_tool
def divide(a: float, b: float) -> dict[str, Any]:
    if b == 0:
        return {"operation": f"{a} ÷ {b}", "error": "除数不能为零"}
    result = a / b
    return {"operation": f"{a} ÷ {b}", "result": result}


def execute_tool(name: str, arguments: dict) -> str:
    """按名称查找并执行工具，返回 JSON 字符串"""
    func = TOOL_REGISTRY.get(name)
    if not func:
        return json.dumps({"error": f"未找到工具: {name}"}, ensure_ascii=False)
    try:
        result = func(**arguments)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════
#  第三步：Agent 单步 — 一次 LLM 完整推理
# ═══════════════════════════════════════════════════════════

def agent_step(user_input: str) -> list[dict]:
    """
    接收用户输入，执行完整的 Agent 单步：
      调 LLM → tool_calls → 执行工具 → 结果回传 → LLM 最终回答

    返回完整消息历史（含所有中间步骤），方便查看。
    """
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个计算助手，可以使用 add / subtract / multiply / divide "
                "工具完成四则运算。请按步骤推理，需要计算时就调用工具。"
                "如果表达式包含多步运算（如 (1+2)*3），请分步调用工具。"
            ),
        },
        {"role": "user", "content": user_input},
    ]

    step_count = 0

    while True:
        step_count += 1
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "stream": False,
        }

        resp = requests.post(URL, headers=HEADERS, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]

        # ── 打印当前步骤 ──
        if msg.get("tool_calls"):
            print(f"\n  🔧 [Agent 第 {step_count} 步] LLM 请求调用工具 ...")
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                print(f"     → {fn['name']}({fn['arguments']})")
        else:
            print(f"\n  💬 [Agent 第 {step_count} 步] LLM 返回最终回答")

        # ── 将 assistant 消息（含 tool_calls）加入历史 ──
        messages.append(msg)

        # ── 如果没有 tool_calls，这就是最终回答 ──
        if not msg.get("tool_calls"):
            return messages

        # ── 执行每个工具调用 ──
        for tc in msg["tool_calls"]:
            fn = tc["function"]
            tool_name = fn["name"]
            tool_args = json.loads(fn["arguments"])

            # 执行
            result_str = execute_tool(tool_name, tool_args)
            result_obj = json.loads(result_str)

            # 打印执行结果
            if "error" in result_obj:
                print(f"     ⚠️  {tool_name} 返回错误: {result_obj['error']}")
            else:
                print(f"     ✅ {tool_name} → {result_obj['result']}")

            # ── 将工具结果（tool role）加入历史 ──
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result_str,
            })

        # 继续循环 → LLM 拿到工具结果后决定下一步


# ═══════════════════════════════════════════════════════════
#  主交互循环
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("【Agent 工具调用演示】加减乘除四则运算")
    print("=" * 60)
    print("输入数学表达式，例如: (15 + 27) × 3 - 10 ÷ 2")
    print("输入 q 退出\n")

    while True:
        user_input = input(">>> ").strip()
        if not user_input:
            continue
        if user_input.lower() == "q":
            print("\n✓ 再见")
            break

        print(f"\n{'─' * 60}")
        print(f"用户: {user_input}")

        # ── 执行完整的 Agent 单步 ──
        messages = agent_step(user_input)

        # ── 提取并打印最终回答 ──
        final_msg = messages[-1]
        print(f"\n{'=' * 60}")
        print(f"最终回答: {final_msg['content']}")
        print(f"{'=' * 60}")

        # ── 打印消息历史概览（便于理解完整流程） ──
        print("\n📋 本轮消息历史（含中间步骤）：")
        for i, m in enumerate(messages):
            role = m["role"]
            if role == "system":
                continue  # 太长，跳过
            elif role == "user":
                print(f"  [{i}] user: {m['content'][:50]}...")
            elif role == "assistant":
                if m.get("tool_calls"):
                    calls = ", ".join(tc["function"]["name"] for tc in m["tool_calls"])
                    print(f"  [{i}] assistant → tool_calls: {calls}")
                else:
                    print(f"  [{i}] assistant: {m['content'][:60]}...")
            elif role == "tool":
                print(f"  [{i}] tool ({m['tool_call_id'][:8]}...): {m['content'][:60]}...")

        print(f"\n{'─' * 60}\n")


if __name__ == "__main__":
    main()
