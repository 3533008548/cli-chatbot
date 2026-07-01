"""
agent_tools_demo.py — Agent 工具调用演示（封装为类）

核心流程（完整 Agent 单步，封装在 Agent 类中）：
  1. 用户提问
  2. LLM 返回 tool_calls（决定调哪个工具 + 参数）
  3. 把 tool_calls 加入消息历史
  4. 执行工具 → 得到结果
  5. 把工具结果（tool role）加入消息历史
  6. 再次调 LLM → 模型把工具结果组织成自然语言回答
  7. 重复 2-6，直到 LLM 不再请求工具
"""

import os
import json
import math
import time
from pathlib import Path
from typing import Any, Callable
from datetime import datetime

import requests


# ═══════════════════════════════════════════════════════════
#  Agent 类
# ═══════════════════════════════════════════════════════════

class Agent:
    """可注册工具、可对话的 Agent 封装"""

    def __init__(
        self,
        system_prompt: str = "你是一个智能助手，可以使用注册的工具来帮助用户。",
        model: str = "deepseek-chat",
        tool_choice: str = "auto",
        api_key: str | None = None,
        api_url: str = "https://api.deepseek.com/chat/completions",
    ):
        """初始化 Agent

        Args:
            system_prompt: 系统提示词
            model: 模型名称
            tool_choice: "auto" / "required" / "none"
            api_key: API Key（默认从 .env 读取）
            api_url: API 端点
        """
        # ── API 配置 ──
        self.api_key = api_key or self._load_api_key()
        self.api_url = api_url
        self.model = model
        self.tool_choice = tool_choice
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # ── 工具注册表 ──
        self._tool_registry: dict[str, Callable] = {}
        self._tool_schemas: list[dict] = []

        # ── 对话状态 ──
        self._messages: list[dict] = []
        self._system_prompt = system_prompt

    # ── 静态工具：加载 API Key ──

    @staticmethod
    def _load_api_key() -> str:
        """从 .env 或环境变量读取 API Key"""
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

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("未找到 DEEPSEEK_API_KEY，请检查 .env 文件")
        return api_key

    # ── 工具注册 ──

    def tool(self, name: str, description: str, parameters: dict):
        """装饰器：将函数注册为一个工具

        Args:
            name: 工具名称（唯一标识）
            description: 工具描述（LLM 据此判断何时调用）
            parameters: JSON Schema 格式的参数描述

        用法:
            @agent.tool(name="add", description="加法", parameters={...})
            def add(a: float, b: float) -> dict:
                ...
        """
        def decorator(func: Callable) -> Callable:
            self._tool_registry[name] = func
            self._tool_schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            })
            return func
        return decorator

    # ── 工具执行 ──

    def _execute_tool(self, name: str, arguments: dict) -> str:
        """按名称查找并执行工具，返回 JSON 字符串"""
        func = self._tool_registry.get(name)
        if not func:
            return json.dumps({"error": f"未找到工具: {name}"}, ensure_ascii=False)
        try:
            result = func(**arguments)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    # ── 对话管理 ──

    def reset(self):
        """清空消息历史，保留 system prompt"""
        self._messages = [{"role": "system", "content": self._system_prompt}]

    def add_message(self, role: str, content: str):
        """手动追加一条消息到历史"""
        self._messages.append({"role": role, "content": content})

    @property
    def history(self) -> list[dict]:
        """获取当前消息历史（只读副本）"""
        return list(self._messages)

    # ── Agent 单步：一次用户输入 → LLM 完整推理 ──

    def step(self, user_input: str, verbose: bool = True) -> str:
        """执行一次完整的 Agent 推理

        1. 添加用户消息
        2. 循环：调 LLM → 处理 tool_calls → 执行工具 → 回传结果
        3. 返回最终回答文本

        Args:
            user_input: 用户输入
            verbose: 是否打印中间步骤

        Returns:
            final_answer: LLM 最终回答文本
        """
        self._messages.append({"role": "user", "content": user_input})

        step_count = 0

        while True:
            step_count += 1
            payload = {
                "model": self.model,
                "messages": self._messages,
                "tools": self._tool_schemas,
                "tool_choice": self.tool_choice,
                "stream": False,
            }

            resp = requests.post(
                self.api_url, headers=self.headers, json=payload, timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]

            # ── 打印当前步骤 ──
            if verbose:
                if msg.get("tool_calls"):
                    print(f"\n  🔧 [步骤 {step_count}] LLM 请求调用工具 ...")
                    for tc in msg["tool_calls"]:
                        fn = tc["function"]
                        print(f"     → {fn['name']}({fn['arguments']})")
                    print()
                else:
                    print(f"\n  💬 [步骤 {step_count}] LLM 返回最终回答")

            # ── 将 assistant 消息（含 tool_calls）加入历史 ──
            self._messages.append(msg)

            # ── 如果没有 tool_calls，返回最终回答 ──
            if not msg.get("tool_calls"):
                return msg["content"]

            # ── 执行每个工具调用 ──
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                tool_name = fn["name"]
                tool_args = json.loads(fn["arguments"])

                result_str = self._execute_tool(tool_name, tool_args)
                result_obj = json.loads(result_str)

                if verbose:
                    if "error" in result_obj:
                        print(f"     ⚠️  {tool_name} 错误: {result_obj['error']}")
                    else:
                        display = (
                            result_obj.get("result")
                            or result_obj.get("value")
                            or result_obj
                        )
                        print(f"     ✅ {tool_name} → {display}")

                # ── 将工具结果（tool role）加入历史 ──
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

    # ── 快捷对话（含打印） ──

    def chat(self, user_input: str) -> str:
        """对话：执行 step + 打印结果 + 返回回答文本"""
        print(f"\n{'─' * 60}")
        print(f"用户: {user_input}")

        answer = self.step(user_input)

        print(f"\n{'=' * 60}")
        print(f"最终回答: {answer}")
        print(f"{'=' * 60}")
        return answer


# ═══════════════════════════════════════════════════════════
#  使用演示
# ═══════════════════════════════════════════════════════════

def main():
    # ── 创建 Agent 实例 ──
    agent = Agent(
        system_prompt=(
            "你是一个智能助手，拥有以下工具：\n"
            "1. calculator — 四则运算（add/subtract/multiply/divide）\n"
            "2. scientific_calc — 科学计算（sqrt/power/sin/cos/tan/log/log10/factorial/abs/round）\n"
            "3. get_current_time — 获取当前时间\n"
            "请按步骤推理，需要计算或获取信息时就调用工具。"
            "如果表达式包含多步运算（如 (1+2)*3），请分步调用工具。"
        ),
        tool_choice="auto",
    )

    # ── 注册工具 1：四则运算 ──

    @agent.tool(
        name="calculator",
        description="四则运算计算器，支持加、减、乘、除",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "第一个数"},
                "b": {"type": "number", "description": "第二个数"},
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "运算类型：add 加法, subtract 减法, multiply 乘法, divide 除法",
                },
            },
            "required": ["a", "b", "operation"],
        },
    )
    def calculator(a: float, b: float, operation: str) -> dict[str, Any]:
        op_map = {
            "add": ("+", lambda: a + b),
            "subtract": ("-", lambda: a - b),
            "multiply": ("×", lambda: a * b),
            "divide": ("/", lambda: a / b if b != 0 else None),
        }
        if operation not in op_map:
            return {"error": f"不支持的运算: {operation}"}
        if operation == "divide" and b == 0:
            return {"operation": f"{a} ÷ {b}", "error": "除数不能为零"}
        sym, fn = op_map[operation]
        return {"operation": f"{a} {sym} {b}", "result": fn()}

    # ── 注册工具 2：科学计算 ──

    @agent.tool(
        name="scientific_calc",
        description=(
            "科学计算器，支持 sqrt（开平方）、power（幂运算）、"
            "sin / cos / tan（三角函数）、log（自然对数）、log10（常用对数）、"
            "factorial（阶乘）、abs（绝对值）、round（四舍五入取整）"
        ),
        parameters={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "sqrt", "power", "sin", "cos", "tan",
                        "log", "log10", "factorial", "abs", "round",
                    ],
                    "description": "科学运算类型",
                },
                "x": {"type": "number", "description": "输入值"},
                "y": {
                    "type": "number",
                    "description": "第二个输入值（仅 power 需要，其余可省略）",
                },
            },
            "required": ["operation", "x"],
        },
    )
    def scientific_calc(operation: str, x: float, y: float = 0) -> dict[str, Any]:
        try:
            ops = {
                "sqrt": lambda: math.sqrt(x) if x >= 0 else None,
                "power": lambda: x ** y,
                "sin": lambda: math.sin(x),
                "cos": lambda: math.cos(x),
                "tan": lambda: math.tan(x),
                "log": lambda: math.log(x) if x > 0 else None,
                "log10": lambda: math.log10(x) if x > 0 else None,
                "factorial": lambda: math.factorial(int(x))
                    if x >= 0 and x == int(x) else None,
                "abs": lambda: abs(x),
                "round": lambda: round(x),
            }
            if operation not in ops:
                return {"error": f"不支持的运算: {operation}"}
            result = ops[operation]()
            if result is None:
                return {"error": f"参数不合法: {operation}({x})"}
            return {"operation": f"{operation}({x})", "result": result}
        except Exception as e:
            return {"error": str(e)}

    # ── 注册工具 3：获取当前时间 ──

    @agent.tool(
        name="get_current_time",
        description="获取当前的日期和时间",
        parameters={
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["date", "time", "datetime", "iso"],
                    "description": "返回格式：date=仅日期, time=仅时间, datetime=日期+时间, iso=ISO 8601",
                },
            },
            "required": [],
        },
    )
    def get_current_time(format: str = "datetime") -> dict[str, Any]:
        now = datetime.now()
        fmt_map = {
            "date": "%Y-%m-%d",
            "time": "%H:%M:%S",
            "datetime": "%Y-%m-%d %H:%M:%S",
        }
        if format == "iso":
            return {"format": "ISO 8601", "value": now.isoformat()}
        fmt = fmt_map.get(format, "%Y-%m-%d %H:%M:%S")
        return {"format": format, "value": now.strftime(fmt)}

    # ── 交互循环 ──
    print("=" * 60)
    print("【Agent 工具调用演示】类封装版")
    print("=" * 60)
    print("例如: (15 + 27) × 3 - 10 ÷ 2")
    print("      sqrt(144) + 现在几点")
    print("输入 q 退出\n")

    while True:
        user_input = input(">>> ").strip()
        if not user_input:
            continue
        if user_input.lower() == "q":
            print("\n✓ 再见")
            break

        agent.chat(user_input)

        # 可选：打印当前消息历史概览
        print("\n📋 当前消息历史：")
        for i, m in enumerate(agent.history):
            if m["role"] == "system":
                continue
            if m["role"] == "user":
                print(f"  [{i}] user: {m['content'][:50]}...")
            elif m["role"] == "assistant":
                if m.get("tool_calls"):
                    calls = ", ".join(tc["function"]["name"] for tc in m["tool_calls"])
                    print(f"  [{i}] assistant → {calls}")
                else:
                    print(f"  [{i}] assistant: {m['content'][:60]}...")
            elif m["role"] == "tool":
                print(f"  [{i}] tool: {m['content'][:60]}...")
        print()


if __name__ == "__main__":
    main()
