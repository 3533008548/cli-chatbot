"""
异步 Agent 示例

同时调用 3 个工具（查天气 / 查数据库 / 查文件），
用 httpx 异步调用 DeepSeek LLM 汇总结果。
"""

import asyncio
import json
import sqlite3
import os
from pathlib import Path
from datetime import datetime

import httpx

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
    raise RuntimeError("未找到 DEEPSEEK_API_KEY，请将 .env.example 复制为 .env 并填入 Key")


# ═══════════════════════════════════════════════════════════
#  工具 1：查天气 — 调用 wttr.in API
# ═══════════════════════════════════════════════════════════

async def tool_weather(city: str = "Beijing") -> dict:
    """获取指定城市的当前天气（异步 HTTP）"""
    url = f"https://wttr.in/{city}?format=%C|%t|%h|%w"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        parts = resp.text.strip().split("|")
        return {
            "tool": "weather",
            "city": city,
            "condition": parts[0] if len(parts) > 0 else "N/A",
            "temperature": parts[1] if len(parts) > 1 else "N/A",
            "humidity": parts[2] if len(parts) > 2 else "N/A",
            "wind": parts[3] if len(parts) > 3 else "N/A",
        }


# ═══════════════════════════════════════════════════════════
#  工具 2：查数据库 — 内存中创建 SQLite 示例表
# ═══════════════════════════════════════════════════════════

def _init_db() -> str:
    """（同步函数，被 asyncio.to_thread 包装）创建示例 DB 并返回 JSON"""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary INTEGER
        );
        INSERT INTO employees VALUES (1, '张三', '技术部', 18000);
        INSERT INTO employees VALUES (2, '李四', '市场部', 15000);
        INSERT INTO employees VALUES (3, '王五', '技术部', 22000);
        INSERT INTO employees VALUES (4, '赵六', '人事部', 14000);
        INSERT INTO employees VALUES (5, '钱七', '技术部', 25000);

        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL,
            stock INTEGER
        );
        INSERT INTO products VALUES (1, '笔记本电脑', 6999.0, 45);
        INSERT INTO products VALUES (2, '机械键盘',   599.0, 120);
        INSERT INTO products VALUES (3, '27寸显示器', 2499.0, 30);
    """)

    rows = cur.execute("SELECT * FROM employees ORDER BY salary DESC").fetchall()
    result = [dict(r) for r in rows]
    conn.close()
    return json.dumps(result, ensure_ascii=False, indent=2)


async def tool_database() -> dict:
    """查询示例数据库，返回员工信息"""
    raw = await asyncio.to_thread(_init_db)
    return {
        "tool": "database",
        "description": "公司员工表（按薪资降序）",
        "data": json.loads(raw),
    }


# ═══════════════════════════════════════════════════════════
#  工具 3：查文件 — 读取本地 Markdown 日报
# ═══════════════════════════════════════════════════════════

SAMPLE_FILE = Path(__file__).parent / "data" / "daily_report.md"


def _ensure_sample_file():
    """若文件不存在则自动创建"""
    SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not SAMPLE_FILE.exists():
        content = f"""# 日报 — {datetime.now().strftime('%Y-%m-%d')}

## 今日完成
1. 完成了用户登录模块的单元测试，覆盖率 92%
2. 修复了 Issue #147：搜索框在移动端不响应的问题
3. 与设计团队确认了新首页的 UI 方案

## 遇到的问题
- 数据库连接池在高峰期出现过一次超时，已调整 max_connections

## 明日计划
1. 开始开发消息推送模块
2. 代码审查 PR #152、#153
"""
        SAMPLE_FILE.write_text(content, encoding="utf-8")


async def tool_file() -> dict:
    """读取本地日报文件（异步）"""
    await asyncio.to_thread(_ensure_sample_file)
    content = await asyncio.to_thread(lambda: SAMPLE_FILE.read_text(encoding="utf-8"))
    return {
        "tool": "file",
        "file": str(SAMPLE_FILE.relative_to(Path.cwd())),
        "content": content,
    }


# ═══════════════════════════════════════════════════════════
#  Agent 主流程：并行调 3 个工具 → httpx 调 LLM → 输出
# ═══════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("【异步 Agent】并行调用 3 个工具中 ...")
    print("=" * 60)

    # ── 同时启动 3 个异步工具 ──
    weather_task = tool_weather()
    db_task = tool_database()
    file_task = tool_file()

    weather_result, db_result, file_result = await asyncio.gather(
        weather_task, db_task, file_task
    )

    print("✓ 3 个工具已全部返回，正在请求 LLM 汇总 ...\n")

    # ── 构造 LLM 调用的消息 ──
    system_prompt = "你是一个数据分析助手。用户同时调用了 3 个工具获取信息，请根据返回结果写一段清晰的中文汇报。"

    user_message = f"""请根据以下 3 个工具的返回结果，整合成一份清晰的中文汇报。

## 🌤️ 天气信息
| 字段 | 值 |
|------|-----|
| 城市 | {weather_result['city']} |
| 天气 | {weather_result['condition']} |
| 温度 | {weather_result['temperature']} |
| 湿度 | {weather_result['humidity']} |
| 风速 | {weather_result['wind']} |

## 🗄️ 数据库查询结果
查询内容：{db_result['description']}
数据：
```json
{json.dumps(db_result['data'], ensure_ascii=False, indent=2)}
```

## 📄 文件读取结果
文件路径：{file_result['file']}
内容：
```markdown
{file_result['content']}
```
"""

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    # ── 打印结果 ──
    print("=" * 60)
    print("【LLM 汇总结果】")
    print("=" * 60)
    print(data["choices"][0]["message"]["content"])

    usage = data["usage"]
    print(f"\nToken 用量: prompt={usage['prompt_tokens']}, "
          f"completion={usage['completion_tokens']}, "
          f"total={usage['total_tokens']}")
    print("\n✓ 全部完成")


if __name__ == "__main__":
    asyncio.run(main())
