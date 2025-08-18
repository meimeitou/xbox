+++
date = '2025-08-14T15:07:58+08:00'
title = 'MCP Python SDK'
description = 'MCP Python SDK 使用指南'
tags = ['python', 'mcp']
categories = ['python', 'mcp']
+++

模型上下文协议（Model Context Protocol，MCP）Python SDK 是用于构建 MCP 服务器和客户端的官方 Python 实现。本文将详细介绍如何使用 MCP Python SDK 来创建高效、可扩展的 AI 应用程序。

## 什么是 MCP？

模型上下文协议（MCP）允许应用程序以标准化方式为大语言模型（LLM）提供上下文，将提供上下文与实际 LLM 交互的关注点分离。MCP 服务器可以：

- **暴露数据**：通过资源（Resources）提供数据，类似于 GET 端点
- **提供功能**：通过工具（Tools）执行代码或产生副作用，类似于 POST 端点
- **定义交互模式**：通过提示（Prompts）提供可重用的 LLM 交互模板

## 安装

### 使用 UV（推荐）

```bash
# 创建新项目
uv init mcp-server-demo
cd mcp-server-demo

# 添加 MCP 依赖
uv add "mcp[cli]"
```

### 运行 MCP 开发工具

```bash
uv run mcp
```

## 快速开始

让我们创建一个简单的 MCP 服务器，展示计算器工具和动态数据：

```python
"""
FastMCP 快速开始示例
"""

from mcp.server.fastmcp import FastMCP

# 创建 MCP 服务器
mcp = FastMCP("Demo")

# 添加加法工具
@mcp.tool()
def add(a: int, b: int) -> int:
    """两数相加"""
    return a + b

# 添加动态问候资源
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """获取个性化问候语"""
    return f"Hello, {name}!"

# 添加提示模板
@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """生成问候提示"""
    styles = {
        "friendly": "请写一个热情友好的问候语",
        "formal": "请写一个正式专业的问候语", 
        "casual": "请写一个随意轻松的问候语",
    }
    
    return f"{styles.get(style, styles['friendly'])}给一个名叫 {name} 的人。"

if __name__ == "__main__":
    mcp.run()
```

## 核心概念

### 服务器生命周期管理

FastMCP 服务器支持生命周期管理，允许在启动时初始化资源：

```python
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import Context, FastMCP

# 模拟数据库类
class Database:
    @classmethod
    async def connect(cls) -> "Database":
        return cls()
    
    async def disconnect(self) -> None:
        pass
    
    def query(self) -> str:
        return "查询结果"

@dataclass
class AppContext:
    db: Database

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    # 启动时初始化
    db = await Database.connect()
    try:
        yield AppContext(db=db)
    finally:
        # 关闭时清理
        await db.disconnect()

# 传递生命周期到服务器
mcp = FastMCP("My App", lifespan=app_lifespan)

# 在工具中访问生命周期上下文
@mcp.tool()
def query_db(ctx: Context) -> str:
    """使用初始化的资源的工具"""
    db = ctx.request_context.lifespan_context.db
    return db.query()
```

### 资源（Resources）

资源用于向 LLM 暴露数据，类似于 REST API 中的 GET 端点：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="Resource Example")

@mcp.resource("file://documents/{name}")
def read_document(name: str) -> str:
    """根据名称读取文档"""
    return f"{name} 的内容"

@mcp.resource("config://settings")
def get_settings() -> str:
    """获取应用设置"""
    return """{
  "theme": "dark",
  "language": "zh-CN",
  "debug": false
}"""
```

### 工具（Tools）

工具让 LLM 能够通过服务器执行操作：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="Tool Example")

@mcp.tool()
def sum_numbers(a: int, b: int) -> int:
    """计算两数之和"""
    return a + b

@mcp.tool()
def get_weather(city: str, unit: str = "celsius") -> str:
    """获取城市天气"""
    return f"{city} 的天气：22°{unit[0].upper()}"
```

### 结构化输出

工具支持结构化输出，使用类型注解自动验证：

```python
from typing import TypedDict
from pydantic import BaseModel, Field

# 使用 Pydantic 模型
class WeatherData(BaseModel):
    temperature: float = Field(description="摄氏度温度")
    humidity: float = Field(description="湿度百分比")
    condition: str
    wind_speed: float

@mcp.tool()
def get_weather_structured(city: str) -> WeatherData:
    """获取结构化天气数据"""
    return WeatherData(
        temperature=22.5,
        humidity=45.0,
        condition="晴朗",
        wind_speed=5.2,
    )

# 使用 TypedDict
class LocationInfo(TypedDict):
    latitude: float
    longitude: float
    name: str

@mcp.tool()
def get_location(address: str) -> LocationInfo:
    """获取位置坐标"""
    return LocationInfo(
        latitude=39.9042,
        longitude=116.4074,
        name="北京, 中国"
    )
```

### 上下文（Context）

工具和资源函数可以接收上下文对象，提供 MCP 功能访问：

```python
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

mcp = FastMCP(name="Context Example")

@mcp.tool()
async def long_running_task(
    task_name: str, 
    ctx: Context[ServerSession, None], 
    steps: int = 5
) -> str:
    """执行带进度更新的任务"""
    await ctx.info(f"开始: {task_name}")

    for i in range(steps):
        progress = (i + 1) / steps
        await ctx.report_progress(
            progress=progress,
            total=1.0,
            message=f"步骤 {i + 1}/{steps}",
        )
        await ctx.debug(f"完成步骤 {i + 1}")

    return f"任务 '{task_name}' 完成"
```

### 提示（Prompts）

提示是可重用的模板，帮助 LLM 有效与服务器交互：

```python
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

mcp = FastMCP(name="Prompt Example")

@mcp.prompt(title="代码审查")
def review_code(code: str) -> str:
    return f"请审查这段代码：\n\n{code}"

@mcp.prompt(title="调试助手")
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("我遇到了这个错误："),
        base.UserMessage(error),
        base.AssistantMessage("我来帮你调试。你已经尝试了什么？"),
    ]
```

### 图像处理

FastMCP 提供 `Image` 类自动处理图像数据：

```python
from PIL import Image as PILImage
from mcp.server.fastmcp import FastMCP, Image

mcp = FastMCP("Image Example")

@mcp.tool()
def create_thumbnail(image_path: str) -> Image:
    """从图像创建缩略图"""
    img = PILImage.open(image_path)
    img.thumbnail((100, 100))
    return Image(data=img.tobytes(), format="png")
```

## 运行服务器

### 开发模式

使用 MCP Inspector 快速测试和调试：

```bash
uv run mcp dev server.py

# 添加依赖
uv run mcp dev server.py --with pandas --with numpy

# 挂载本地代码
uv run mcp dev server.py --with-editable .
```

### Claude Desktop 集成

安装到 Claude Desktop：

```bash
uv run mcp install server.py

# 自定义名称
uv run mcp install server.py --name "我的分析服务器"

# 环境变量
uv run mcp install server.py -v API_KEY=abc123 -v DB_URL=postgres://...
```

### 直接执行

用于自定义部署场景：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")

@mcp.tool()
def hello(name: str = "World") -> str:
    """向某人问好"""
    return f"Hello, {name}!"

def main():
    mcp.run()

if __name__ == "__main__":
    main()
```

### Streamable HTTP 传输

适用于生产部署：

```python
from mcp.server.fastmcp import FastMCP

# 有状态服务器（维护会话状态）
mcp = FastMCP("StatefulServer")

# 无状态服务器（无会话持久化）
# mcp = FastMCP("StatelessServer", stateless_http=True)

@mcp.tool()
def greet(name: str = "World") -> str:
    """按名称问候某人"""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

## 高级用法

### 用户交互（Elicitation）

请求用户提供额外信息：

```python
from pydantic import BaseModel, Field
from mcp.server.fastmcp import Context, FastMCP

class BookingPreferences(BaseModel):
    checkAlternative: bool = Field(description="是否想检查其他日期？")
    alternativeDate: str = Field(
        default="2024-12-26",
        description="替代日期 (YYYY-MM-DD)",
    )

@mcp.tool()
async def book_table(
    date: str, 
    time: str, 
    party_size: int, 
    ctx: Context
) -> str:
    """预订餐桌并检查日期可用性"""
    if date == "2024-12-25":
        result = await ctx.elicit(
            message=f"{date} 没有 {party_size} 人的空桌。想试试其他日期吗？",
            schema=BookingPreferences,
        )
        
        if result.action == "accept" and result.data:
            if result.data.checkAlternative:
                return f"[成功] 已预订 {result.data.alternativeDate}"
            return "[取消] 未进行预订"
        return "[取消] 预订已取消"
    
    return f"[成功] 已预订 {date} {time}"
```

### 采样（与 LLM 交互）

工具可以通过采样与 LLM 交互：

```python
from mcp.types import SamplingMessage, TextContent

@mcp.tool()
async def generate_poem(topic: str, ctx: Context) -> str:
    """使用 LLM 采样生成诗歌"""
    prompt = f"写一首关于 {topic} 的短诗"

    result = await ctx.session.create_message(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=prompt),
            )
        ],
        max_tokens=100,
    )

    if result.content.type == "text":
        return result.content.text
    return str(result.content)
```

### OAuth 认证

对于需要访问受保护资源的服务器：

```python
from pydantic import AnyHttpUrl
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP

class SimpleTokenVerifier(TokenVerifier):
    async def verify_token(self, token: str) -> AccessToken | None:
        # 在此实现实际的令牌验证
        pass

# 创建具有认证的 FastMCP 实例
mcp = FastMCP(
    "Weather Service",
    token_verifier=SimpleTokenVerifier(),
    auth=AuthSettings(
        issuer_url=AnyHttpUrl("https://auth.example.com"),
        resource_server_url=AnyHttpUrl("http://localhost:3001"),
        required_scopes=["user"],
    ),
)

@mcp.tool()
async def get_weather(city: str = "北京") -> dict[str, str]:
    """获取城市天气数据"""
    return {
        "city": city,
        "temperature": "22",
        "condition": "多云转晴",
        "humidity": "65%",
    }
```

## MCP 客户端

SDK 还提供了用于连接 MCP 服务器的高级客户端接口：

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 创建服务器参数
server_params = StdioServerParameters(
    command="uv",
    args=["run", "server", "fastmcp_quickstart", "stdio"],
)

async def run_client():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化连接
            await session.initialize()

            # 列出可用工具
            tools = await session.list_tools()
            print(f"可用工具: {[t.name for t in tools.tools]}")

            # 调用工具
            result = await session.call_tool("add", arguments={"a": 5, "b": 3})
            print(f"工具结果: {result.content[0].text}")

            # 读取资源
            resource = await session.read_resource("greeting://World")
            print(f"资源内容: {resource.contents[0].text}")

if __name__ == "__main__":
    asyncio.run(run_client())
```

## 最佳实践

1. **使用结构化输出**：为工具定义明确的返回类型，提供更好的类型安全
2. **实现生命周期管理**：在服务器启动时初始化资源，在关闭时清理
3. **提供丰富的上下文**：使用 Context 对象进行日志记录、进度报告和用户交互
4. **合理组织资源和工具**：将相关功能分组，使用清晰的命名约定
5. **错误处理**：实现适当的错误处理和用户友好的错误消息
6. **性能优化**：对于长时间运行的操作，使用进度报告和异步处理
