+++
date = '2025-08-14T14:46:30+08:00'
title = 'Uv包管理'
description = 'python Uv包管理'
tags = ['python']
categories = ['python']
+++

UV 是一个用 Rust 编写的极速 Python 包管理器，作为 pip、pip-tools 和 virtualenv 的直接替代品。它比传统工具快 10-100 倍，同时提供更好的用户体验。

## 特性

- **极速性能**: 比 pip 快 10-100 倍
- **统一工具**: 集成了包安装、虚拟环境管理、依赖解析等功能
- **兼容性**: 与现有的 Python 生态系统完全兼容
- **可靠性**: 使用 Rust 编写，内存安全且性能卓越
- **跨平台**: 支持 Windows、macOS 和 Linux

## 安装 UV

### 使用安装脚本（推荐）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 使用 pip 安装

```bash
pip install uv
```

### 使用 Homebrew（macOS）

```bash
brew install uv
```

### 使用 Cargo（Rust）

```bash
cargo install uv
```

## 基本使用

### 创建和管理虚拟环境

```bash
# 创建虚拟环境
uv venv

# 指定 Python 版本创建虚拟环境
uv venv --python 3.11

# 指定虚拟环境名称和位置
uv venv myproject --python 3.11
```

### 激活虚拟环境

```bash
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 包管理

#### 安装包

```bash
# 安装单个包
uv add requests

# 安装多个包
uv add requests numpy pandas

# 从 requirements.txt 安装
uv add -r requirements.txt

# 安装开发依赖
uv add pytest --dev
```

#### 移除包

```bash
# 移除单个包
uv remove requests

# 移除多个包
uv remove requests numpy
```

#### 列出已安装包

```bash
uv pip list
```

#### 更新包

```bash
# 更新所有包
uv lock --upgrade

# 更新特定包
uv add requests --upgrade
```

## 项目管理

### 初始化项目

```bash
# 初始化新项目
uv init myproject
cd myproject

# 在现有目录初始化
uv init
```

### 项目配置文件

UV 使用 `pyproject.toml` 文件来管理项目配置：

```toml
[project]
name = "myproject"
version = "0.1.0"
description = "My awesome project"
authors = ["Your Name <your.email@example.com>"]
dependencies = [
    "requests>=2.25.0",
    "click>=8.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0.0",
    "black>=22.0.0",
    "flake8>=4.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 运行命令

```bash
# 在项目环境中运行 Python
uv run python script.py

# 运行模块
uv run -m pytest

# 运行可执行文件
uv run mycommand
```

## 依赖锁定

### 生成锁文件

```bash
# 生成 uv.lock 文件
uv lock
```

### 同步依赖

```bash
# 根据锁文件安装精确版本的依赖
uv sync

# 只安装生产依赖
uv sync --no-dev

# 安装额外的依赖组
uv sync --extra dev
```

## 工具链管理

### Python 版本管理

```bash
# 列出可用的 Python 版本
uv python list

# 安装特定 Python 版本
uv python install 3.11

# 设置项目 Python 版本
uv python pin 3.11
```

## 与传统工具的对比

| 功能 | pip + virtualenv | UV |
|------|-----------------|-----|
| 创建虚拟环境 | `python -m venv .venv` | `uv venv` |
| 激活环境 | `source .venv/bin/activate` | 同左 |
| 安装包 | `pip install requests` | `uv add requests` |
| 安装依赖文件 | `pip install -r requirements.txt` | `uv add -r requirements.txt` |
| 冻结依赖 | `pip freeze > requirements.txt` | `uv lock` |
| 速度 | 基准 | 10-100x 更快 |

## 迁移指南

### 从 pip 迁移

1. **安装 UV**

   ```bash
   pip install uv
   ```

2. **创建虚拟环境**

   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. **安装现有依赖**

   ```bash
   uv add -r requirements.txt
   ```

4. **生成新的锁文件**

   ```bash
   uv lock
   ```

### 从 Poetry 迁移

UV 可以直接读取 `pyproject.toml` 文件，使迁移变得简单：

```bash
# 初始化 UV 项目
uv init

# 安装依赖
uv sync
```

## 高级用法

### 自定义索引

```bash
# 使用私有 PyPI 索引
uv add --index-url https://private.pypi.org/simple/ mypackage

# 使用额外索引
uv add --extra-index-url https://test.pypi.org/simple/ mypackage
```

### 缓存管理

```bash
# 查看缓存信息
uv cache info

# 清理缓存
uv cache clean
```

### 全局工具安装

```bash
# 全局安装工具
uv tool install black

# 运行全局工具
uv tool run black --check .

# 列出已安装的全局工具
uv tool list
```

## 最佳实践

1. **使用锁文件**: 始终提交 `uv.lock` 文件到版本控制系统
2. **分离依赖**: 区分生产依赖和开发依赖
3. **固定 Python 版本**: 使用 `uv python pin` 确保团队使用相同的 Python 版本
4. **利用缓存**: UV 的全局缓存可以显著提升安装速度
5. **渐进迁移**: 可以在现有项目中逐步使用 UV 替代传统工具

## 常见问题

### Q: UV 与 pip 完全兼容吗？

A: 是的，UV 设计为 pip 的直接替代品，支持相同的包格式和索引。

### Q: 如何在 CI/CD 中使用 UV？

A: 可以使用 `uv sync --frozen` 确保安装与锁文件完全一致的依赖版本。

### Q: UV 支持私有包仓库吗？

A: 支持，可以使用 `--index-url` 和 `--extra-index-url` 参数配置私有仓库。

## 总结

UV 作为新一代的 Python 包管理器，以其卓越的性能和良好的用户体验正在改变 Python 生态系统。无论您是个人开发者还是企业团队，UV 都能为您的 Python 项目管理带来显著的效率提升。

立即开始使用 UV，体验极速的 Python 开发工作流程吧！
