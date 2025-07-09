+++
title = 'Ollama+Deepseek自动化任务'
description = '使用Ollama部署DeepSeek V2模型，构建自定义Agent，实现结构化输出和人工确认的自动化任务执行。'
tags = ['Ollama', 'DeepSeek', '自动化任务', '大语言模型']
categories = ['人工智能', '大语言模型']
+++

Ollama + DeepSeek V2 本地部署与自定义Agent自动化任务执行指南

## 概述

本文将详细介绍如何使用 Ollama 部署 DeepSeek V2 模型，并构建具备结构化输出和人工确认机制的自定义 Agent，实现安全可控的自动化任务执行。

## 目录

1. [环境准备](#环境准备)
2. [Ollama 安装与配置](#ollama-安装与配置)
3. [DeepSeek V2 模型部署](#deepseek-v2-模型部署)
4. [自定义Agent架构设计](#自定义agent架构设计)
5. [结构化响应实现](#结构化响应实现)
6. [人工确认机制](#人工确认机制)
7. [完整示例代码](#完整示例代码)
8. [最佳实践与安全考虑](#最佳实践与安全考虑)

## 环境准备

### 系统要求

- **操作系统**: Linux (推荐 Ubuntu 20.04+) / macOS / Windows
- **内存**: 至少 16GB RAM (推荐 32GB+)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (可选，显著提升性能)
- **存储**: 至少 50GB 可用空间
- **Python**: 3.8+

### 依赖安装

```bash
# 安装 Python 依赖
pip install requests pydantic typer rich ollama-python aiohttp

# 如果使用 GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Ollama 安装与配置

### 1. 安装 Ollama

```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# 或者使用包管理器
# Ubuntu/Debian
sudo apt install ollama

# macOS
brew install ollama

# Windows - 下载并安装 .exe 文件
# https://ollama.ai/download/windows
```

### 2. 启动 Ollama 服务

```bash
# 启动服务
ollama serve

# 验证安装
ollama --version
```

### 3. 配置环境变量

```bash
# ~/.bashrc 或 ~/.zshrc
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS_PATH=/usr/share/ollama/.ollama/models
```

## DeepSeek V2 模型部署

### 1. 拉取 DeepSeek V2 模型

```bash
# 拉取 DeepSeek Coder V2 模型
ollama pull deepseek-coder:6.7b

# 或者拉取其他版本
ollama pull deepseek-coder:33b
ollama pull deepseek-coder:1.3b

# 验证模型安装
ollama list
```

### 2. 测试模型运行

```bash
# 交互式测试
ollama run deepseek-coder:6.7b

# API 测试
curl http://localhost:11434/api/generate -d '{
  "model": "deepseek-coder:6.7b",
  "prompt": "写一个Python快速排序函数",
  "stream": false
}'
```

## 自定义Agent架构设计

### 核心组件架构

```python
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
import json
import asyncio
from datetime import datetime

class TaskType(str, Enum):
    """任务类型枚举"""
    CODE_GENERATION = "code_generation"
    FILE_OPERATION = "file_operation"
    SYSTEM_COMMAND = "system_command"
    DATA_ANALYSIS = "data_analysis"
    API_CALL = "api_call"

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"

class StructuredResponse(BaseModel):
    """结构化响应模型"""
    task_id: str = Field(description="任务唯一标识符")
    task_type: TaskType = Field(description="任务类型")
    title: str = Field(description="任务标题")
    description: str = Field(description="任务描述")
    estimated_duration: int = Field(description="预估执行时间(秒)")
    risk_level: str = Field(description="风险等级: low/medium/high")
    required_permissions: List[str] = Field(description="所需权限列表")
    execution_steps: List[str] = Field(description="执行步骤列表")
    code_blocks: Optional[List[Dict[str, str]]] = Field(default=None, description="代码块")
    files_to_modify: Optional[List[str]] = Field(default=None, description="需要修改的文件")
    backup_required: bool = Field(default=False, description="是否需要备份")
    rollback_plan: Optional[str] = Field(default=None, description="回滚计划")
    
class TaskExecutionPlan(BaseModel):
    """任务执行计划"""
    response: StructuredResponse
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
```

## 结构化响应实现

### 1. Ollama 客户端封装

```python
import ollama
from typing import AsyncGenerator

class OllamaClient:
    """Ollama 客户端封装"""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "deepseek-coder:6.7b"):
        self.host = host
        self.model = model
        self.client = ollama.Client(host=host)
    
    async def generate_structured_response(self, prompt: str) -> StructuredResponse:
        """生成结构化响应"""
        
        system_prompt = """
你是一个专业的任务分析和执行规划助手。请根据用户的请求，生成一个详细的结构化响应。

要求：
1. 仔细分析用户请求的类型和复杂度
2. 评估执行风险和所需权限
3. 制定详细的执行步骤
4. 如果涉及代码，提供完整的代码块
5. 考虑备份和回滚策略

请严格按照以下JSON格式输出：
{
    "task_id": "生成唯一ID",
    "task_type": "选择合适的任务类型",
    "title": "简洁的任务标题",
    "description": "详细的任务描述",
    "estimated_duration": 预估秒数,
    "risk_level": "low/medium/high",
    "required_permissions": ["权限列表"],
    "execution_steps": ["步骤1", "步骤2", "..."],
    "code_blocks": [{"language": "python", "code": "代码内容"}],
    "files_to_modify": ["文件路径列表"],
    "backup_required": true/false,
    "rollback_plan": "回滚策略描述"
}
"""
        
        full_prompt = f"{system_prompt}\n\n用户请求：{prompt}"
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 2048
                }
            )
            
            # 提取JSON内容
            response_text = response['response']
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_content = response_text[json_start:json_end]
                response_data = json.loads(json_content)
                return StructuredResponse(**response_data)
            else:
                raise ValueError("无法提取有效的JSON响应")
                
        except Exception as e:
            # 降级处理：创建基础响应
            import uuid
            return StructuredResponse(
                task_id=str(uuid.uuid4()),
                task_type=TaskType.CODE_GENERATION,
                title="处理用户请求",
                description=prompt,
                estimated_duration=60,
                risk_level="medium",
                required_permissions=["user_confirmation"],
                execution_steps=["分析请求", "生成响应", "等待确认"],
                backup_required=True,
                rollback_plan="如果执行失败，恢复到原始状态"
            )
```

### 2. 智能提示工程

```python
class PromptTemplate:
    """智能提示模板"""
    
    @staticmethod
    def create_analysis_prompt(user_request: str, context: Dict[str, Any] = None) -> str:
        """创建分析提示"""
        
        context_info = ""
        if context:
            context_info = f"""
当前上下文：
- 工作目录: {context.get('working_dir', 'unknown')}
- 可用工具: {', '.join(context.get('available_tools', []))}
- 系统信息: {context.get('system_info', 'unknown')}
"""
        
        return f"""
作为一个专业的AI助手，请分析以下用户请求并生成结构化的执行计划。

{context_info}

用户请求: {user_request}

分析要点：
1. 任务类型识别
2. 风险评估 (考虑数据安全、系统稳定性、不可逆操作等)
3. 权限需求分析
4. 执行步骤规划
5. 异常处理和回滚策略

请确保输出的JSON格式正确且完整。
"""

    @staticmethod
    def create_code_generation_prompt(task_description: str, language: str = "python") -> str:
        """创建代码生成提示"""
        
        return f"""
请为以下任务生成高质量的{language}代码：

任务描述: {task_description}

代码要求：
1. 遵循最佳实践和编码规范
2. 包含适当的错误处理
3. 添加详细的注释
4. 考虑边界情况
5. 确保代码安全性

请提供完整的可执行代码。
"""
```

## 人工确认机制

### 1. 交互式确认界面

```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
import typer

class HumanConfirmationInterface:
    """人工确认交互界面"""
    
    def __init__(self):
        self.console = Console()
    
    def display_task_summary(self, plan: TaskExecutionPlan) -> None:
        """显示任务摘要"""
        
        # 创建任务信息表格
        table = Table(title=f"任务执行计划 - {plan.response.task_id}")
        table.add_column("属性", style="cyan")
        table.add_column("值", style="white")
        
        table.add_row("任务类型", plan.response.task_type.value)
        table.add_row("标题", plan.response.title)
        table.add_row("风险等级", self._format_risk_level(plan.response.risk_level))
        table.add_row("预估时间", f"{plan.response.estimated_duration} 秒")
        table.add_row("需要备份", "是" if plan.response.backup_required else "否")
        
        self.console.print(table)
        
        # 显示描述
        self.console.print(Panel(
            plan.response.description,
            title="任务描述",
            border_style="blue"
        ))
        
        # 显示执行步骤
        if plan.response.execution_steps:
            steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan.response.execution_steps)])
            self.console.print(Panel(
                steps_text,
                title="执行步骤",
                border_style="green"
            ))
        
        # 显示所需权限
        if plan.response.required_permissions:
            perms_text = "\n".join([f"• {perm}" for perm in plan.response.required_permissions])
            self.console.print(Panel(
                perms_text,
                title="所需权限",
                border_style="yellow"
            ))
        
        # 显示代码块
        if plan.response.code_blocks:
            for i, code_block in enumerate(plan.response.code_blocks):
                syntax = Syntax(
                    code_block.get('code', ''),
                    code_block.get('language', 'text'),
                    theme="github-dark",
                    line_numbers=True
                )
                self.console.print(Panel(
                    syntax,
                    title=f"代码块 {i+1} ({code_block.get('language', 'text')})",
                    border_style="magenta"
                ))
        
        # 显示文件修改列表
        if plan.response.files_to_modify:
            files_text = "\n".join([f"• {file}" for file in plan.response.files_to_modify])
            self.console.print(Panel(
                files_text,
                title="将要修改的文件",
                border_style="red"
            ))
        
        # 显示回滚计划
        if plan.response.rollback_plan:
            self.console.print(Panel(
                plan.response.rollback_plan,
                title="回滚计划",
                border_style="orange3"
            ))
    
    def _format_risk_level(self, risk_level: str) -> str:
        """格式化风险等级显示"""
        colors = {
            "low": "[green]低风险[/green]",
            "medium": "[yellow]中等风险[/yellow]",
            "high": "[red]高风险[/red]"
        }
        return colors.get(risk_level.lower(), risk_level)
    
    def get_user_confirmation(self, plan: TaskExecutionPlan) -> bool:
        """获取用户确认"""
        
        self.display_task_summary(plan)
        
        # 风险警告
        if plan.response.risk_level == "high":
            self.console.print("\n[bold red]⚠️  警告：这是一个高风险操作！[/bold red]")
        
        self.console.print("\n")
        
        # 确认选项
        choice = Prompt.ask(
            "请选择操作",
            choices=["approve", "reject", "modify", "details"],
            default="details"
        )
        
        if choice == "approve":
            return True
        elif choice == "reject":
            return False
        elif choice == "details":
            self._show_detailed_info(plan)
            return self.get_user_confirmation(plan)  # 递归调用
        elif choice == "modify":
            self._handle_modification_request(plan)
            return self.get_user_confirmation(plan)  # 递归调用
    
    def _show_detailed_info(self, plan: TaskExecutionPlan) -> None:
        """显示详细信息"""
        
        self.console.print("[bold]详细技术信息：[/bold]")
        
        # 显示完整的响应数据
        response_dict = plan.response.dict()
        for key, value in response_dict.items():
            if value is not None:
                self.console.print(f"  {key}: {value}")
        
        input("\n按 Enter 键继续...")
    
    def _handle_modification_request(self, plan: TaskExecutionPlan) -> None:
        """处理修改请求"""
        
        self.console.print("[yellow]修改功能暂未实现，请选择批准或拒绝。[/yellow]")
        input("按 Enter 键继续...")
```

### 2. 安全检查机制

```python
class SecurityChecker:
    """安全检查器"""
    
    def __init__(self):
        self.high_risk_patterns = [
            r'rm\s+-rf',
            r'sudo\s+rm',
            r'dd\s+if=',
            r'format\s+c:',
            r'del\s+/s',
            r'DROP\s+DATABASE',
            r'DELETE\s+FROM.*WHERE\s+1=1',
            r'chmod\s+777',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
            r'getattr\s*\(',
            r'setattr\s*\(',
        ]
        
        self.sensitive_paths = [
            '/etc/passwd',
            '/etc/shadow',
            '/boot/',
            'C:\\Windows\\System32',
            '~/.ssh/',
            '/var/log/',
        ]
    
    def assess_risk_level(self, plan: TaskExecutionPlan) -> str:
        """评估风险等级"""
        
        risk_score = 0
        
        # 检查代码块中的危险模式
        if plan.response.code_blocks:
            for code_block in plan.response.code_blocks:
                code = code_block.get('code', '')
                for pattern in self.high_risk_patterns:
                    if re.search(pattern, code, re.IGNORECASE):
                        risk_score += 3
        
        # 检查文件路径
        if plan.response.files_to_modify:
            for file_path in plan.response.files_to_modify:
                for sensitive_path in self.sensitive_paths:
                    if sensitive_path in file_path:
                        risk_score += 2
        
        # 检查权限要求
        dangerous_permissions = ['root', 'admin', 'sudo', 'system']
        if plan.response.required_permissions:
            for perm in plan.response.required_permissions:
                if any(dangerous in perm.lower() for dangerous in dangerous_permissions):
                    risk_score += 2
        
        # 确定风险等级
        if risk_score >= 5:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"
    
    def validate_execution_safety(self, plan: TaskExecutionPlan) -> tuple[bool, List[str]]:
        """验证执行安全性"""
        
        warnings = []
        is_safe = True
        
        # 检查是否有备份计划
        if plan.response.backup_required and not plan.response.rollback_plan:
            warnings.append("需要备份但没有明确的回滚计划")
            is_safe = False
        
        # 检查文件操作安全性
        if plan.response.files_to_modify:
            for file_path in plan.response.files_to_modify:
                if not os.path.exists(file_path):
                    warnings.append(f"目标文件不存在: {file_path}")
                elif not os.access(file_path, os.W_OK):
                    warnings.append(f"没有写入权限: {file_path}")
        
        return is_safe, warnings
```

## 完整示例代码

### 主要的Agent类

```python
import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import subprocess
import os
import shutil

class AutomationAgent:
    """自动化任务执行Agent"""
    
    def __init__(self, model_name: str = "deepseek-coder:6.7b"):
        self.ollama_client = OllamaClient(model=model_name)
        self.confirmation_interface = HumanConfirmationInterface()
        self.security_checker = SecurityChecker()
        self.task_history: List[TaskExecutionPlan] = []
        
    async def process_request(self, user_request: str, context: Dict[str, Any] = None) -> TaskExecutionPlan:
        """处理用户请求"""
        
        try:
            # 1. 生成结构化响应
            print("🤖 正在分析您的请求...")
            structured_response = await self.ollama_client.generate_structured_response(user_request)
            
            # 2. 创建执行计划
            plan = TaskExecutionPlan(response=structured_response)
            
            # 3. 安全检查和风险评估
            plan.response.risk_level = self.security_checker.assess_risk_level(plan)
            is_safe, warnings = self.security_checker.validate_execution_safety(plan)
            
            if not is_safe:
                print("⚠️ 安全检查发现问题:")
                for warning in warnings:
                    print(f"  - {warning}")
            
            # 4. 人工确认
            print("\n📋 任务分析完成，请确认执行计划:")
            user_approved = self.confirmation_interface.get_user_confirmation(plan)
            
            if user_approved:
                plan.status = TaskStatus.APPROVED
                plan.approved_at = datetime.now()
                
                # 5. 执行任务
                execution_result = await self.execute_task(plan)
                plan.execution_result = execution_result
                plan.completed_at = datetime.now()
                
                if execution_result.get('success', False):
                    plan.status = TaskStatus.COMPLETED
                    print("✅ 任务执行成功!")
                else:
                    plan.status = TaskStatus.FAILED
                    plan.error_message = execution_result.get('error', 'Unknown error')
                    print(f"❌ 任务执行失败: {plan.error_message}")
            else:
                plan.status = TaskStatus.REJECTED
                print("❌ 任务被用户拒绝")
            
            # 6. 记录任务历史
            self.task_history.append(plan)
            
            return plan
            
        except Exception as e:
            print(f"❌ 处理请求时发生错误: {str(e)}")
            raise
    
    async def execute_task(self, plan: TaskExecutionPlan) -> Dict[str, Any]:
        """执行任务"""
        
        plan.status = TaskStatus.EXECUTING
        plan.executed_at = datetime.now()
        
        try:
            result = {"success": True, "outputs": []}
            
            # 创建备份
            if plan.response.backup_required:
                backup_result = self._create_backup(plan)
                result["backup"] = backup_result
            
            # 根据任务类型执行不同的操作
            if plan.response.task_type == TaskType.CODE_GENERATION:
                result.update(await self._execute_code_generation(plan))
            elif plan.response.task_type == TaskType.FILE_OPERATION:
                result.update(await self._execute_file_operation(plan))
            elif plan.response.task_type == TaskType.SYSTEM_COMMAND:
                result.update(await self._execute_system_command(plan))
            elif plan.response.task_type == TaskType.DATA_ANALYSIS:
                result.update(await self._execute_data_analysis(plan))
            elif plan.response.task_type == TaskType.API_CALL:
                result.update(await self._execute_api_call(plan))
            else:
                result["success"] = False
                result["error"] = f"不支持的任务类型: {plan.response.task_type}"
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _create_backup(self, plan: TaskExecutionPlan) -> Dict[str, Any]:
        """创建备份"""
        
        backup_info = {
            "timestamp": datetime.now().isoformat(),
            "files": []
        }
        
        if plan.response.files_to_modify:
            backup_dir = f"backup_{plan.response.task_id}"
            os.makedirs(backup_dir, exist_ok=True)
            
            for file_path in plan.response.files_to_modify:
                if os.path.exists(file_path):
                    backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, backup_path)
                    backup_info["files"].append({
                        "original": file_path,
                        "backup": backup_path
                    })
        
        return backup_info
    
    async def _execute_code_generation(self, plan: TaskExecutionPlan) -> Dict[str, Any]:
        """执行代码生成任务"""
        
        result = {"outputs": []}
        
        if plan.response.code_blocks:
            for i, code_block in enumerate(plan.response.code_blocks):
                language = code_block.get('language', 'python')
                code = code_block.get('code', '')
                
                # 保存代码到文件
                filename = f"generated_code_{i+1}.{self._get_file_extension(language)}"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                result["outputs"].append({
                    "type": "file_created",
                    "filename": filename,
                    "language": language,
                    "size": len(code)
                })
        
        return result
    
    async def _execute_file_operation(self, plan: TaskExecutionPlan) -> Dict[str, Any]:
        """执行文件操作任务"""
        
        result = {"outputs": []}
        
        # 这里实现具体的文件操作逻辑
        # 注意：实际实现时需要更严格的安全检查
        
        return result
    
    async def _execute_system_command(self, plan: TaskExecutionPlan) -> Dict[str, Any]:
        """执行系统命令任务"""
        
        result = {"outputs": []}
        
        # 警告：系统命令执行需要极其谨慎的安全控制
        # 这里只是示例，实际使用时应该有更严格的限制
        
        return result
    
    async def _execute_data_analysis(self, plan: TaskExecutionPlan) -> Dict[str, Any]:
        """执行数据分析任务"""
        
        result = {"outputs": []}
        
        # 实现数据分析逻辑
        
        return result
    
    async def _execute_api_call(self, plan: TaskExecutionPlan) -> Dict[str, Any]:
        """执行API调用任务"""
        
        result = {"outputs": []}
        
        # 实现API调用逻辑
        
        return result
    
    def _get_file_extension(self, language: str) -> str:
        """根据编程语言获取文件扩展名"""
        
        extensions = {
            'python': 'py',
            'javascript': 'js',
            'typescript': 'ts',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'go': 'go',
            'rust': 'rs',
            'shell': 'sh',
            'bash': 'sh',
            'sql': 'sql',
            'html': 'html',
            'css': 'css',
            'json': 'json',
            'yaml': 'yaml',
            'xml': 'xml'
        }
        
        return extensions.get(language.lower(), 'txt')
```

### 使用示例

```python
async def main():
    """主函数示例"""
    
    # 初始化Agent
    agent = AutomationAgent()
    
    # 测试用例
    test_requests = [
        "创建一个Python函数，用于计算斐波那契数列",
        "分析当前目录下的日志文件，统计错误数量",
        "写一个脚本备份数据库",
        "创建一个简单的Web服务器",
        "批量重命名文件夹中的图片文件"
    ]
    
    for request in test_requests:
        print(f"\n{'='*60}")
        print(f"处理请求: {request}")
        print('='*60)
        
        try:
            plan = await agent.process_request(request)
            print(f"任务状态: {plan.status.value}")
            
            if plan.execution_result:
                print("执行结果:")
                print(json.dumps(plan.execution_result, indent=2, ensure_ascii=False))
                
        except Exception as e:
            print(f"错误: {str(e)}")
        
        # 询问是否继续
        if not typer.confirm("继续下一个测试?"):
            break

if __name__ == "__main__":
    asyncio.run(main())
```

## 最佳实践与安全考虑

### 1. 安全原则

- **最小权限原则**: 只授予完成任务所需的最小权限
- **沙箱执行**: 在隔离环境中执行危险操作
- **备份策略**: 自动创建重要文件的备份
- **审计日志**: 记录所有操作和决策过程
- **用户确认**: 对高风险操作要求明确确认

### 2. 错误处理

```python
class RobustExecutor:
    """健壮的执行器"""
    
    async def safe_execute(self, operation: callable, *args, **kwargs):
        """安全执行操作"""
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                
                print(f"操作失败，正在重试 ({retry_count}/{max_retries}): {str(e)}")
                await asyncio.sleep(2 ** retry_count)  # 指数退避
```

### 3. 配置管理

```python
class AgentConfig:
    """Agent配置管理"""
    
    def __init__(self, config_file: str = "agent_config.json"):
        self.config_file = config_file
        self.load_config()
    
    def load_config(self):
        """加载配置"""
        
        default_config = {
            "ollama": {
                "host": "http://localhost:11434",
                "model": "deepseek-coder:6.7b",
                "timeout": 30
            },
            "security": {
                "max_file_size": 10485760,  # 10MB
                "allowed_extensions": [".txt", ".py", ".js", ".json", ".yaml"],
                "forbidden_paths": ["/etc", "/boot", "C:\\Windows\\System32"],
                "require_confirmation_for_high_risk": True
            },
            "execution": {
                "max_execution_time": 300,  # 5分钟
                "auto_backup": True,
                "backup_retention_days": 7
            },
            "logging": {
                "level": "INFO",
                "file": "agent.log",
                "max_size": 52428800,  # 50MB
                "backup_count": 5
            }
        }
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # 深度合并配置
                self.config = self._deep_merge(default_config, user_config)
        except FileNotFoundError:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """保存配置"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def _deep_merge(self, base: dict, update: dict) -> dict:
        """深度合并字典"""
        result = base.copy()
        for key, value in update.items():
            if isinstance(value, dict) and key in result:
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
```

### 4. 监控和日志

```python
import logging
from rich.logging import RichHandler

class AgentLogger:
    """Agent日志管理"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志"""
        
        logging.basicConfig(
            level=getattr(logging, self.config.config['logging']['level']),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                RichHandler(rich_tracebacks=True),
                logging.FileHandler(
                    self.config.config['logging']['file'],
                    encoding='utf-8'
                )
            ]
        )
        
        self.logger = logging.getLogger("AutomationAgent")
    
    def log_task_start(self, task_id: str, request: str):
        """记录任务开始"""
        self.logger.info(f"任务开始 - ID: {task_id}, 请求: {request}")
    
    def log_task_completion(self, task_id: str, status: str, duration: float):
        """记录任务完成"""
        self.logger.info(f"任务完成 - ID: {task_id}, 状态: {status}, 耗时: {duration:.2f}秒")
    
    def log_security_event(self, event_type: str, details: dict):
        """记录安全事件"""
        self.logger.warning(f"安全事件 - 类型: {event_type}, 详情: {details}")
```

## 总结

本文介绍了一个完整的基于 Ollama + DeepSeek V2 的自动化任务执行系统，具有以下特点：

1. **结构化响应**: 使用 Pydantic 模型确保输出格式一致性
2. **人工确认**: 提供友好的交互界面进行任务确认
3. **安全控制**: 多层安全检查和风险评估
4. **可扩展性**: 模块化设计便于功能扩展
5. **错误处理**: 完善的异常处理和恢复机制

这个系统可以作为构建更复杂AI自动化工具的基础框架，通过适当的配置和扩展，可以适应各种自动化场景的需求。

### 下一步改进方向

- 增加更多任务类型支持
- 实现分布式执行能力
- 添加任务调度和队列管理
- 集成更多外部工具和API
- 增强安全沙箱机制
- 实现任务执行的可视化监控
