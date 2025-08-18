+++
date = '2025-08-14T16:33:10+08:00'
title = 'Python异步编程完整指南'
description = '异步编程是一种并发编程范式，允许程序在等待某些操作完成时继续执行其他任务。本文详细介绍Python中的异步编程概念、语法和实践。'
tags = ['python', 'async', 'asyncio', '并发编程']
categories = ['python']
+++

- [什么是异步编程？](#什么是异步编程)
  - [同步 vs 异步](#同步-vs-异步)
- [Python异步编程基础](#python异步编程基础)
  - [1. 核心概念](#1-核心概念)
    - [协程（Coroutine）](#协程coroutine)
    - [事件循环（Event Loop）](#事件循环event-loop)
    - [await 关键字](#await-关键字)
  - [2. 异步任务管理](#2-异步任务管理)
    - [创建和运行任务](#创建和运行任务)
    - [并发执行多个任务](#并发执行多个任务)
  - [3. 异步上下文管理器](#3-异步上下文管理器)
  - [4. 异步迭代器](#4-异步迭代器)
- [实际应用示例](#实际应用示例)
  - [1. 异步HTTP请求](#1-异步http请求)
  - [2. 异步文件操作](#2-异步文件操作)
  - [3. 异步队列](#3-异步队列)
- [错误处理和超时](#错误处理和超时)
  - [1. 异步异常处理](#1-异步异常处理)
  - [2. 超时处理](#2-超时处理)
- [性能优化建议](#性能优化建议)
  - [1. 合理使用并发](#1-合理使用并发)
  - [2. 避免阻塞操作](#2-避免阻塞操作)
- [常见陷阱和最佳实践](#常见陷阱和最佳实践)
  - [1. 不要在异步函数中使用同步阻塞操作](#1-不要在异步函数中使用同步阻塞操作)
  - [2. 正确处理异步上下文](#2-正确处理异步上下文)
  - [3. 避免忘记await](#3-避免忘记await)
- [总结](#总结)
- [相关资源](#相关资源)

## 什么是异步编程？

异步编程是一种编程范式，它允许程序在等待耗时操作（如I/O操作、网络请求）完成时，不阻塞主线程，而是继续执行其他任务。这种方式可以显著提高程序的效率和响应性。

### 同步 vs 异步

**同步编程**：

- 任务按顺序执行
- 一个任务完成后才能执行下一个
- 阻塞式执行

**异步编程**：

- 任务可以并发执行
- 不需要等待一个任务完成就可以开始另一个
- 非阻塞式执行

## Python异步编程基础

### 1. 核心概念

#### 协程（Coroutine）

协程是可以暂停和恢复的函数，使用`async def`定义：

```python
import asyncio

async def hello():
    print("Hello")
    await asyncio.sleep(1)  # 异步等待1秒
    print("World")

# 运行协程
asyncio.run(hello())
```

#### 事件循环（Event Loop）

事件循环是异步编程的核心，负责执行异步任务：

```python
import asyncio

async def main():
    print("开始执行")
    await asyncio.sleep(1)
    print("执行完成")

# 方式1：使用 asyncio.run()
asyncio.run(main())

# 方式2：手动管理事件循环
# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())
```

#### await 关键字

`await`用于等待异步操作完成：

```python
import asyncio
import aiohttp

async def fetch_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def main():
    content = await fetch_url("https://httpbin.org/get")
    print(len(content))

asyncio.run(main())
```

### 2. 异步任务管理

#### 创建和运行任务

```python
import asyncio

async def task1():
    print("任务1开始")
    await asyncio.sleep(2)
    print("任务1完成")
    return "结果1"

async def task2():
    print("任务2开始")
    await asyncio.sleep(1)
    print("任务2完成")
    return "结果2"

async def main():
    # 方式1：使用 create_task
    t1 = asyncio.create_task(task1())
    t2 = asyncio.create_task(task2())
    
    result1 = await t1
    result2 = await t2
    
    print(f"结果：{result1}, {result2}")

asyncio.run(main())
```

#### 并发执行多个任务

```python
import asyncio

async def worker(name, delay):
    print(f"Worker {name} 开始工作")
    await asyncio.sleep(delay)
    print(f"Worker {name} 完成工作")
    return f"Worker {name} 的结果"

async def main():
    # 方式1：使用 asyncio.gather
    results = await asyncio.gather(
        worker("A", 2),
        worker("B", 1),
        worker("C", 3)
    )
    print("所有任务完成：", results)

    # 方式2：使用 asyncio.wait
    tasks = [
        asyncio.create_task(worker("X", 1)),
        asyncio.create_task(worker("Y", 2)),
        asyncio.create_task(worker("Z", 1.5))
    ]
    
    done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
    results = [task.result() for task in done]
    print("Wait结果：", results)

asyncio.run(main())
```

### 3. 异步上下文管理器

```python
import asyncio
import aiofiles

class AsyncResource:
    async def __aenter__(self):
        print("获取资源")
        await asyncio.sleep(0.1)  # 模拟异步初始化
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("释放资源")
        await asyncio.sleep(0.1)  # 模拟异步清理
    
    async def do_something(self):
        print("使用资源做某事")
        await asyncio.sleep(0.5)

async def main():
    async with AsyncResource() as resource:
        await resource.do_something()

asyncio.run(main())
```

### 4. 异步迭代器

```python
import asyncio

class AsyncRange:
    def __init__(self, start, stop):
        self.current = start
        self.stop = stop
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.current < self.stop:
            await asyncio.sleep(0.1)  # 模拟异步操作
            self.current += 1
            return self.current - 1
        else:
            raise StopAsyncIteration

async def main():
    async for i in AsyncRange(0, 5):
        print(f"异步迭代：{i}")

asyncio.run(main())
```

## 实际应用示例

### 1. 异步HTTP请求

```python
import asyncio
import aiohttp
import time

async def fetch_url(session, url):
    try:
        async with session.get(url) as response:
            return {
                'url': url,
                'status': response.status,
                'length': len(await response.text())
            }
    except Exception as e:
        return {'url': url, 'error': str(e)}

async def fetch_multiple_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

async def main():
    urls = [
        'https://httpbin.org/delay/1',
        'https://httpbin.org/delay/2',
        'https://httpbin.org/delay/1',
        'https://httpbin.org/status/200',
        'https://httpbin.org/json'
    ]
    
    start_time = time.time()
    results = await fetch_multiple_urls(urls)
    end_time = time.time()
    
    for result in results:
        print(result)
    
    print(f"总用时：{end_time - start_time:.2f}秒")

# asyncio.run(main())
```

### 2. 异步文件操作

```python
import asyncio
import aiofiles

async def read_file_async(filename):
    async with aiofiles.open(filename, 'r', encoding='utf-8') as f:
        content = await f.read()
        return content

async def write_file_async(filename, content):
    async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
        await f.write(content)

async def process_files():
    # 并发读取多个文件
    files = ['file1.txt', 'file2.txt', 'file3.txt']
    
    tasks = [read_file_async(f) for f in files if f.endswith('.txt')]
    contents = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理结果
    for i, content in enumerate(contents):
        if isinstance(content, Exception):
            print(f"读取文件 {files[i]} 失败：{content}")
        else:
            print(f"文件 {files[i]} 长度：{len(content)}")

# asyncio.run(process_files())
```

### 3. 异步队列

```python
import asyncio
import random

async def producer(queue, name):
    for i in range(5):
        await asyncio.sleep(random.uniform(0.5, 1.5))
        item = f"{name}-item-{i}"
        await queue.put(item)
        print(f"生产者 {name} 生产了 {item}")
    
    # 发送结束信号
    await queue.put(None)

async def consumer(queue, name):
    while True:
        item = await queue.get()
        if item is None:
            # 收到结束信号
            queue.task_done()
            break
        
        # 处理项目
        await asyncio.sleep(random.uniform(0.5, 2))
        print(f"消费者 {name} 处理了 {item}")
        queue.task_done()

async def main():
    queue = asyncio.Queue(maxsize=3)
    
    # 启动生产者和消费者
    await asyncio.gather(
        producer(queue, "P1"),
        consumer(queue, "C1"),
        consumer(queue, "C2")
    )

# asyncio.run(main())
```

## 错误处理和超时

### 1. 异步异常处理

```python
import asyncio

async def risky_operation():
    await asyncio.sleep(1)
    raise ValueError("模拟错误")

async def safe_operation():
    try:
        result = await risky_operation()
        return result
    except ValueError as e:
        print(f"捕获异常：{e}")
        return None

async def main():
    # 处理单个异常
    await safe_operation()
    
    # 处理多个任务的异常
    tasks = [
        asyncio.create_task(risky_operation()),
        asyncio.create_task(asyncio.sleep(2))
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"任务 {i} 发生异常：{result}")
        else:
            print(f"任务 {i} 成功完成：{result}")

# asyncio.run(main())
```

### 2. 超时处理

```python
import asyncio

async def slow_operation():
    await asyncio.sleep(5)
    return "操作完成"

async def main():
    try:
        # 设置3秒超时
        result = await asyncio.wait_for(slow_operation(), timeout=3.0)
        print(result)
    except asyncio.TimeoutError:
        print("操作超时")

    # 使用 asyncio.timeout (Python 3.11+)
    try:
        async with asyncio.timeout(2.0):
            result = await slow_operation()
            print(result)
    except asyncio.TimeoutError:
        print("操作超时")

# asyncio.run(main())
```

## 性能优化建议

### 1. 合理使用并发

```python
import asyncio
import aiohttp

async def fetch_with_semaphore(session, url, semaphore):
    async with semaphore:  # 限制并发数
        async with session.get(url) as response:
            return await response.text()

async def optimized_fetch(urls, max_concurrent=10):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_with_semaphore(session, url, semaphore) 
            for url in urls
        ]
        results = await asyncio.gather(*tasks)
        return results
```

### 2. 避免阻塞操作

```python
import asyncio
import concurrent.futures
import time

def blocking_operation():
    # 模拟CPU密集型任务
    time.sleep(2)
    return "阻塞操作完成"

async def run_blocking_in_thread():
    loop = asyncio.get_event_loop()
    
    # 在线程池中运行阻塞操作
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, blocking_operation)
        return result

async def main():
    # 并发运行阻塞操作
    results = await asyncio.gather(*[
        run_blocking_in_thread() for _ in range(3)
    ])
    print(results)

# asyncio.run(main())
```

## 常见陷阱和最佳实践

### 1. 不要在异步函数中使用同步阻塞操作

```python
# ❌ 错误做法
async def bad_example():
    import time
    time.sleep(1)  # 这会阻塞整个事件循环
    
# ✅ 正确做法
async def good_example():
    await asyncio.sleep(1)  # 非阻塞等待
```

### 2. 正确处理异步上下文

```python
# ❌ 错误做法
async def bad_context():
    session = aiohttp.ClientSession()
    response = await session.get("https://httpbin.org/get")
    # 忘记关闭session

# ✅ 正确做法
async def good_context():
    async with aiohttp.ClientSession() as session:
        async with session.get("https://httpbin.org/get") as response:
            return await response.text()
```

### 3. 避免忘记await

```python
# ❌ 错误做法
async def bad_await():
    asyncio.sleep(1)  # 忘记await，返回协程对象
    
# ✅ 正确做法
async def good_await():
    await asyncio.sleep(1)  # 正确等待
```

## 总结

Python异步编程的核心要点：

1. **基础概念**：理解协程、事件循环和await的工作原理
2. **任务管理**：合理使用`asyncio.gather()`、`asyncio.wait()`等
3. **错误处理**：妥善处理异步操作中的异常和超时
4. **性能优化**：控制并发数量，避免阻塞操作
5. **最佳实践**：正确使用异步上下文管理器，不忘记await

异步编程特别适用于I/O密集型任务，如网络请求、文件操作、数据库查询等。通过合理使用异步编程，可以显著提高程序的效率和响应性。

## 相关资源

- [Python asyncio 官方文档](https://docs.python.org/3/library/asyncio.html)
- [aiohttp 异步HTTP客户端](https://docs.aiohttp.org/)
- [aiofiles 异步文件操作](https://github.com/Tinche/aiofiles)
