+++
title = 'BCC介绍'
+++

BCC（BPF Compiler Collection）是一个基于 eBPF 的强大工具集，方便开发者在 Linux 上编写、加载和运行 eBPF 程序。它广泛用于性能分析、故障排查和安全监控领域。本文将带你快速了解 BCC 的基本使用方法。

---

## 1. BCC 简介

BCC 提供了丰富的工具库和 Python/C++ API，使得用户可以用高级语言编写 eBPF 程序并在内核态运行。常见的 BCC 工具包括 `execsnoop`、`opensnoop`、`tcpconnect` 等。

---

## 2. 安装 BCC

以 Ubuntu 为例：

```bash
sudo apt update
sudo apt install bpfcc-tools linux-headers-$(uname -r) python3-bcc
```

或使用源码安装（适用于其他发行版或获取最新版）：

```bash
git clone https://github.com/iovisor/bcc.git
cd bcc
mkdir build; cd build
cmake ..
make
sudo make install
```

---

## 3. BCC 常用工具

### 3.1 查看有哪些工具

```bash
ls /usr/share/bcc/tools
```

### 3.2 示例：追踪新进程的创建

```bash
sudo /usr/share/bcc/tools/execsnoop
```

### 3.3 示例：追踪文件打开

```bash
sudo /usr/share/bcc/tools/opensnoop
```

### 3.4 示例：追踪 TCP 连接

```bash
sudo /usr/share/bcc/tools/tcpconnect
```

---

## 4. 用 Python 编写自定义 BCC 脚本

示例：统计每秒创建的进程数

```python
from bcc import BPF
import time

program = """
int count_exec(struct pt_regs *ctx) {
    bpf_trace_printk("exec() called\\n");
    return 0;
}
"""

b = BPF(text=program)
b.attach_kprobe(event="sys_execve", fn_name="count_exec")

print("Tracing execve()... Hit Ctrl-C to end.")
while True:
    try:
        (task, pid, cpu, flags, ts, msg) = b.trace_fields()
        print(f"{ts}: {msg}")
    except KeyboardInterrupt:
        exit()
```

保存为 `exec_count.py`，执行：

```bash
sudo python3 exec_count.py
```

---

## 5. 常见问题

- 需 root 权限运行
- 内核需支持 eBPF（4.1 及以上较好）
- 某些工具依赖对应内核版本的 headers

---

## 6. 参考资料

- [BCC 官方文档](https://github.com/iovisor/bcc/blob/master/INSTALL.md)
- [BCC 工具中文文档](https://bcc.iovisor.org/tools.html)
- [eBPF 入门](https://ebpf.io/)
