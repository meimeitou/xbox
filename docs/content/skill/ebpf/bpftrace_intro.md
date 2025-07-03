+++
title = 'Bpftrace介绍'
+++

bpftrace 是一种基于 eBPF 的高层次追踪工具，支持用类似 DTrace 的脚本语言编写内核和用户空间的追踪逻辑。bpftrace 适用于 Linux 性能分析、故障排查、安全监控等多种场景。本文将简要介绍 bpftrace 的安装、基本用法及常见示例。

---

## 1. bpftrace 简介

bpftrace 允许开发者用简单脚本描述内核或用户态的观测点（probe），并直接在命令行运行，实时获取数据。其语法简洁，适合日常运维和开发快速定位问题。

---

## 2. 安装 bpftrace

以 Ubuntu 为例：

```bash
sudo apt update
sudo apt install bpftrace linux-headers-$(uname -r)
```

或在其他发行版使用包管理器安装，或参考 [官方编译指南](https://github.com/iovisor/bpftrace/blob/master/INSTALL.md)。

---

## 3. bpftrace 基本语法

bpftrace 的脚本结构为：

```bpftrace
probe_type:probe_name
/过滤条件/
{
    动作
}
```

- `probe_type` 如 kprobe、uprobe、tracepoint 等
- `probe_name` 为函数名或事件名
- `/过滤条件/` 可选，为布尔表达式
- `{动作}` 为 BPF 动作代码块

---

## 4. 常用示例

### 4.1 统计 sys_execve 系统调用频率

```bash
sudo bpftrace -e 'kprobe:sys_execve { @[comm] = count(); }'
```

打印每个进程调用 `execve` 的次数。

### 4.2 跟踪 open 系统调用参数

```bash
sudo bpftrace -e 'tracepoint:syscalls:sys_enter_openat { printf("%s: %s\n", comm, str(args->filename)); }'
```

输出每个进程打开的文件名。

### 4.3 跟踪特定进程的网络发送

```bash
sudo bpftrace -e 'kprobe:tcp_sendmsg /comm == "nginx"/ { @[comm] = count(); }'
```

统计 nginx 进程的 tcp_sendmsg 调用次数。

### 4.4 统计函数延迟

```bash
sudo bpftrace -e '
kprobe:vfs_read { @start[tid] = nsecs; }
kretprobe:vfs_read /@start[tid]/ {
    @latency = hist(nsecs - @start[tid]);
    delete(@start[tid]);
}
'
```

查看 vfs_read 的延迟直方图。

---

## 5. 编写和运行 bpftrace 脚本文件

编写脚本 `example.bt`：

```bpftrace
#!/usr/bin/env bpftrace

BEGIN { printf("Tracing execve syscalls... Hit Ctrl-C to end.\n"); }
kprobe:sys_execve { @[comm] = count(); }
```

运行：

```bash
sudo bpftrace example.bt
```

---

## 6. 注意事项

- bpftrace 需要 root 权限运行
- 内核需支持 eBPF（推荐 4.9+）
- 某些 probe 依赖内核调试符号（如 kprobe）
- 生产环境使用需注意对性能的影响

---

## 7. 参考资料

- [bpftrace 官方文档](https://github.com/iovisor/bpftrace)
- [bpftrace Reference Guide](https://github.com/iovisor/bpftrace/blob/master/docs/reference_guide.md)
- [eBPF 入门](https://ebpf.io/)
