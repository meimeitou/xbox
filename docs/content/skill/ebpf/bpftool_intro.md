+++
title = 'Bpftool介绍'
+++

- [什么是 bpftool？](#什么是-bpftool)
- [安装 bpftool](#安装-bpftool)
  - [Linux 发行版安装](#linux-发行版安装)
    - [Ubuntu/Debian](#ubuntudebian)
    - [CentOS/RHEL/Fedora](#centosrhelfedora)
  - [从源码编译](#从源码编译)
- [基本语法](#基本语法)
- [主要功能模块](#主要功能模块)
  - [1. 程序管理 (prog)](#1-程序管理-prog)
  - [2. 映射管理 (map)](#2-映射管理-map)
  - [3. 网络管理 (net)](#3-网络管理-net)
  - [4. 性能监控 (perf)](#4-性能监控-perf)
  - [5. 控制组管理 (cgroup)](#5-控制组管理-cgroup)
- [常用选项](#常用选项)
- [eBPF 程序管理](#ebpf-程序管理)
  - [查看程序](#查看程序)
  - [程序信息详解](#程序信息详解)
  - [程序固定和解除](#程序固定和解除)
  - [程序跟踪](#程序跟踪)
- [eBPF 映射管理](#ebpf-映射管理)
  - [查看映射](#查看映射)
  - [映射操作](#映射操作)
  - [映射固定](#映射固定)
- [网络 eBPF 程序管理](#网络-ebpf-程序管理)
  - [查看网络程序](#查看网络程序)
  - [网络程序操作](#网络程序操作)
- [性能监控](#性能监控)
  - [perf 事件管理](#perf-事件管理)
- [BTF (BPF Type Format) 管理](#btf-bpf-type-format-管理)
  - [BTF 信息查看](#btf-信息查看)
- [版本差异和兼容性](#版本差异和兼容性)
  - [内核版本要求](#内核版本要求)
  - [检查兼容性](#检查兼容性)
- [总结](#总结)

## 什么是 bpftool？

bpftool 是 Linux 内核提供的一个功能强大的命令行工具，用于管理和调试 eBPF (Extended Berkeley Packet Filter) 程序和映射。它是 Linux 内核源码树的一部分，为开发者和系统管理员提供了一个统一的接口来操作 eBPF 对象。

eBPF 是一种革命性的技术，允许在内核空间中运行用户定义的程序，而无需修改内核代码或加载内核模块。bpftool 作为 eBPF 生态系统的重要组成部分，提供了查看、操作和调试 eBPF 程序的能力。

## 安装 bpftool

### Linux 发行版安装

#### Ubuntu/Debian

```bash
# Ubuntu 20.04 及以上版本
sudo apt update
sudo apt install linux-tools-common linux-tools-generic

# 或者安装特定内核版本的工具
sudo apt install linux-tools-$(uname -r)
```

#### CentOS/RHEL/Fedora

```bash
# CentOS 8/RHEL 8
sudo dnf install bpftool

# CentOS 7/RHEL 7
sudo yum install bpftool

# Fedora
sudo dnf install bpftool
```

### 从源码编译

```bash
# 克隆内核源码
git clone https://github.com/torvalds/linux.git
cd linux/tools/bpf/bpftool

# 编译
make

# 安装
sudo make install
```

## 基本语法

```bash
bpftool [OPTIONS] OBJECT { COMMAND | help }
```

其中：

- `OPTIONS`: 全局选项
- `OBJECT`: 操作对象（prog、map、cgroup、perf、net、feature、btf、gen、struct_ops、link）
- `COMMAND`: 具体命令

## 主要功能模块

### 1. 程序管理 (prog)

管理 eBPF 程序的加载、卸载和查看。

### 2. 映射管理 (map)

管理 eBPF 映射的创建、删除和数据操作。

### 3. 网络管理 (net)

管理网络相关的 eBPF 程序。

### 4. 性能监控 (perf)

管理性能事件相关的 eBPF 程序。

### 5. 控制组管理 (cgroup)

管理 cgroup 相关的 eBPF 程序。

## 常用选项

| 选项 | 说明 |
|------|------|
| `-j, --json` | 以 JSON 格式输出 |
| `-p, --pretty` | 美化输出格式 |
| `-V, --version` | 显示版本信息 |
| `-h, --help` | 显示帮助信息 |
| `-f, --bpffs` | 显示 bpffs 挂载点 |
| `-n, --nomount` | 不自动挂载 bpffs |
| `-d, --debug` | 调试模式 |

## eBPF 程序管理

### 查看程序

```bash
# 列出所有 eBPF 程序
sudo bpftool prog list

# 以 JSON 格式输出
sudo bpftool -j prog list

# 美化输出
sudo bpftool -p prog list

# 显示程序详细信息
sudo bpftool prog show id 123
```

### 程序信息详解

```bash
# 查看程序字节码
sudo bpftool prog dump xlated id 123

# 查看程序原始字节码
sudo bpftool prog dump jited id 123

# 以可视化格式显示
sudo bpftool prog dump xlated id 123 visual
```

### 程序固定和解除

```bash
# 固定程序到文件系统
sudo bpftool prog pin id 123 /sys/fs/bpf/my_prog

# 从文件系统加载程序
sudo bpftool prog load program.o /sys/fs/bpf/my_prog

# 解除固定
sudo rm /sys/fs/bpf/my_prog
```

### 程序跟踪

```bash
# 跟踪程序执行
sudo bpftool prog tracelog

# 分析程序性能
sudo bpftool prog profile id 123
```

## eBPF 映射管理

### 查看映射

```bash
# 列出所有映射
sudo bpftool map list

# 显示映射详细信息
sudo bpftool map show id 456

# 以十六进制格式显示映射内容
sudo bpftool map dump id 456
```

### 映射操作

```bash
# 查看映射中的键值对
sudo bpftool map lookup id 456 key 0x01 0x02 0x03 0x04

# 更新映射值
sudo bpftool map update id 456 key 0x01 0x02 0x03 0x04 value 0x05 0x06 0x07 0x08

# 删除映射条目
sudo bpftool map delete id 456 key 0x01 0x02 0x03 0x04

# 创建新映射
sudo bpftool map create /sys/fs/bpf/my_map type hash key 4 value 8 entries 1024
```

### 映射固定

```bash
# 固定映射到文件系统
sudo bpftool map pin id 456 /sys/fs/bpf/my_map

# 从固定位置获取映射信息
sudo bpftool map show pinned /sys/fs/bpf/my_map
```

## 网络 eBPF 程序管理

### 查看网络程序

```bash
# 列出所有网络相关的 eBPF 程序
sudo bpftool net list

# 显示特定网络接口的程序
sudo bpftool net show dev eth0

# 查看 XDP 程序
sudo bpftool net show xdp

# 查看 TC 程序
sudo bpftool net show tc
```

### 网络程序操作

```bash
# 加载 XDP 程序
sudo bpftool net attach xdp id 123 dev eth0

# 卸载 XDP 程序
sudo bpftool net detach xdp dev eth0

# 查看网络命名空间中的程序
sudo bpftool -n /var/run/netns/test net list
```

## 性能监控

### perf 事件管理

```bash
# 列出所有 perf 事件
sudo bpftool perf list

# 显示 perf 事件详细信息
sudo bpftool perf show

# 以 JSON 格式输出
sudo bpftool -j perf list
```

## BTF (BPF Type Format) 管理

### BTF 信息查看

```bash
# 列出所有 BTF 对象
sudo bpftool btf list

# 显示 BTF 详细信息
sudo bpftool btf show id 789

# 转储 BTF 格式
sudo bpftool btf dump id 789

# 转储为 C 头文件格式
sudo bpftool btf dump id 789 format c
```

## 版本差异和兼容性

### 内核版本要求

| 功能 | 最低内核版本 |
|------|-------------|
| 基本 bpftool | 4.15 |
| BTF 支持 | 4.18 |
| 链接管理 | 5.7 |
| 结构体操作 | 5.6 |

### 检查兼容性

```bash
# 检查内核版本
uname -r

# 检查 eBPF 功能支持
sudo bpftool feature probe

# 检查特定功能
sudo bpftool feature probe kernel | grep -i btf
```

## 总结

bpftool 是管理 eBPF 程序和映射的重要工具，提供了丰富的功能来：

1. **程序管理** - 加载、卸载、查看和调试 eBPF 程序
2. **映射管理** - 创建、操作和监控 eBPF 映射
3. **网络管理** - 管理网络相关的 eBPF 程序
4. **性能监控** - 监控和分析 eBPF 程序性能
5. **调试支持** - 提供强大的调试和故障排除功能

掌握 bpftool 的使用对于 eBPF 开发和系统管理至关重要。随着 eBPF 技术的发展，bpftool 也在不断完善，为用户提供更加强大和便利的功能。

建议在使用 bpftool 时：

- 始终使用 sudo 权限
- 关注内核版本兼容性
- 合理使用输出格式选项
- 定期备份重要的 eBPF 对象
- 结合其他工具进行综合分析

希望这篇介绍能够帮助您更好地理解和使用 bpftool 工具！

---

*作者：meimeitou*  
