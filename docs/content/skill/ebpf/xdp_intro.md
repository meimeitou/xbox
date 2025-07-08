+++
title = 'XDP 简介'
weight = 2
description = 'XDP（eXpress Data Path）是 Linux 内核中的一种高性能、可编程网络数据包处理机制。本文将介绍 XDP 的工作原理、优点以及简单的使用示例。'
+++

- [XDP 工作原理](#xdp-工作原理)
- [XDP 的优点](#xdp-的优点)
- [XDP 简单使用示例](#xdp-简单使用示例)
  - [1. 安装依赖](#1-安装依赖)
  - [2. 编写 XDP 程序](#2-编写-xdp-程序)
  - [3. 编译 XDP 程序](#3-编译-xdp-程序)
  - [4. 加载 XDP 程序](#4-加载-xdp-程序)
  - [5. 卸载 XDP 程序](#5-卸载-xdp-程序)
- [参考资料](#参考资料)

XDP（eXpress Data Path）是 Linux 内核中的一种高性能、可编程网络数据包处理机制。它允许开发者在非常早的网络包接收路径（驱动层）对数据包进行处理，从而实现更高效的数据包过滤、转发或丢弃等功能。XDP 主要应用于网络安全、DDOS 防护、高性能负载均衡等场景。

## XDP 工作原理

XDP 程序在网络驱动层（内核空间）直接运行，能够在数据包上送协议栈之前进行处理。它基于 eBPF 技术，可以动态加载和卸载，不需要重启内核或驱动。

其典型工作流程如下：

1. **接收数据包**：网卡驱动收到数据包。
2. **运行 XDP 程序**：在驱动层调用已加载的 XDP 程序进行处理。
3. **做出决策**：
    - XDP_PASS：允许数据包进入内核协议栈。
    - XDP_DROP：丢弃数据包。
    - XDP_TX：直接将数据包发送回收到的网卡端口。
    - XDP_REDIRECT：将数据包重定向到另一个接口或用户空间。
4. **后续处理**：根据 XDP 程序的返回值，驱动对数据包做相应处理。

## XDP 的优点

- **低延迟、高性能**：减少上下文切换和协议栈开销，适合高性能网络场景。
- **可编程性强**：基于 eBPF，可以灵活定制数据包处理逻辑。
- **无需修改内核或重启服务**：XDP 程序可以动态加载和卸载。

## XDP 简单使用示例

### 1. 安装依赖

XDP 依赖于较新的 Linux 内核（4.8+），建议使用较新的发行版。开发和加载 XDP 程序可以用 [libbpf](https://github.com/libbpf/libbpf) 库和 [iproute2](https://wiki.linuxfoundation.org/networking/iproute2) 工具。

```bash
sudo apt update
sudo apt install clang llvm libbpf-dev gcc-multilib iproute2
```

### 2. 编写 XDP 程序

下面是一个简单的 XDP 程序，用于丢弃所有 ICMP 数据包（例如 ping 测试）：

```c
// xdp_drop_icmp.c
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/icmp.h>
#include <bpf/bpf_helpers.h>

SEC("xdp")
int xdp_drop_icmp(struct xdp_md *ctx) {
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;
    struct ethhdr *eth = data;

    // 以太网头部长度检查
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;
    // 只处理 IP 包
    if (eth->h_proto != __constant_htons(ETH_P_IP))
        return XDP_PASS;

    struct iphdr *ip = data + sizeof(struct ethhdr);
    if ((void *)(ip + 1) > data_end)
        return XDP_PASS;
    // 丢弃 ICMP 包
    if (ip->protocol == IPPROTO_ICMP)
        return XDP_DROP;

    return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
```

### 3. 编译 XDP 程序

```bash
clang -O2 -target bpf -c xdp_drop_icmp.c -o xdp_drop_icmp.o
```

### 4. 加载 XDP 程序

使用 `ip` 工具将 XDP 程序加载到指定网卡（如 `eth0`）：

```bash
sudo ip link set dev eth0 xdp obj xdp_drop_icmp.o
```

### 5. 卸载 XDP 程序

```bash
sudo ip link set dev eth0 xdp off
```

## 参考资料

- [XDP 官方文档](https://xdp-project.net/)
- [Linux XDP 入门教程](https://blog.csdn.net/weixin_44162047/article/details/120555058)
- [eBPF 官方文档](https://ebpf.io/)
