+++
title = 'Bpf2go'
weight = 4
+++

[bpf2go](https://github.com/cilium/ebpf/tree/master/cmd/bpf2go) 是 Cilium eBPF 项目中用于将 eBPF C 代码自动生成 Go 绑定代码的工具，极大地方便了在 Golang 项目中集成 eBPF 程序。本文介绍 bpf2go 的工作原理、使用方法和常用技巧。

## 一、bpf2go 简介

bpf2go 是一个命令行工具，可以自动完成以下工作：

1. 使用 clang/llvm 编译 eBPF C 程序为 `.o` 对象文件；
2. 自动生成 Go 语言绑定代码，方便直接在 Go 程序中加载和操作 eBPF 程序和 Map。

其本质是将 eBPF C 代码编译、嵌入到 Go 源文件，并自动生成访问 Map、加载程序的代码，大幅降低手动集成成本。

---

## 二、环境准备

- Go 1.18+
- clang/llvm 工具链（用于编译 eBPF C 代码）
- Linux 4.18+ 支持 eBPF
- 安装 cilium/ebpf 包：

```bash
go get github.com/cilium/ebpf
```

---

## 三、bpf2go 用法

### 1. 安装 bpf2go

bpf2go 随 cilium/ebpf 项目发布，无需单独安装。通过 `go generate` 自动调用。

### 2. 编写 eBPF C 代码

例如，创建一个简单的 eBPF 程序 `xdp_pass_kern.c`：

```c
// xdp_pass_kern.c
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

SEC("xdp")
int xdp_pass_prog(struct xdp_md *ctx) {
    return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
```

### 3. 在 Go 代码中使用 bpf2go

在你的 Go 源码中添加如下指令（假设文件名为 `main.go`）：

```go
//go:generate go run github.com/cilium/ebpf/cmd/bpf2go -cc clang -cflags "-O2 -g -Wall" XdpPass xdp_pass_kern.c -- -I/usr/include
```

- `XdpPass` 是生成的 Go 绑定的前缀。
- `xdp_pass_kern.c` 是 eBPF 源码文件。
- `--` 后面为 clang 的参数（比如头文件目录）。

### 4. 生成代码

在工程目录下执行：

```bash
go generate
```

这将自动完成 eBPF C 代码编译，并生成两个文件：

- `xdp_pass_kern_bpfel.go`（或 `xdp_pass_kern_bpfeb.go`，视系统架构而定）
- `xdp_pass_kern_bpfel.c`（编译后的对象文件）

### 5. 在 Go 代码中加载和使用 eBPF 程序

```go
package main

import (
    "github.com/cilium/ebpf"
    "log"
)

func main() {
    // 加载自动生成的对象
    objs := XdpPassObjects{}
    if err := LoadXdpPassObjects(&objs, nil); err != nil {
        log.Fatalf("loading objects: %v", err)
    }
    defer objs.Close()

    // 访问 eBPF 程序
    prog := objs.XdpPassProg

    // 后续可以用 prog.Attach... 等方式挂载 XDP 程序到网卡
}
```

---

## 四、常用技巧

- 可以用多个 `//go:generate` 指令生成多个 eBPF 程序的绑定。
- 支持 Map、PerfEvent、RingBuffer 等类型的自动绑定。
- 结合 Makefile、CI 流水线自动化生成。

---

## 五、参考资料

- [bpf2go 官方文档](https://pkg.go.dev/github.com/cilium/ebpf/cmd/bpf2go)
- [cilium/ebpf 项目](https://github.com/cilium/ebpf)
- [eBPF 入门教程](https://ebpf.io/)
