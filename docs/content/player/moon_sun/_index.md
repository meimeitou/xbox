+++
title = "Sunshine and Moonlight"
weight = 1
+++

# 原神启动

## 摘要

开源的 Sunshine 和 Moonlight 是两个流行的游戏流媒体解决方案，允许用户在不同设备上流式传输 PC 游戏。Sunshine 是一个 GameStream 服务器，而 Moonlight 是一个 GameStream 客户端。

- [moonlight](https://github.com/moonlight-stream/moonlight-qt) 是一个开源的 NVIDIA GameStream 客户端，允许用户在各种设备上流式传输游戏。
- [sunshine](https://github.com/LizardByte/Sunshine) 是一个开源的 GameStream 服务器，旨在提供与 NVIDIA GameStream 相似的功能，但不依赖于 NVIDIA 硬件。

## ✅ 简介

Sunshine：开源的 NVIDIA GameStream 兼容服务器，运行在主机（Windows/Linux）上，用于编码和推流画面。

Moonlight：客户端程序（支持 Android、iOS、Windows、macOS、Linux），用于接收 Sunshine 推送的画面，实现远程控制和观看。

## ✅ 工作流程

主机（无显示器）运行 Sunshine：

使用 GPU 编码（如 NVENC） 把桌面图像压缩成视频流。

通过局域网（或公网）推送到客户端。

客户端（如 iPad）运行 Moonlight：

接收并解码 Sunshine 推来的流。

通过触控或蓝牙键鼠回传控制指令。

🎯 关键问题：主机无显示器（Headless）时的注意事项
默认情况下，NVIDIA GPU 在无显示器连接时，不会启用桌面图形输出，Sunshine 也无法正常编码。

🧩 解决方案：使用“虚拟显示器”
要解决这个问题，你可以采用以下方法之一：

方法 1：插入 虚拟显示器（Dummy HDMI Plug）
在主机显卡 HDMI/DP 接口插入一个 HDMI 虚拟显示器头（Dummy Plug）。

操作系统将认为有一个显示器连接，从而启用 GPU 桌面输出。

优点：最稳定、兼容性好，成本低（淘宝/拼多多几块钱一个）。

方法 2：软件虚拟显示器（不推荐）
某些系统可以使用虚拟显示驱动（如 Dummy Display Driver）实现假显示器。

但在现代系统/NVIDIA 驱动中不稳定，经常失败或不支持硬件加速。

## 🚀 实际部署步骤（以 Windows 主机 + iPad 为例）

### 1. 主机安装 Sunshine

官网：<https://github.com/LizardByte/Sunshine>

安装后配置：

启用 Allow without display

添加远程启动程序（如 explorer.exe 启动整个桌面）

配置分辨率和码率

### 2. 插入 Dummy HDMI Plug

插入后，前往 显示设置 检查系统是否识别显示器（通常为 1920x1080）

### 3. 客户端安装 Moonlight

iOS：App Store 下载 “Moonlight”

Android：Play Store 下载

初次连接输入主机的 IP 地址（确保网络通）

### 4. 配置连接

确保防火墙允许 47984/47989 等端口（UDP）

在主机 Sunshine 配置中授权客户端配对

## ✅ 总结

项目 描述
主机系统 Windows 或 Linux，建议配备 NVIDIA GPU
显示器模拟 插 Dummy HDMI Plug
推流服务端 Sunshine
客户端应用 Moonlight（支持触控、蓝牙键鼠）
编码方式 NVENC（推荐）或软件编码（性能差）
实用场景 家用游戏主机远程玩、办公主机无显示器远程控制
