+++
title = 'Wireguard VPN'
+++

- [什么是 WireGuard？](#什么是-wireguard)
- [核心特性](#核心特性)
- [安装 WireGuard](#安装-wireguard)
  - [Ubuntu/Debian](#ubuntudebian)
  - [CentOS/RHEL](#centosrhel)
  - [macOS](#macos)
- [基本概念](#基本概念)
  - [密钥对生成](#密钥对生成)
  - [配置文件结构](#配置文件结构)
- [常用场景配置示例](#常用场景配置示例)
  - [场景一：点对点连接](#场景一点对点连接)
  - [场景二：远程办公 VPN](#场景二远程办公-vpn)
  - [场景三：全流量代理](#场景三全流量代理)
  - [场景四：站点到站点连接](#场景四站点到站点连接)
  - [场景五：多跳连接](#场景五多跳连接)
  - [场景六：动态 IP 客户端](#场景六动态-ip-客户端)
- [管理和操作命令](#管理和操作命令)
  - [启动和停止服务](#启动和停止服务)
  - [查看状态](#查看状态)
  - [动态配置管理](#动态配置管理)
- [安全最佳实践](#安全最佳实践)
  - [1. 密钥管理](#1-密钥管理)
  - [2. 防火墙配置](#2-防火墙配置)
  - [3. 定期轮换密钥](#3-定期轮换密钥)
- [监控和故障排除](#监控和故障排除)
  - [日志查看](#日志查看)
  - [网络测试](#网络测试)
  - [常见问题解决](#常见问题解决)
- [性能优化](#性能优化)
  - [1. 调整 MTU](#1-调整-mtu)
  - [2. 优化内核参数](#2-优化内核参数)
  - [3. 使用多个工作线程](#3-使用多个工作线程)
- [移动设备配置](#移动设备配置)
  - [生成 QR 码](#生成-qr-码)
  - [iOS/Android 应用](#iosandroid-应用)
- [总结](#总结)

WireGuard 是一个现代、快速、安全的 VPN 实现，以其简洁的设计和卓越的性能而闻名。本文将详细介绍 WireGuard 的使用方法和常见场景的配置示例。

## 什么是 WireGuard？

WireGuard 是一个极其简单且快速的现代 VPN，它利用了最先进的加密技术。它旨在比 IPsec 更快、更简单、更精简，同时避免了令人头疼的配置。它比 OpenVPN 更高效，具有更好的性能，同时提供更强的安全性。

## 核心特性

- **极简设计**：只有约 4000 行代码
- **高性能**：比传统 VPN 协议更快
- **强加密**：使用现代密码学算法
- **跨平台**：支持 Linux、Windows、macOS、iOS、Android
- **配置简单**：基于公钥密码学，配置文件易于理解

## 安装 WireGuard

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install wireguard
```

### CentOS/RHEL

```bash
sudo yum install epel-release
sudo yum install wireguard-tools
```

### macOS

```bash
brew install wireguard-tools
```

## 基本概念

### 密钥对生成

```bash
# 生成私钥
wg genkey > private.key

# 从私钥生成公钥
wg pubkey < private.key > public.key

# 一键生成密钥对
wg genkey | tee private.key | wg pubkey > public.key
```

### 配置文件结构

WireGuard 配置文件通常位于 `/etc/wireguard/` 目录下，文件扩展名为 `.conf`。

## 常用场景配置示例

### 场景一：点对点连接

**使用场景**：两台服务器之间建立安全连接

**服务器 A 配置 (wg0.conf)**：

```ini
[Interface]
PrivateKey = SERVER_A_PRIVATE_KEY
Address = 10.0.0.1/24
ListenPort = 51820

[Peer]
PublicKey = SERVER_B_PUBLIC_KEY
Endpoint = server-b-ip:51820
AllowedIPs = 10.0.0.2/32
PersistentKeepalive = 25
```

**服务器 B 配置 (wg0.conf)**：

```ini
[Interface]
PrivateKey = SERVER_B_PRIVATE_KEY
Address = 10.0.0.2/24
ListenPort = 51820

[Peer]
PublicKey = SERVER_A_PUBLIC_KEY
Endpoint = server-a-ip:51820
AllowedIPs = 10.0.0.1/32
PersistentKeepalive = 25
```

### 场景二：远程办公 VPN

**使用场景**：员工远程连接到公司内网

公司可能会通报使用 WireGuard...

**VPN 服务器配置 (wg0.conf)**：

```ini
[Interface]
PrivateKey = VPN_SERVER_PRIVATE_KEY
Address = 10.8.0.1/24
ListenPort = 51820
PostUp = iptables -A FORWARD -i %i -j ACCEPT; iptables -A FORWARD -o %i -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i %i -j ACCEPT; iptables -D FORWARD -o %i -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

# 客户端 1
[Peer]
PublicKey = CLIENT1_PUBLIC_KEY
AllowedIPs = 10.8.0.2/32

# 客户端 2
[Peer]
PublicKey = CLIENT2_PUBLIC_KEY
AllowedIPs = 10.8.0.3/32

# 客户端 3
[Peer]
PublicKey = CLIENT3_PUBLIC_KEY
AllowedIPs = 10.8.0.4/32
```

**客户端配置 (client.conf)**：

```ini
[Interface]
PrivateKey = CLIENT_PRIVATE_KEY
Address = 10.8.0.2/24
DNS = 8.8.8.8, 1.1.1.1

[Peer]
PublicKey = VPN_SERVER_PUBLIC_KEY
Endpoint = vpn.company.com:51820
AllowedIPs = 192.168.1.0/24, 10.8.0.0/24
PersistentKeepalive = 25
```

### 场景三：全流量代理

**使用场景**：所有网络流量通过 VPN 服务器

**服务器配置 (wg0.conf)**：

```ini
[Interface]
PrivateKey = VPN_SERVER_PRIVATE_KEY
Address = 10.7.0.1/24
ListenPort = 51820
PostUp = iptables -A FORWARD -i %i -j ACCEPT; iptables -A FORWARD -o %i -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE; iptables -A INPUT -p udp -m udp --dport 51820 -j ACCEPT
PostDown = iptables -D FORWARD -i %i -j ACCEPT; iptables -D FORWARD -o %i -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE; iptables -D INPUT -p udp -m udp --dport 51820 -j ACCEPT

[Peer]
PublicKey = CLIENT_PUBLIC_KEY
AllowedIPs = 10.7.0.2/32
```

**客户端配置 (client.conf)**：

```ini
[Interface]
PrivateKey = CLIENT_PRIVATE_KEY
Address = 10.7.0.2/24
DNS = 1.1.1.1, 8.8.8.8

[Peer]
PublicKey = VPN_SERVER_PUBLIC_KEY
Endpoint = your-server-ip:51820
AllowedIPs = 0.0.0.0/0
PersistentKeepalive = 25
```

### 场景四：站点到站点连接

**使用场景**：连接两个不同地理位置的办公网络

**总部网关配置 (wg0.conf)**：

```ini
[Interface]
PrivateKey = HQ_GATEWAY_PRIVATE_KEY
Address = 10.100.0.1/30
ListenPort = 51820
PostUp = iptables -A FORWARD -i %i -j ACCEPT; iptables -A FORWARD -o %i -j ACCEPT
PostDown = iptables -D FORWARD -i %i -j ACCEPT; iptables -D FORWARD -o %i -j ACCEPT

[Peer]
PublicKey = BRANCH_GATEWAY_PUBLIC_KEY
Endpoint = branch-office-ip:51820
AllowedIPs = 10.100.0.2/32, 192.168.2.0/24
PersistentKeepalive = 25
```

**分支机构网关配置 (wg0.conf)**：

```ini
[Interface]
PrivateKey = BRANCH_GATEWAY_PRIVATE_KEY
Address = 10.100.0.2/30
ListenPort = 51820
PostUp = iptables -A FORWARD -i %i -j ACCEPT; iptables -A FORWARD -o %i -j ACCEPT
PostDown = iptables -D FORWARD -i %i -j ACCEPT; iptables -D FORWARD -o %i -j ACCEPT

[Peer]
PublicKey = HQ_GATEWAY_PUBLIC_KEY
Endpoint = hq-office-ip:51820
AllowedIPs = 10.100.0.1/32, 192.168.1.0/24
PersistentKeepalive = 25
```

### 场景五：多跳连接

**使用场景**：通过中继服务器连接多个节点

**中继服务器配置 (wg0.conf)**：

```ini
[Interface]
PrivateKey = RELAY_PRIVATE_KEY
Address = 10.200.0.1/24
ListenPort = 51820
PostUp = iptables -A FORWARD -i %i -j ACCEPT; iptables -A FORWARD -o %i -j ACCEPT
PostDown = iptables -D FORWARD -i %i -j ACCEPT; iptables -D FORWARD -o %i -j ACCEPT

# 节点 A
[Peer]
PublicKey = NODE_A_PUBLIC_KEY
AllowedIPs = 10.200.0.2/32

# 节点 B
[Peer]
PublicKey = NODE_B_PUBLIC_KEY
AllowedIPs = 10.200.0.3/32

# 节点 C
[Peer]
PublicKey = NODE_C_PUBLIC_KEY
AllowedIPs = 10.200.0.4/32
```

**节点配置示例 (node-a.conf)**：

```ini
[Interface]
PrivateKey = NODE_A_PRIVATE_KEY
Address = 10.200.0.2/24

[Peer]
PublicKey = RELAY_PUBLIC_KEY
Endpoint = relay-server-ip:51820
AllowedIPs = 10.200.0.0/24
PersistentKeepalive = 25
```

### 场景六：动态 IP 客户端

**使用场景**：客户端 IP 地址经常变化

**服务器配置 (wg0.conf)**：

```ini
[Interface]
PrivateKey = SERVER_PRIVATE_KEY
Address = 10.6.0.1/24
ListenPort = 51820
PostUp = iptables -A FORWARD -i %i -j ACCEPT; iptables -A FORWARD -o %i -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i %i -j ACCEPT; iptables -D FORWARD -o %i -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

[Peer]
PublicKey = MOBILE_CLIENT_PUBLIC_KEY
AllowedIPs = 10.6.0.2/32
# 不设置 Endpoint，允许客户端从任何 IP 连接
```

**移动客户端配置 (mobile.conf)**：

```ini
[Interface]
PrivateKey = MOBILE_CLIENT_PRIVATE_KEY
Address = 10.6.0.2/24
DNS = 8.8.8.8

[Peer]
PublicKey = SERVER_PUBLIC_KEY
Endpoint = your-server.com:51820
AllowedIPs = 0.0.0.0/0
PersistentKeepalive = 25
```

## 管理和操作命令

### 启动和停止服务

```bash
# 启动接口
sudo wg-quick up wg0

# 停止接口
sudo wg-quick down wg0

# 重启接口
sudo wg-quick down wg0 && sudo wg-quick up wg0

# 开机自启动
sudo systemctl enable wg-quick@wg0
```

### 查看状态

```bash
# 查看接口状态
sudo wg show

# 查看特定接口
sudo wg show wg0

# 查看详细信息
sudo wg show wg0 dump
```

### 动态配置管理

```bash
# 添加新的 peer
sudo wg set wg0 peer PEER_PUBLIC_KEY allowed-ips 10.8.0.10/32

# 删除 peer
sudo wg set wg0 peer PEER_PUBLIC_KEY remove

# 保存配置
sudo wg-quick save wg0
```

## 安全最佳实践

### 1. 密钥管理

```bash
# 确保私钥文件权限安全
sudo chmod 600 /etc/wireguard/*.conf
sudo chown root:root /etc/wireguard/*.conf
```

### 2. 防火墙配置

```bash
# 只允许必要的端口
sudo ufw allow 51820/udp
sudo ufw enable

# 限制管理端口访问
sudo ufw allow from 10.8.0.0/24 to any port 22
```

### 3. 定期轮换密钥

```bash
#!/bin/bash
# 密钥轮换脚本示例
OLD_KEY=$(wg show wg0 private-key)
NEW_KEY=$(wg genkey)
echo $NEW_KEY | sudo wg set wg0 private-key /dev/stdin
echo "密钥已更新，请更新客户端配置"
```

## 监控和故障排除

### 日志查看

```bash
# 查看系统日志
sudo journalctl -u wg-quick@wg0 -f

# 查看内核消息
sudo dmesg | grep wireguard
```

### 网络测试

```bash
# 测试连通性
ping 10.8.0.1

# 追踪路由
traceroute 10.8.0.1

# 测试 UDP 连接
nc -u server-ip 51820
```

### 常见问题解决

```bash
# 检查接口状态
ip addr show wg0

# 检查路由表
ip route show table main

# 重新加载配置
sudo wg-quick down wg0
sudo wg-quick up wg0
```

## 性能优化

### 1. 调整 MTU

```ini
[Interface]
MTU = 1420
```

### 2. 优化内核参数

```bash
# /etc/sysctl.conf
net.core.default_qdisc = fq
net.ipv4.tcp_congestion_control = bbr
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
```

### 3. 使用多个工作线程

```bash
# 启用多队列支持
echo 'net.core.netdev_max_backlog = 5000' >> /etc/sysctl.conf
```

## 移动设备配置

### 生成 QR 码

```bash
# 为移动设备生成 QR 码
qrencode -t ansiutf8 < client.conf
```

### iOS/Android 应用

1. 安装官方 WireGuard 应用
2. 扫描 QR 码或手动输入配置
3. 启用连接

## 总结

WireGuard 以其简洁、高效、安全的特点，正在成为现代 VPN 解决方案的首选。通过本文介绍的各种场景配置，您可以：

1. **快速部署**：简单的配置文件格式易于理解和维护
2. **灵活应用**：支持点对点、站点到站点、移动接入等多种场景
3. **高性能**：相比传统 VPN 协议有更好的性能表现
4. **强安全性**：使用现代加密算法确保通信安全
