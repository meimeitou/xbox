+++
title = 'frp内外穿透'
+++

- [什么是 FRP？](#什么是-frp)
- [核心特性](#核心特性)
  - [1. 协议支持](#1-协议支持)
  - [2. 高级功能](#2-高级功能)
  - [3. 管理功能](#3-管理功能)
- [安装 FRP](#安装-frp)
  - [1. 下载预编译版本](#1-下载预编译版本)
  - [2. 使用 Docker 安装](#2-使用-docker-安装)
- [基本配置](#基本配置)
  - [服务端配置（frps.ini）](#服务端配置frpsini)
  - [客户端配置（frpc.ini）](#客户端配置frpcini)
- [常用代理配置](#常用代理配置)
  - [1. SSH 代理](#1-ssh-代理)
  - [2. HTTP 代理](#2-http-代理)
  - [3. HTTPS 代理](#3-https-代理)
  - [4. UDP 代理](#4-udp-代理)
- [高级配置](#高级配置)
  - [1. 负载均衡](#1-负载均衡)
  - [2. 端口范围代理](#2-端口范围代理)
  - [3. 健康检查](#3-健康检查)
  - [4. 带宽限制](#4-带宽限制)
  - [5. 插件配置](#5-插件配置)
- [安全配置](#安全配置)
  - [1. TLS 加密](#1-tls-加密)
  - [2. 基于用户的访问控制](#2-基于用户的访问控制)
  - [3. 端口白名单](#3-端口白名单)
  - [4. HTTP 基本认证](#4-http-基本认证)
- [部署和管理](#部署和管理)
  - [1. 系统服务配置](#1-系统服务配置)
    - [systemd 服务（frps.service）](#systemd-服务frpsservice)
    - [systemd 服务（frpc.service）](#systemd-服务frpcservice)
  - [2. 启动和管理](#2-启动和管理)
  - [3. Docker 部署](#3-docker-部署)
    - [Docker Compose 配置](#docker-compose-配置)
- [性能优化](#性能优化)
  - [1. 系统优化](#1-系统优化)
  - [2. FRP 优化配置](#2-frp-优化配置)
- [故障排除](#故障排除)
  - [1. 常见问题](#1-常见问题)
    - [连接失败](#连接失败)
    - [性能问题](#性能问题)
  - [2. 调试技巧](#2-调试技巧)
  - [3. 配置验证](#3-配置验证)
- [实际应用场景](#实际应用场景)
  - [1. 家庭服务器](#1-家庭服务器)
  - [2. 开发环境](#2-开发环境)
  - [3. 远程办公](#3-远程办公)
- [总结](#总结)

## 什么是 FRP？

FRP（Fast Reverse Proxy）是一个高性能的反向代理应用，专注于内网穿透。它可以将内网服务通过具有公网 IP 的服务器暴露给公网，支持 TCP、UDP、HTTP、HTTPS 等多种协议。

FRP 由客户端（frpc）和服务端（frps）组成，通过在具有公网 IP 的机器上部署 frps，在内网机器上部署 frpc，轻松实现内网穿透功能。

## 核心特性

### 1. 协议支持

- **TCP**: 支持任意 TCP 协议
- **UDP**: 支持 UDP 协议穿透
- **HTTP**: HTTP 服务代理
- **HTTPS**: HTTPS 服务代理
- **STCP**: 安全的 TCP 连接
- **SUDP**: 安全的 UDP 连接

### 2. 高级功能

- **负载均衡**: 支持多个客户端负载均衡
- **端口复用**: HTTP/HTTPS 端口复用
- **加密传输**: 支持 TLS 加密
- **压缩传输**: 支持 gzip 压缩
- **带宽限制**: 可限制代理带宽
- **连接池**: 连接池提升性能

### 3. 管理功能

- **Web 管理界面**: 友好的 Web 管理界面
- **API 接口**: RESTful API 管理
- **统计信息**: 详细的连接统计
- **日志记录**: 完整的操作日志

## 安装 FRP

### 1. 下载预编译版本

```bash
# 下载最新版本（以 Linux amd64 为例）
wget https://github.com/fatedier/frp/releases/download/v0.52.3/frp_0.52.3_linux_amd64.tar.gz

# 解压
tar -zxvf frp_0.52.3_linux_amd64.tar.gz

# 进入目录
cd frp_0.52.3_linux_amd64

# 查看文件
ls -la
```

### 2. 使用 Docker 安装

```bash
# 服务端
docker pull fatedier/frps:latest

# 客户端
docker pull fatedier/frpc:latest
```

## 基本配置

### 服务端配置（frps.ini）

```ini
[common]
# 服务端监听地址
bind_addr = 0.0.0.0
# 服务端监听端口
bind_port = 7000

# 用于接收客户端连接的端口范围
proxy_bind_addr = 0.0.0.0

# HTTP 监听端口
vhost_http_port = 80
# HTTPS 监听端口
vhost_https_port = 443

# 管理后台端口
dashboard_port = 7500
# 管理后台用户名
dashboard_user = admin
# 管理后台密码
dashboard_pwd = admin123

# 身份验证令牌
token = your_token_here

# 允许的端口范围
allow_ports = 2000-3000,3001,3003,4000-50000

# 最大连接池大小
max_pool_count = 5

# 日志配置
log_file = ./frps.log
log_level = info
log_max_days = 3

# 子域名
subdomain_host = frp.example.com

# TLS 配置
tls_only = false
```

### 客户端配置（frpc.ini）

```ini
[common]
# 服务端地址
server_addr = your_server_ip
# 服务端端口
server_port = 7000

# 身份验证令牌
token = your_token_here

# 管理后台
admin_addr = 127.0.0.1
admin_port = 7400
admin_user = admin
admin_pwd = admin123

# 连接池大小
pool_count = 1

# 日志配置
log_file = ./frpc.log
log_level = info
log_max_days = 3

# 心跳配置
heartbeat_interval = 30
heartbeat_timeout = 90

# 启用压缩
use_compression = true
# 启用加密
use_encryption = true

# 协议类型
protocol = tcp

# DNS 服务器
dns_server = 8.8.8.8

# 用户标识
user = test_user
```

## 常用代理配置

### 1. SSH 代理

```ini
# 客户端配置
[ssh]
type = tcp
local_ip = 127.0.0.1
local_port = 22
remote_port = 6000
```

连接方式：

```bash
ssh -p 6000 user@your_server_ip
```

### 2. HTTP 代理

```ini
# 客户端配置
[web]
type = http
local_ip = 127.0.0.1
local_port = 8080
custom_domains = web.example.com
```

访问方式：

```bash
curl http://web.example.com
```

### 3. HTTPS 代理

```ini
# 客户端配置
[web_https]
type = https
local_ip = 127.0.0.1
local_port = 8443
custom_domains = web.example.com
```

### 4. UDP 代理

```ini
# 客户端配置
[dns]
type = udp
local_ip = 127.0.0.1
local_port = 53
remote_port = 6053
```

## 高级配置

### 1. 负载均衡

```ini
# 客户端 1
[web1]
type = http
local_ip = 127.0.0.1
local_port = 8080
custom_domains = web.example.com
group = web
group_key = 123456

# 客户端 2
[web2]
type = http
local_ip = 127.0.0.1
local_port = 8081
custom_domains = web.example.com
group = web
group_key = 123456
```

### 2. 端口范围代理

```ini
# 客户端配置
[range_tcp]
type = tcp
local_ip = 127.0.0.1
local_port = 6000-6010
remote_port = 6000-6010
```

### 3. 健康检查

```ini
# 客户端配置
[web_health]
type = http
local_ip = 127.0.0.1
local_port = 8080
custom_domains = web.example.com
health_check_type = http
health_check_url = /health
health_check_interval_s = 10
health_check_max_failed = 3
health_check_timeout_s = 3
```

### 4. 带宽限制

```ini
# 客户端配置
[web_limited]
type = http
local_ip = 127.0.0.1
local_port = 8080
custom_domains = web.example.com
bandwidth_limit = 1MB
```

### 5. 插件配置

```ini
# 客户端配置
[unix_domain_socket]
type = tcp
remote_port = 6003
plugin = unix_domain_socket
plugin_unix_path = /tmp/frp_sock
```

## 安全配置

### 1. TLS 加密

```ini
# 服务端配置
[common]
tls_only = true
tls_cert_file = server.crt
tls_key_file = server.key
tls_trusted_ca_file = ca.crt

# 客户端配置
[common]
tls_enable = true
tls_cert_file = client.crt
tls_key_file = client.key
tls_trusted_ca_file = ca.crt
```

### 2. 基于用户的访问控制

```ini
# 服务端配置
[common]
user_conn_timeout = 10

# 客户端配置
[common]
user = test_user
meta_var1 = 123
meta_var2 = 456
```

### 3. 端口白名单

```ini
# 服务端配置
[common]
allow_ports = 2000-3000,3001,3003,4000-50000
```

### 4. HTTP 基本认证

```ini
# 客户端配置
[web_auth]
type = http
local_ip = 127.0.0.1
local_port = 8080
custom_domains = web.example.com
http_user = admin
http_pwd = admin123
```

## 部署和管理

### 1. 系统服务配置

#### systemd 服务（frps.service）

```ini
[Unit]
Description=Frp Server Service
After=network.target

[Service]
Type=simple
User=frp
Restart=on-failure
RestartSec=5s
ExecStart=/usr/local/bin/frps -c /etc/frp/frps.ini
ExecReload=/bin/kill -s HUP $MAINPID
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
```

#### systemd 服务（frpc.service）

```ini
[Unit]
Description=Frp Client Service
After=network.target

[Service]
Type=simple
User=frp
Restart=on-failure
RestartSec=5s
ExecStart=/usr/local/bin/frpc -c /etc/frp/frpc.ini
ExecReload=/bin/kill -s HUP $MAINPID
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
```

### 2. 启动和管理

```bash
# 安装服务
sudo systemctl daemon-reload
sudo systemctl enable frps
sudo systemctl enable frpc

# 启动服务
sudo systemctl start frps
sudo systemctl start frpc

# 查看状态
sudo systemctl status frps
sudo systemctl status frpc

# 查看日志
sudo journalctl -u frps -f
sudo journalctl -u frpc -f
```

### 3. Docker 部署

#### Docker Compose 配置

```yaml
version: '3.8'

services:
  frps:
    image: fatedier/frps:latest
    container_name: frps
    restart: unless-stopped
    ports:
      - "7000:7000"
      - "7500:7500"
      - "80:80"
      - "443:443"
    volumes:
      - ./frps.ini:/etc/frp/frps.ini
      - ./logs:/var/log/frp
    command: ["/usr/bin/frps", "-c", "/etc/frp/frps.ini"]

  frpc:
    image: fatedier/frpc:latest
    container_name: frpc
    restart: unless-stopped
    volumes:
      - ./frpc.ini:/etc/frp/frpc.ini
      - ./logs:/var/log/frp
    command: ["/usr/bin/frpc", "-c", "/etc/frp/frpc.ini"]
    depends_on:
      - frps
```

## 性能优化

### 1. 系统优化

```bash
# 增加文件描述符限制
echo "* soft nofile 1048576" >> /etc/security/limits.conf
echo "* hard nofile 1048576" >> /etc/security/limits.conf

# 优化网络参数
echo "net.core.rmem_max = 16777216" >> /etc/sysctl.conf
echo "net.core.wmem_max = 16777216" >> /etc/sysctl.conf
echo "net.ipv4.tcp_rmem = 4096 8192 16777216" >> /etc/sysctl.conf
echo "net.ipv4.tcp_wmem = 4096 8192 16777216" >> /etc/sysctl.conf

# 应用配置
sysctl -p
```

### 2. FRP 优化配置

```ini
# 服务端优化
[common]
max_pool_count = 10
max_ports_per_client = 10
tcp_keepalive = 7200

# 客户端优化
[common]
pool_count = 5
tcp_mux = true
login_fail_exit = false
heartbeat_interval = 30
heartbeat_timeout = 90
```

## 故障排除

### 1. 常见问题

#### 连接失败

```bash
# 检查防火墙
sudo ufw status
sudo firewall-cmd --list-ports

# 检查端口监听
netstat -tuln | grep 7000
ss -tuln | grep 7000

# 检查 SELinux
getenforce
```

#### 性能问题

```bash
# 查看系统资源
top -p $(pgrep frps)
htop -p $(pgrep frps)

# 查看网络连接
netstat -an | grep :7000
ss -an | grep :7000

# 查看文件描述符
lsof -p $(pgrep frps) | wc -l
```

### 2. 调试技巧

```bash
# 启用详细日志
frps -c frps.ini -L trace

# 使用 tcpdump 抓包
tcpdump -i any -w frp.pcap host your_server_ip and port 7000

# 查看进程信息
ps aux | grep frp
```

### 3. 配置验证

```bash
# 验证配置文件
frps -c frps.ini -t
frpc -c frpc.ini -t

# 测试连接
telnet your_server_ip 7000
nc -v your_server_ip 7000
```

## 实际应用场景

### 1. 家庭服务器

```ini
# 家庭 NAS 访问
[nas_web]
type = http
local_ip = 192.168.1.100
local_port = 5000
custom_domains = nas.example.com

[nas_ssh]
type = tcp
local_ip = 192.168.1.100
local_port = 22
remote_port = 2222
```

### 2. 开发环境

```ini
# 本地开发服务器
[dev_web]
type = http
local_ip = 127.0.0.1
local_port = 3000
custom_domains = dev.example.com

[dev_api]
type = http
local_ip = 127.0.0.1
local_port = 8080
custom_domains = api.example.com
```

### 3. 远程办公

```ini
# 远程桌面
[rdp]
type = tcp
local_ip = 192.168.1.10
local_port = 3389
remote_port = 3389

# 文件共享
[smb]
type = tcp
local_ip = 192.168.1.10
local_port = 445
remote_port = 445
```

## 总结

FRP 作为一个功能强大的内网穿透工具，具有以下优势：

1. **简单易用**: 配置简单，部署方便
2. **功能丰富**: 支持多种协议和高级功能
3. **性能优秀**: 基于 Go 语言，性能稳定
4. **活跃开发**: 持续更新和改进
5. **社区支持**: 活跃的开源社区

选择 FRP 时需要考虑：

- 网络环境和带宽需求
- 安全性要求
- 管理和监控需求
- 成本考虑

---

*作者：meimeitou*  
