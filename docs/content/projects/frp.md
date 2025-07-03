+++
title = 'frp内外穿透'
+++

- [什么是 FRP？](#什么是-frp)
- [基本架构](#基本架构)
- [安装部署](#安装部署)
- [常用场景配置示例](#常用场景配置示例)
  - [场景一：SSH 远程连接](#场景一ssh-远程连接)
  - [场景二：Web 服务访问](#场景二web-服务访问)
  - [场景三：HTTPS 服务](#场景三https-服务)
  - [场景四：文件共享服务](#场景四文件共享服务)
  - [场景五：数据库远程访问](#场景五数据库远程访问)
- [安全配置建议](#安全配置建议)
  - [1. 启用身份验证](#1-启用身份验证)
  - [2. 限制客户端权限](#2-限制客户端权限)
  - [3. 启用加密和压缩](#3-启用加密和压缩)
- [启动和管理](#启动和管理)
  - [启动服务](#启动服务)
  - [系统服务配置](#系统服务配置)
- [监控和日志](#监控和日志)
  - [启用日志记录](#启用日志记录)
  - [Web 管理界面](#web-管理界面)
- [总结](#总结)

FRP (Fast Reverse Proxy) 是一个高性能的反向代理应用，可以帮助您将内网服务通过具有公网 IP 的服务器暴露到互联网上。本文将介绍 FRP 的基本使用方法和几个常用场景的配置示例。

## 什么是 FRP？

FRP 是一个专注于内网穿透的高性能的反向代理应用，支持 TCP、UDP、HTTP、HTTPS 等多种协议。它可以让您在没有公网 IP 的情况下，将内网服务暴露到公网，非常适合开发调试、远程访问等场景。

## 基本架构

FRP 由两部分组成：

- **frps (server)**：运行在具有公网 IP 的服务器上
- **frpc (client)**：运行在内网环境中

## 安装部署

1. 从 [GitHub Release](https://github.com/fatedier/frp/releases) 下载对应平台的二进制文件
2. 解压到目标目录
3. 根据需求配置服务端和客户端

## 常用场景配置示例

### 场景一：SSH 远程连接

**使用场景**：远程访问家里或办公室的 Linux 服务器

**服务端配置 (frps.toml)**：

```toml
bindPort = 7000
```

**客户端配置 (frpc.toml)**：

```toml
serverAddr = "your-server-ip"
serverPort = 7000

[[proxies]]
name = "ssh"
type = "tcp"
localIP = "127.0.0.1"
localPort = 22
remotePort = 6000
```

**使用方法**：

```bash
ssh -p 6000 username@your-server-ip
```

### 场景二：Web 服务访问

**使用场景**：将本地开发的 Web 应用暴露到公网进行测试

**服务端配置 (frps.toml)**：

```toml
bindPort = 7000
vhostHTTPPort = 8080
```

**客户端配置 (frpc.toml)**：

```toml
serverAddr = "your-server-ip"
serverPort = 7000

[[proxies]]
name = "web"
type = "http"
localIP = "127.0.0.1"
localPort = 3000
customDomains = ["your-domain.com"]
```

**访问方式**：

```txt
http://your-domain.com:8080
```

### 场景三：HTTPS 服务

**使用场景**：需要 HTTPS 访问的 Web 服务

**服务端配置 (frps.toml)**：

```toml
bindPort = 7000
vhostHTTPSPort = 8443
```

**客户端配置 (frpc.toml)**：

```toml
serverAddr = "your-server-ip"
serverPort = 7000

[[proxies]]
name = "web-secure"
type = "https"
localIP = "127.0.0.1"
localPort = 443
customDomains = ["secure.your-domain.com"]
```

### 场景四：文件共享服务

**使用场景**：临时共享文件或搭建简单的文件服务器

**服务端配置 (frps.toml)**：

```toml
bindPort = 7000
```

**客户端配置 (frpc.toml)**：

```toml
serverAddr = "your-server-ip"
serverPort = 7000

[[proxies]]
name = "file-server"
type = "tcp"
localIP = "127.0.0.1"
localPort = 8000
remotePort = 8080

# 启用静态文件服务
[proxies.file-server.plugin]
type = "static_file"
localPath = "/path/to/your/files"
stripPrefix = "static"
httpUser = "admin"
httpPassword = "password"
```

### 场景五：数据库远程访问

**使用场景**：远程访问内网数据库进行开发或管理

**服务端配置 (frps.toml)**：

```toml
bindPort = 7000
```

**客户端配置 (frpc.toml)**：

```toml
serverAddr = "your-server-ip"
serverPort = 7000

# MySQL 数据库
[[proxies]]
name = "mysql"
type = "tcp"
localIP = "127.0.0.1"
localPort = 3306
remotePort = 3307

# PostgreSQL 数据库
[[proxies]]
name = "postgres"
type = "tcp"
localIP = "127.0.0.1"
localPort = 5432
remotePort = 5433
```

## 安全配置建议

### 1. 启用身份验证

**服务端配置**：

```toml
bindPort = 7000
auth.method = "token"
auth.token = "your-secret-token"
```

**客户端配置**：

```toml
serverAddr = "your-server-ip"
serverPort = 7000
auth.method = "token"
auth.token = "your-secret-token"
```

### 2. 限制客户端权限

```toml
# 服务端配置
bindPort = 7000
allowPorts = [6000-6100, 8080, 8443]
maxPoolCount = 5
```

### 3. 启用加密和压缩

```toml
# 客户端配置
transport.useEncryption = true
transport.useCompression = true
```

## 启动和管理

### 启动服务

**服务端**：

```bash
./frps -c frps.toml
```

**客户端**：

```bash
./frpc -c frpc.toml
```

### 系统服务配置

创建 systemd 服务文件 `/etc/systemd/system/frps.service`：

```ini
[Unit]
Description=FRP Server
After=network.target

[Service]
Type=simple
User=frp
Group=frp
WorkingDirectory=/opt/frp
ExecStart=/opt/frp/frps -c /opt/frp/frps.toml
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## 监控和日志

### 启用日志记录

```toml
# 服务端配置
log.to = "./frps.log"
log.level = "info"
log.maxDays = 30

# 客户端配置
log.to = "./frpc.log"
log.level = "info"
log.maxDays = 30
```

### Web 管理界面

```toml
# 服务端配置
webServer.addr = "0.0.0.0"
webServer.port = 7500
webServer.user = "admin"
webServer.password = "admin"
```

访问 `http://your-server-ip:7500` 查看状态。

## 总结

FRP 是一个功能强大且易于使用的内网穿透工具，通过合理的配置可以满足各种场景的需求。在使用过程中要注意：

1. **安全性**：始终使用强密码和加密连接
2. **性能**：根据带宽情况调整连接数和缓冲区大小
3. **稳定性**：配置自动重启和监控机制
4. **合规性**：确保使用符合相关法律法规
