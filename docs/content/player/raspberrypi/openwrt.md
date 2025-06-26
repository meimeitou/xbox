+++
title = 'Openwrt家庭防火墙'
+++


世界那么大我想去看看！！

## 项目概述

使用Raspberry Pi搭建基于OpenWrt的家庭防火墙，提供网络安全防护、流量控制、内容过滤等功能，打造专业级的家庭网络安全中心。

## 硬件要求

### 推荐配置

| 组件 | 推荐规格 | 价格范围 | 说明 |
|------|----------|----------|------|
| **主板** | Raspberry Pi 4B (4GB/8GB) | | 8GB版本处理能力更强 |
| **存储** | 32GB+ microSD卡 (A2) | | 高速存储，推荐SanDisk Extreme |
| **网络接口** | USB 3.0 千兆网卡 | | 实现双网口配置 |
| **电源** | 官方5V 3A电源适配器 | 10 | 稳定供电确保网络不中断 |
| **散热** | 主动散热风扇套装 | 10-15 | 长时间运行必需 |
| **机箱** | 金属散热机箱 | | 保护和散热 |
| **网线** | CAT6网线若干 |  | 连接路由器和设备 |

### 网络拓扑规划

```txt
互联网 -> 光猫 -> Raspberry Pi (OpenWrt防火墙) -> 交换机 -> 内网设备
```

## OpenWrt系统安装

### 第一步：下载OpenWrt镜像

```bash
# 官方下载地址
wget https://downloads.openwrt.org/releases/22.03.5/targets/bcm27xx/bcm2711/openwrt-22.03.5-bcm27xx-bcm2711-rpi-4-ext4-factory.img.gz

# 解压镜像
gunzip openwrt-22.03.5-bcm27xx-bcm2711-rpi-4-ext4-factory.img.gz
```

### 第二步：烧录到SD卡

```bash
# 使用dd命令烧录 (Linux/macOS)
sudo dd if=openwrt-22.03.5-bcm27xx-bcm2711-rpi-4-ext4-factory.img of=/dev/sdX bs=4M status=progress sync

# 或使用Raspberry Pi Imager (Windows/macOS/Linux)
# 选择"Use custom image"，选择下载的镜像文件
```

### 第三步：初始配置

```bash
# 首次启动后通过以太网连接
# 默认IP: 192.168.1.1
# 用户名: root (无密码)

# SSH连接
ssh root@192.168.1.1

# 设置root密码
passwd root
```

## 网络接口配置

### 配置双网口设置

```bash
# 编辑网络配置文件
vi /etc/config/network
```

```uci
# /etc/config/network 配置示例
config interface 'loopback'
    option ifname 'lo'
    option proto 'static'
    option ipaddr '127.0.0.1'
    option netmask '255.0.0.0'

# WAN接口 (连接上级路由器或光猫)
config interface 'wan'
    option ifname 'eth0'
    option proto 'dhcp'
    option peerdns '0'
    option dns '114.114.114.114 8.8.8.8'

# LAN接口 (内网)
config interface 'lan'
    option ifname 'eth1'  # USB网卡
    option proto 'static'
    option ipaddr '192.168.100.1'
    option netmask '255.255.255.0'
    option ip6assign '60'

# DMZ区域 (可选)
config interface 'dmz'
    option proto 'static'
    option ipaddr '192.168.200.1'
    option netmask '255.255.255.0'
```

### 重启网络服务

```bash
# 重启网络
/etc/init.d/network restart

# 查看接口状态
ip addr show
```

## 防火墙规则配置

### 基础防火墙配置

```bash
# 编辑防火墙配置
vi /etc/config/firewall
```

```uci
# /etc/config/firewall 基础配置
config defaults
    option syn_flood '1'
    option input 'REJECT'
    option output 'ACCEPT'
    option forward 'REJECT'

# LAN区域
config zone
    option name 'lan'
    option input 'ACCEPT'
    option output 'ACCEPT'
    option forward 'ACCEPT'
    option network 'lan'

# WAN区域
config zone
    option name 'wan'
    option input 'REJECT'
    option output 'ACCEPT'
    option forward 'REJECT'
    option masq '1'
    option mtu_fix '1'
    option network 'wan'

# DMZ区域
config zone
    option name 'dmz'
    option input 'REJECT'
    option output 'ACCEPT'
    option forward 'REJECT'
    option network 'dmz'

# 允许LAN到WAN的转发
config forwarding
    option src 'lan'
    option dest 'wan'

# SSH访问规则
config rule
    option name 'Allow-SSH'
    option src 'lan'
    option proto 'tcp'
    option dest_port '22'
    option target 'ACCEPT'

# Web管理界面访问
config rule
    option name 'Allow-LUCI'
    option src 'lan'
    option proto 'tcp'
    option dest_port '80'
    option target 'ACCEPT'

# HTTPS管理界面
config rule
    option name 'Allow-HTTPS'
    option src 'lan'
    option proto 'tcp'
    option dest_port '443'
    option target 'ACCEPT'

# DNS服务
config rule
    option name 'Allow-DNS'
    option src 'lan'
    option proto 'tcpudp'
    option dest_port '53'
    option target 'ACCEPT'

# DHCP服务
config rule
    option name 'Allow-DHCP'
    option src 'lan'
    option proto 'udp'
    option dest_port '67-68'
    option target 'ACCEPT'
```

### 高级安全规则

```bash
# 防DDoS规则
iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --set --name SSH
iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --update --seconds 60 --hitcount 4 --name SSH -j DROP

# 防端口扫描
iptables -A INPUT -m recent --name portscan --rcheck --seconds 86400 -j DROP
iptables -A FORWARD -m recent --name portscan --rcheck --seconds 86400 -j DROP

# SYN flood防护
iptables -A INPUT -p tcp --syn -m limit --limit 1/s --limit-burst 3 -j RETURN
iptables -A INPUT -p tcp --syn -j DROP

# Ping flood防护
iptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 1/s -j ACCEPT
iptables -A INPUT -p icmp --icmp-type echo-request -j DROP
```

## 内容过滤和广告拦截

### 安装AdBlock Plus

```bash
# 更新软件包列表
opkg update

# 安装adblock
opkg install adblock luci-app-adblock

# 安装依赖
opkg install ca-bundle ca-certificates
```

### 配置AdBlock

```bash
# 编辑adblock配置
vi /etc/config/adblock
```

```uci
config adblock 'global'
    option adb_enabled '1'
    option adb_debug '0'
    option adb_forcedns '1'
    option adb_forcesafesearch '1'
    option adb_backup '1'
    option adb_backupdir '/tmp/adblock'
    
config source 'adaway'
    option enabled '1'
    option adb_src 'https://raw.githubusercontent.com/AdguardTeam/AdguardFilters/master/BaseFilter/sections/adservers.txt'
    option adb_src_rset 'BEGIN{FS="[|^]"}{print $1}'

config source 'disconnect'
    option enabled '1'
    option adb_src 'https://s3.amazonaws.com/lists.disconnect.me/simple_tracking.txt'

config source 'yoyo'
    option enabled '1'
    option adb_src 'https://pgl.yoyo.org/adservers/serverlist.php?hostformat=hosts&showintro=1&mimetype=plaintext'
```

### 启动AdBlock服务

```bash
# 启动adblock服务
/etc/init.d/adblock enable
/etc/init.d/adblock start

# 手动更新广告列表
/etc/init.d/adblock reload
```

## 流量控制和QoS

### 安装SQM包

```bash
# 安装SQM (Smart Queue Management)
opkg install luci-app-sqm

# 重启LuCI
/etc/init.d/uhttpd restart
```

### 配置QoS规则

```bash
# 编辑QoS配置
vi /etc/config/sqm
```

```uci
config queue 'eth0'
    option enabled '1'
    option interface 'eth0'
    option download '100000'  # 下载带宽 (kbps)
    option upload '20000'     # 上传带宽 (kbps)
    option script 'piece_of_cake.qos'
    option qdisc 'cake'
    option linklayer 'ethernet'
```

### 带宽限制规则

```bash
# 创建流量控制脚本
cat > /etc/tc-setup.sh << 'EOF'
#!/bin/sh

# 清除现有规则
tc qdisc del dev eth1 root 2>/dev/null
tc qdisc del dev eth1 ingress 2>/dev/null

# 创建根队列
tc qdisc add dev eth1 root handle 1: htb default 30

# 创建主类
tc class add dev eth1 parent 1: classid 1:1 htb rate 100mbit

# 高优先级流量 (40%)
tc class add dev eth1 parent 1:1 classid 1:10 htb rate 40mbit ceil 100mbit prio 1

# 正常流量 (50%)
tc class add dev eth1 parent 1:1 classid 1:20 htb rate 50mbit ceil 80mbit prio 2

# 低优先级流量 (10%)
tc class add dev eth1 parent 1:1 classid 1:30 htb rate 10mbit ceil 30mbit prio 3

# 流量分类规则
tc filter add dev eth1 protocol ip parent 1:0 prio 1 u32 match ip sport 22 0xffff flowid 1:10
tc filter add dev eth1 protocol ip parent 1:0 prio 1 u32 match ip sport 53 0xffff flowid 1:10
tc filter add dev eth1 protocol ip parent 1:0 prio 2 u32 match ip sport 80 0xffff flowid 1:20
tc filter add dev eth1 protocol ip parent 1:0 prio 2 u32 match ip sport 443 0xffff flowid 1:20
EOF

chmod +x /etc/tc-setup.sh

# 添加到启动脚本
echo '/etc/tc-setup.sh' >> /etc/rc.local
```

## 入侵检测系统

### 安装Suricata

```bash
# 由于OpenWrt空间限制，使用轻量级的入侵检测
opkg install snort

# 或者安装自定义监控脚本
cat > /usr/bin/ids-monitor << 'EOF'
#!/bin/sh

LOG_FILE="/var/log/security.log"
ALERT_THRESHOLD=10

# 监控异常连接
netstat -an | grep ":22 " | grep -c "ESTABLISHED" > /tmp/ssh_conn
ssh_count=$(cat /tmp/ssh_conn)

if [ $ssh_count -gt $ALERT_THRESHOLD ]; then
    echo "$(date): 警告 - SSH连接数异常: $ssh_count" >> $LOG_FILE
    logger "安全警告: SSH连接数过多"
fi

# 监控端口扫描
dmesg | tail -100 | grep -i "scan" && {
    echo "$(date): 检测到端口扫描行为" >> $LOG_FILE
}

# 检查暴力破解尝试
grep "authentication failure" /var/log/messages | tail -5 | while read line; do
    echo "$(date): 认证失败: $line" >> $LOG_FILE
done
EOF

chmod +x /usr/bin/ids-monitor

# 添加到cron任务
echo "*/5 * * * * /usr/bin/ids-monitor" >> /etc/crontabs/root
```

## VPN服务配置

### 安装OpenVPN服务器

```bash
# 安装OpenVPN
opkg install openvpn-openssl luci-app-openvpn

# 生成证书和密钥
openvpn --genkey --secret /etc/openvpn/static.key

# 创建服务器配置
cat > /etc/openvpn/server.conf << 'EOF'
port 1194
proto udp
dev tun
server 10.8.0.0 255.255.255.0
ifconfig-pool-persist ipp.txt
keepalive 10 120
cipher AES-256-CBC
persist-key
persist-tun
status openvpn-status.log
verb 3
secret static.key
EOF
```

### WireGuard配置 (推荐)

```bash
# 安装WireGuard
opkg install luci-app-wireguard wireguard-tools kmod-wireguard

# 生成服务器密钥对
cd /etc/wireguard
wg genkey | tee server_private.key | wg pubkey > server_public.key

# 创建服务器配置
cat > /etc/config/wireguard_server << 'EOF'
[Interface]
PrivateKey = $(cat server_private.key)
Address = 10.0.0.1/24
ListenPort = 51820
SaveConfig = true

# 客户端1
[Peer]
PublicKey = CLIENT1_PUBLIC_KEY
AllowedIPs = 10.0.0.2/32

# 客户端2
[Peer]
PublicKey = CLIENT2_PUBLIC_KEY
AllowedIPs = 10.0.0.3/32
EOF
```

## 监控和日志

### 系统监控配置

```bash
# 安装collectd监控
opkg install collectd collectd-mod-cpu collectd-mod-memory collectd-mod-network

# 配置collectd
cat > /etc/collectd.conf << 'EOF'
Hostname "openwrt-firewall"
FQDNLookup true
BaseDir "/var/lib/collectd"
PIDFile "/var/run/collectd.pid"
PluginDir "/usr/lib/collectd"

LoadPlugin cpu
LoadPlugin memory
LoadPlugin network
LoadPlugin interface
LoadPlugin load
LoadPlugin disk

<Plugin cpu>
    ReportByCpu true
    ReportByState true
    ValuesPercentage true
</Plugin>

<Plugin memory>
    ValuesAbsolute true
    ValuesPercentage false
</Plugin>

<Plugin network>
    Server "192.168.100.100" "25826"
</Plugin>
EOF
```

### 日志配置

```bash
# 配置系统日志
cat > /etc/config/system << 'EOF'
config system
    option hostname 'OpenWrt-Firewall'
    option log_size '64'
    option log_ip '192.168.100.100'
    option log_port '514'
    option log_proto 'udp'
    
config timeserver
    option enabled '1'
    list server 'ntp.aliyun.com'
    list server 'time.nist.gov'
EOF

# 配置日志轮转
cat > /etc/logrotate.d/openwrt << 'EOF'
/var/log/messages {
    size 10M
    rotate 5
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
}

/var/log/security.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
}
EOF
```

## Web管理界面定制

### 安装额外的LuCI模块

```bash
# 安装常用管理模块
opkg install luci-app-statistics  # 系统统计
opkg install luci-app-nlbwmon     # 带宽监控
opkg install luci-app-upnp        # UPnP支持
opkg install luci-app-ddns        # 动态DNS
opkg install luci-app-watchcat    # 连接监控

# 重启Web服务
/etc/init.d/uhttpd restart
```

### 自定义监控面板

```bash
# 创建自定义状态页面
cat > /www/status.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>防火墙状态监控</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .status-box { border: 1px solid #ddd; padding: 10px; margin: 10px 0; }
        .good { background-color: #d4edda; }
        .warning { background-color: #fff3cd; }
        .error { background-color: #f8d7da; }
    </style>
</head>
<body>
    <h1>家庭防火墙状态</h1>
    
    <div class="status-box good">
        <h3>系统状态</h3>
        <p>运行时间: <span id="uptime"></span></p>
        <p>CPU使用率: <span id="cpu"></span></p>
        <p>内存使用: <span id="memory"></span></p>
    </div>
    
    <div class="status-box good">
        <h3>网络状态</h3>
        <p>WAN状态: <span id="wan-status"></span></p>
        <p>已拦截广告: <span id="blocked-ads"></span></p>
        <p>活跃连接: <span id="connections"></span></p>
    </div>
    
    <script>
        function updateStatus() {
            // 这里可以通过AJAX获取实时数据
            fetch('/cgi-bin/luci/admin/status/overview')
                .then(response => response.text())
                .then(data => {
                    // 解析并更新状态信息
                });
        }
        
        setInterval(updateStatus, 30000); // 30秒更新一次
        updateStatus();
    </script>
</body>
</html>
EOF
```

## 性能优化

### 系统调优

```bash
# 编辑系统参数
cat >> /etc/sysctl.conf << 'EOF'
# 网络性能优化
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 65536 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# 连接跟踪优化
net.netfilter.nf_conntrack_max = 32768
net.netfilter.nf_conntrack_tcp_timeout_established = 1800

# 转发性能优化
net.ipv4.ip_forward = 1
net.ipv4.conf.all.forwarding = 1

# 安全加固
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.all.secure_redirects = 0
net.ipv4.conf.all.send_redirects = 0
EOF

# 应用设置
sysctl -p
```

### 存储优化

```bash
# 配置临时文件系统
echo 'tmpfs /tmp tmpfs defaults,noatime,mode=1777 0 0' >> /etc/fstab
echo 'tmpfs /var/log tmpfs defaults,noatime,mode=0755,size=32M 0 0' >> /etc/fstab

# 优化SD卡使用
echo 'deadline' > /sys/block/mmcblk0/queue/scheduler
echo '1' > /sys/block/mmcblk0/queue/iosched/fifo_batch
```

## 备份和恢复

### 配置备份脚本

```bash
# 创建备份脚本
cat > /usr/bin/backup-config << 'EOF'
#!/bin/sh

BACKUP_DIR="/tmp/backup"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="openwrt_backup_${DATE}.tar.gz"

mkdir -p $BACKUP_DIR

# 备份关键配置文件
tar -czf $BACKUP_DIR/$BACKUP_FILE \
    /etc/config/ \
    /etc/openvpn/ \
    /etc/wireguard/ \
    /etc/firewall.user \
    /etc/crontabs/ \
    /etc/dropbear/ \
    /root/.ssh/

echo "备份完成: $BACKUP_DIR/$BACKUP_FILE"

# 通过SCP传输到远程服务器 (可选)
# scp $BACKUP_DIR/$BACKUP_FILE user@backup-server:/path/to/backup/

# 清理老备份 (保留最近5个)
cd $BACKUP_DIR
ls -t openwrt_backup_*.tar.gz | tail -n +6 | xargs rm -f
EOF

chmod +x /usr/bin/backup-config

# 添加定时备份
echo "0 2 * * 0 /usr/bin/backup-config" >> /etc/crontabs/root
```

### 配置恢复

```bash
# 恢复配置脚本
cat > /usr/bin/restore-config << 'EOF'
#!/bin/sh

if [ -z "$1" ]; then
    echo "用法: $0 <backup_file.tar.gz>"
    exit 1
fi

BACKUP_FILE="$1"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "备份文件不存在: $BACKUP_FILE"
    exit 1
fi

echo "开始恢复配置..."
tar -xzf "$BACKUP_FILE" -C /

echo "重启相关服务..."
/etc/init.d/network restart
/etc/init.d/firewall restart
/etc/init.d/dnsmasq restart

echo "配置恢复完成，建议重启系统"
EOF

chmod +x /usr/bin/restore-config
```

## 安全加固

### SSH安全配置

```bash
# 配置SSH安全设置
cat > /etc/dropbear/dropbear_ed25519_host_key << 'EOF'
# 生成新的主机密钥
dropbearkey -t ed25519 -f /etc/dropbear/dropbear_ed25519_host_key
EOF

# 禁用密码登录，仅允许密钥认证
sed -i 's/#PasswordAuth yes/PasswordAuth no/' /etc/config/dropbear
sed -i 's/#PubkeyAuth yes/PubkeyAuth yes/' /etc/config/dropbear
```

### 访问控制

```bash
# 创建访问控制脚本
cat > /etc/access-control.sh << 'EOF'
#!/bin/sh

# 管理员IP白名单
ADMIN_IPS="192.168.100.10 192.168.100.11"

# 允许管理员访问所有端口
for ip in $ADMIN_IPS; do
    iptables -I INPUT -s $ip -j ACCEPT
done

# 限制其他用户访问管理端口
iptables -A INPUT -p tcp --dport 22 -j DROP
iptables -A INPUT -p tcp --dport 80 -j DROP
iptables -A INPUT -p tcp --dport 443 -j DROP

# 工作时间访问控制 (9:00-18:00)
iptables -A FORWARD -m time --timestart 09:00 --timestop 18:00 --weekdays Mon,Tue,Wed,Thu,Fri -j ACCEPT
iptables -A FORWARD -m time --timestart 19:00 --timestop 23:59 --weekdays Mon,Tue,Wed,Thu,Fri -j DROP
EOF

chmod +x /etc/access-control.sh
```

## 故障排除

### 常见问题解决

#### 1. 网络连接问题

```bash
# 检查接口状态
ip link show
ip addr show

# 检查路由表
ip route show

# 测试连通性
ping -c 4 8.8.8.8
ping -c 4 192.168.100.1

# 检查防火墙规则
iptables -L -n -v
```

#### 2. 性能问题

```bash
# 查看系统负载
uptime
top
free

# 查看网络统计
cat /proc/net/dev
ss -tuln

# 查看连接跟踪
cat /proc/net/nf_conntrack | wc -l
```

#### 3. 服务问题

```bash
# 检查关键服务状态
/etc/init.d/network status
/etc/init.d/firewall status
/etc/init.d/dnsmasq status

# 查看日志
logread | tail -50
dmesg | tail -20
```

### 诊断脚本

```bash
cat > /usr/bin/firewall-diagnostic << 'EOF'
#!/bin/sh

echo "=== 防火墙诊断报告 ==="
echo "时间: $(date)"
echo

echo "=== 系统信息 ==="
uname -a
uptime
free -h
df -h

echo -e "\n=== 网络接口 ==="
ip addr show

echo -e "\n=== 路由表 ==="
ip route show

echo -e "\n=== 防火墙规则 ==="
iptables -L -n

echo -e "\n=== 活跃连接 ==="
ss -tuln

echo -e "\n=== 系统日志 (最近10条) ==="
logread | tail -10

echo -e "\n=== 诊断完成 ==="
EOF

chmod +x /usr/bin/firewall-diagnostic
```

## 性能监控

### 网络流量监控

```bash
# 安装带宽监控工具
opkg install luci-app-nlbwmon

# 创建流量统计脚本
cat > /usr/bin/traffic-stats << 'EOF'
#!/bin/sh

# 获取接口流量统计
get_interface_stats() {
    local interface=$1
    local rx_bytes=$(cat /sys/class/net/$interface/statistics/rx_bytes)
    local tx_bytes=$(cat /sys/class/net/$interface/statistics/tx_bytes)
    
    echo "接口 $interface:"
    echo "  接收: $(( $rx_bytes / 1024 / 1024 )) MB"
    echo "  发送: $(( $tx_bytes / 1024 / 1024 )) MB"
}

echo "=== 网络流量统计 ==="
get_interface_stats eth0
get_interface_stats eth1

# 连接数统计
echo -e "\n=== 连接统计 ==="
echo "总连接数: $(cat /proc/net/nf_conntrack | wc -l)"
echo "TCP连接: $(ss -t | wc -l)"
echo "UDP连接: $(ss -u | wc -l)"
EOF

chmod +x /usr/bin/traffic-stats
```

## 总结

这个基于Raspberry Pi和OpenWrt的家庭防火墙方案提供了：

### 核心功能

- ✅ **专业级防火墙** - 多区域安全策略
- ✅ **内容过滤** - 广告拦截和恶意网站防护
- ✅ **流量控制** - QoS和带宽管理
- ✅ **VPN服务** - 安全远程访问
- ✅ **入侵检测** - 实时安全监控
- ✅ **Web管理** - 直观的管理界面

### 优势特点

- **成本效益**: 总成本约$150，相比商业防火墙节省数千元
- **高度定制**: 完全开源，可根据需求定制功能
- **学习价值**: 深入理解网络安全原理
- **社区支持**: OpenWrt拥有活跃的开发者社区

### 适用场景

- 家庭网络安全防护
- 小型办公室网络
- 技术学习和实验
- 网络安全研究
