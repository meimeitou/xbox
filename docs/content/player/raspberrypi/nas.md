+++
title = '家庭NAS'
description = '使用Raspberry Pi搭建家庭网络存储'
+++

百度网盘要我35大洋一个月，家里网络也不行，忍不了必须得上方案，家庭NAS（Network Attached Storage）。

## 概述

本方案将详细介绍如何将Raspberry Pi打造成功能完整的家庭网络存储中心。

## 硬件要求

### 推荐配置

| 组件 | 推荐规格 | 价格范围 | 说明 |
|------|----------|----------|------|
| **主板** | Raspberry Pi 4B (4GB/8GB) | 300多大洋 | 8GB版本更适合多用户访问 |
| **存储** | 32GB+ microSD卡 (Class 10) | 50 | 系统盘，推荐A2规格 |
| **外置硬盘** | 2.5"/3.5" USB 3.0硬盘 | 看需求 | 数据存储，建议2TB+ |
| **电源** | 官方电源适配器 5V 3A | 20 | 稳定供电很重要 |
| **网络** | 有线网络连接 | - | Wi-Fi可用但有线更稳定 |
| **散热** | 散热片+风扇 | 50 | 长时间运行必需 |
| **机箱** | 带风扇的金属机箱 | 100 | 保护和散热 |

### 可选配置

- **UPS不间断电源**: 防止突然断电损坏数据
- **多个硬盘**: 实现RAID冗余备份
- **网络交换机**: 扩展网络连接

## 软件方案选择

### 方案一：OpenMediaVault (推荐)

**特点**: 专业NAS系统，Web界面友好，功能全面

**优势**:

- 基于Debian，稳定可靠
- 丰富的插件生态
- 支持多种文件系统
- 完善的用户权限管理
- 支持RAID、快照等高级功能

### 方案二：树莓派官方系统 + Samba

**特点**: 轻量级方案，适合简单需求

**优势**:

- 系统资源占用少
- 配置相对简单
- 可以同时运行其他服务

### 方案三：Docker容器化方案

**特点**: 模块化部署，便于管理和扩展

## 详细实施步骤

### 第一步：系统安装 (以OpenMediaVault为例)

#### 1. 下载和烧录系统

```bash
# 下载官方镜像
wget https://www.openmediavault.org/download

# 使用Raspberry Pi Imager烧录到SD卡
# 或使用dd命令
sudo dd if=openmediavault-x.x.x-raspberry-pi.img of=/dev/sdX bs=4M status=progress
```

#### 2. 初始设置

```bash
# 启动后通过SSH连接 (默认用户名: root, 密码: openmediavault)
ssh root@树莓派IP地址

# 更新系统
apt update && apt upgrade -y

# 修改默认密码
passwd root
```

### 第二步：存储配置

#### 1. 硬盘分区和格式化

```bash
# 查看硬盘设备
lsblk

# 使用fdisk分区 (以/dev/sda为例)
fdisk /dev/sda
# n (新建分区) -> p (主分区) -> 1 (分区号) -> 回车 -> 回车 -> w (写入)

# 格式化为ext4文件系统
mkfs.ext4 /dev/sda1

# 创建挂载点
mkdir /srv/dev-disk-by-uuid-硬盘UUID
```

#### 2. Web界面配置

```txt
访问地址: http://树莓派IP地址
默认用户名: admin
默认密码: openmediavault
```

**配置步骤**:

1. **存储** -> **磁盘** -> 扫描硬盘
2. **存储** -> **文件系统** -> 创建文件系统
3. **存储** -> **文件系统** -> 挂载文件系统

### 第三步：共享服务配置

#### 1. SMB/CIFS 共享 (Windows兼容)

```bash
# Web界面操作
# 服务 -> SMB/CIFS -> 启用
# 服务 -> SMB/CIFS -> 共享 -> 添加
```

**配置参数**:

```ini
# SMB配置示例
[共享名称]
path = /srv/dev-disk-by-uuid-xxx/共享文件夹
valid users = 用户名
read only = no
browseable = yes
```

#### 2. NFS 共享 (Linux兼容)

```bash
# 启用NFS服务
# 服务 -> NFS -> 启用
# 服务 -> NFS -> 共享 -> 添加
```

#### 3. FTP服务

```bash
# 启用FTP
# 服务 -> FTP -> 启用
# 配置被动模式端口范围
```

### 第四步：用户和权限管理

#### 1. 创建用户

```bash
# Web界面: 访问权限管理 -> 用户 -> 添加
# 设置用户名、密码、主目录等
```

#### 2. 创建用户组

```bash
# 访问权限管理 -> 组 -> 添加
# 将用户加入相应组
```

#### 3. 设置共享权限

```bash
# 访问权限管理 -> 共享文件夹 -> 编辑
# 配置读写权限和访问用户
```

### 第五步：高级功能配置

#### 1. RAID配置 (多硬盘)

```bash
# Web界面: 存储 -> RAID管理
# 创建RAID1 (镜像) 或 RAID5 (分布式奇偶校验)

# 命令行创建RAID1示例
mdadm --create --verbose /dev/md0 --level=1 --raid-devices=2 /dev/sda1 /dev/sdb1
```

#### 2. 快照和备份

```bash
# 安装rsync插件
# 系统 -> 插件 -> 搜索rsync -> 安装

# 配置定时备份任务
# 服务 -> Rsync -> 任务 -> 添加
```

#### 3. DLNA媒体服务器

```bash
# 安装minidlna插件
# 系统 -> 插件 -> 搜索minidlna -> 安装

# 配置媒体目录
# 服务 -> DLNA -> 启用并配置
```

## 性能优化

### 1. 系统优化

```bash
# 调整虚拟内存
echo 'vm.swappiness=10' >> /etc/sysctl.conf

# 优化网络参数
echo 'net.core.rmem_max = 16777216' >> /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' >> /etc/sysctl.conf

# 禁用不必要的服务
systemctl disable bluetooth
systemctl disable avahi-daemon
```

### 2. 存储优化

```bash
# 挂载选项优化 (在/etc/fstab中)
UUID=硬盘UUID /挂载点 ext4 defaults,noatime,errors=remount-ro 0 1

# 启用文件系统缓存
echo 'vm.dirty_ratio = 15' >> /etc/sysctl.conf
echo 'vm.dirty_background_ratio = 5' >> /etc/sysctl.conf
```

### 3. 网络优化

```bash
# 调整TCP缓冲区
echo 'net.ipv4.tcp_rmem = 4096 65536 16777216' >> /etc/sysctl.conf
echo 'net.ipv4.tcp_wmem = 4096 65536 16777216' >> /etc/sysctl.conf
```

## 安全配置

### 1. 防火墙设置

```bash
# 安装ufw
apt install ufw

# 配置基本规则
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80/tcp    # Web管理界面
ufw allow 139,445/tcp  # SMB
ufw allow 2049/tcp  # NFS
ufw enable
```

### 2. SSH安全

```bash
# 编辑SSH配置
nano /etc/ssh/sshd_config

# 修改配置
Port 2222  # 更改默认端口
PermitRootLogin no  # 禁止root直接登录
PasswordAuthentication yes  # 根据需要启用/禁用密码认证
```

### 3. 定期更新

```bash
# 创建自动更新脚本
cat > /usr/local/bin/auto-update.sh << 'EOF'
#!/bin/bash
apt update
apt upgrade -y
apt autoremove -y
EOF

chmod +x /usr/local/bin/auto-update.sh

# 添加到定时任务
echo "0 3 * * 0 /usr/local/bin/auto-update.sh" | crontab -
```

## 监控和维护

### 1. 系统监控

```bash
# 安装监控插件
# 系统 -> 插件 -> 搜索monit -> 安装

# 监控脚本示例
cat > /usr/local/bin/system-check.sh << 'EOF'
#!/bin/bash
# 检查磁盘空间
df -h | grep -vE '^Filesystem|tmpfs|cdrom' | awk '{print $5 " " $1}' | while read output;
do
  percentage=$(echo $output | awk '{print $1}' | cut -d'%' -f1)
  partition=$(echo $output | awk '{print $2}')
  if [ $percentage -ge 90 ]; then
    echo "警告: $partition 磁盘使用率已达 $percentage%"
  fi
done

# 检查温度
temp=$(vcgencmd measure_temp | cut -d'=' -f2 | cut -d"'" -f1)
if (( $(echo "$temp > 70" | bc -l) )); then
  echo "警告: CPU温度过高: ${temp}°C"
fi
EOF

chmod +x /usr/local/bin/system-check.sh
```

### 2. 日志管理

```bash
# 配置日志轮转
cat > /etc/logrotate.d/nas-logs << 'EOF'
/var/log/samba/*.log {
    weekly
    rotate 4
    compress
    delaycompress
    missingok
    notifempty
}
EOF
```

### 3. 备份策略

```bash
# 系统配置备份脚本
cat > /usr/local/bin/config-backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/srv/backup/config"
DATE=$(date +%Y%m%d)

mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/system-config-$DATE.tar.gz \
    /etc /var/lib/openmediavault/config.xml

# 保留最近7天的备份
find $BACKUP_DIR -name "system-config-*.tar.gz" -mtime +7 -delete
EOF

chmod +x /usr/local/bin/config-backup.sh

# 每天凌晨2点执行备份
echo "0 2 * * * /usr/local/bin/config-backup.sh" | crontab -
```

## 客户端连接方法

### Windows客户端

```cmd
# 通过文件资源管理器
\\树莓派IP地址\共享名称

# 映射网络驱动器
net use Z: \\树莓派IP地址\共享名称 /persistent:yes
```

### macOS客户端

```bash
# 通过Finder
Cmd+K -> smb://树莓派IP地址

# 命令行挂载
sudo mkdir /mnt/nas
sudo mount -t smbfs //用户名@树莓派IP地址/共享名称 /mnt/nas
```

### Linux客户端

```bash
# 安装smbclient
sudo apt install smbclient cifs-utils

# 临时挂载
sudo mount -t cifs //树莓派IP地址/共享名称 /mnt/nas -o username=用户名

# 永久挂载 (添加到/etc/fstab)
//树莓派IP地址/共享名称 /mnt/nas cifs username=用户名,password=密码,uid=1000,gid=1000,iocharset=utf8 0 0
```

### 移动设备

- **Android**: 使用ES文件浏览器、Solid Explorer等应用
- **iOS**: 使用FileBrowser、FE File Explorer等应用

## 故障排除

### 常见问题及解决方案

#### 1. 无法访问Web界面

```bash
# 检查服务状态
systemctl status nginx
systemctl status openmediavault-engined

# 重启服务
systemctl restart nginx
systemctl restart openmediavault-engined
```

#### 2. 硬盘无法挂载

```bash
# 检查硬盘健康状态
smartctl -a /dev/sda

# 检查文件系统
fsck -f /dev/sda1

# 重新挂载
umount /dev/sda1
mount /dev/sda1 /挂载点
```

#### 3. 传输速度慢

```bash
# 检查网络连接
ethtool eth0

# 测试磁盘速度
hdparm -tT /dev/sda

# 检查CPU使用率
top
htop
```

#### 4. 温度过高

```bash
# 检查当前温度
vcgencmd measure_temp

# 检查散热器安装
# 考虑更换更好的散热方案
```

## 扩展功能

### 1. 私有云同步

```bash
# 安装Nextcloud
docker run -d \
  --name nextcloud \
  -p 8080:80 \
  -v /srv/nextcloud:/var/www/html \
  nextcloud
```

### 2. 下载服务器

```bash
# 安装Transmission (BT下载)
apt install transmission-daemon

# 配置下载目录
nano /etc/transmission-daemon/settings.json
```

### 3. 媒体服务器

```bash
# 安装Plex Media Server
wget -O- https://downloads.plex.tv/plex-keys/PlexSign.key | apt-key add -
echo deb https://downloads.plex.tv/repo/deb public main | tee /etc/apt/sources.list.d/plexmediaserver.list
apt update && apt install plexmediaserver
```

## 成本分析

### 初始投资

### 运行成本

- **电费**: 约15W功耗，
- **维护**: 偶尔需要更换SD卡或硬盘

### 与商业NAS对比

## 总结

**关键优势**:

- 成本低廉，性价比高
- 高度可定制，功能丰富
- 良好的社区支持
- 适合学习和实践

**注意事项**:

- 定期备份重要数据
- 保持系统和软件更新
- 监控硬件状态，特别是温度
- 根据实际需求调整配置
