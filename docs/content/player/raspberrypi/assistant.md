+++
draft = true
title = '家庭助手'
+++

家里的灯开关离睡觉的床太远了！破热水器还要我定时去打开!,最近AI助手太火了，崽崽也想要一个家庭助手。

## 项目概述

使用Raspberry Pi作为智能家居中控系统，整合各种传感器、执行器和智能设备，实现全屋自动化控制。本方案涵盖照明控制、环境监测、安防系统、语音控制等多个方面。

## 系统架构

```txt
                    ┌─────────────────────────────────┐
                    │      Raspberry Pi 4B           │
                    │    (Home Assistant Core)       │
                    │                                 │
                    │  ┌─────────┐  ┌─────────────┐   │
                    │  │ Node-RED│  │  Mosquitto  │   │
                    │  │   自动化 │  │  MQTT Broker│   │
                    │  └─────────┘  └─────────────┘   │
                    └─────────────┬───────────────────┘
                                  │
                    ┌─────────────┴───────────────────┐
                    │         网络层 (WiFi/有线)        │
                    └─────────────┬───────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
   ┌────▼────┐              ┌────▼────┐              ┌────▼────┐
   │ 传感器层  │              │ 控制层   │              │ 设备层   │
   │         │              │         │              │         │
   │• 温湿度  │              │• 继电器  │              │• 智能灯泡│
   │• 光照    │              │• 舵机    │              │• 智能插座│
   │• 运动    │              │• 步进电机│              │• 摄像头  │
   │• 烟雾    │              │• PWM控制 │              │• 音响    │
   │• 门窗    │              │• GPIO    │              │• 空调    │
   └─────────┘              └─────────┘              └─────────┘
```

## 想要的硬件清单

### 核心控制器

| 组件 | 型号规格 | 用途 |
|------|----------|------|
| **主控** | Raspberry Pi 4B (4GB) | 系统核心 |
| **存储** | 64GB microSD卡 (A2) | 系统存储 |
| **电源** | 5V 3A 官方电源 | 稳定供电 |
| **散热** | 主动散热风扇套装 | 温度控制 |
| **机箱** | 透明亚克力机箱 | 保护展示 |

### 通信模块

| 组件 | 型号规格 | 用途 |
|------|----------|------|
| **Zigbee** | CC2531 USB 棒 | Zigbee设备通信 |
| **433MHz** | RF433收发模块 | 无线遥控 |
| **红外** | IR发射接收模块 | 家电控制 |
| **蓝牙** | 内置蓝牙5.0 | 近距离通信 |

### 传感器套件

| 传感器类型 | 型号 | 功能 |
|------------|------|------|
| **温湿度** | DHT22/AM2302 | 环境监测 |
| **光照** | BH1750 数字光照 | 光线检测 |
| **运动** | PIR HC-SR501 | 人体感应 |
| **烟雾** | MQ-2 烟雾传感器 | 安全监测 |
| **气压** | BMP280 气压传感器 | 天气预测 |
| **湿度** | 湿度传感器 | 防护 |
| **声音** | 声音检测模块 | 噪音监测 |
| **门窗** | 磁簧开关 | 安防监控 |

## 实际装备

1. Raspberry Pi 4B (4GB)
2. 简易灯开关器
3. DHT22 温湿度传感器
4. 摄像头
5. 声音检测模块
6. 红外发射接收模块
7. 蓝牙模块

## 软件架构

### Home Assistant 核心系统

#### 系统安装

```bash
# 使用官方镜像安装 Home Assistant OS
# 下载镜像
wget https://github.com/home-assistant/operating-system/releases/download/10.3/haos_rpi4-64-10.3.img.xz

# 解压并烧录
xz -d haos_rpi4-64-10.3.img.xz
sudo dd if=haos_rpi4-64-10.3.img of=/dev/sdX bs=4M status=progress

# 首次启动后访问 http://树莓派IP:8123
```

#### 基础配置文件

```yaml
# configuration.yaml
homeassistant:
  name: 智能家居
  latitude: 39.9042  # 北京坐标
  longitude: 116.4074
  elevation: 43
  unit_system: metric
  time_zone: Asia/Shanghai
  currency: CNY
  external_url: "http://你的域名:8123"
  
# 启用默认配置
default_config:

# HTTP配置
http:
  use_x_forwarded_for: true
  trusted_proxies:
    - 192.168.1.0/24

# 录音机配置
recorder:
  db_url: sqlite:///config/home-assistant_v2.db
  purge_keep_days: 30
  include:
    domains:
      - sensor
      - switch
      - light
      - binary_sensor
      - climate

# 历史数据
history:
  include:
    domains:
      - sensor
      - switch
      - light

# 日志配置
logger:
  default: warning
  logs:
    homeassistant.core: info
    homeassistant.components.mqtt: debug

# MQTT配置
mqtt:
  broker: localhost
  port: 1883
  username: homeassistant
  password: your_password
  discovery: true
  discovery_prefix: homeassistant

# Zigbee2MQTT集成
zigbee2mqtt:
  
# 通知服务
notify:
  - platform: smtp
    name: email_notification
    server: smtp.gmail.com
    port: 587
    timeout: 15
    sender: your_email@gmail.com
    encryption: starttls
    username: your_email@gmail.com
    password: your_app_password
    recipient:
      - your_email@gmail.com

# 天气集成
weather:
  - platform: openweathermap
    api_key: your_api_key
    mode: onecall
```

未完。。
