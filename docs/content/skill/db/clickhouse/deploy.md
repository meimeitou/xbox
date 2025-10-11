+++
date = '2025-10-10T11:17:30+08:00'
title = 'ClickHouse 部署指南'
description = 'ClickHouse Docker Compose 和高可用集群部署完整指南'
categories = ['db', 'clickhouse', 'docker', '运维']
tags = ['db', 'clickhouse', 'docker-compose', '高可用', '集群部署']
+++

## 概述

ClickHouse 作为高性能的列式数据库，在生产环境中的部署策略至关重要。本文将详细介绍使用 Docker Compose 进行单节点部署，以及构建高可用集群的完整方案。

## Docker Compose 单节点部署

### 基础单节点部署

首先创建最简单的单节点 ClickHouse 部署：

```yaml
# docker-compose.yml
version: '3.8'

services:
  clickhouse:
    image: clickhouse/clickhouse-server:latest
    container_name: clickhouse-server
    ports:
      - "8123:8123"    # HTTP接口
      - "9000:9000"    # TCP接口
      - "9009:9009"    # HTTP代理接口
    volumes:
      - clickhouse_data:/var/lib/clickhouse
      - clickhouse_logs:/var/log/clickhouse-server
      - ./config:/etc/clickhouse-server/config.d
      - ./users:/etc/clickhouse-server/users.d
    environment:
      CLICKHOUSE_DB: default
      CLICKHOUSE_USER: admin
      CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT: 1
    ulimits:
      nofile:
        soft: 262144
        hard: 262144
    restart: unless-stopped
    networks:
      - clickhouse-network

volumes:
  clickhouse_data:
  clickhouse_logs:

networks:
  clickhouse-network:
    driver: bridge
```

### 增强配置部署

创建包含自定义配置的完整单节点部署：

```yaml
# docker-compose.yml
version: '3.8'

services:
  clickhouse:
    image: clickhouse/clickhouse-server:23.8
    container_name: clickhouse-server
    hostname: clickhouse-server
    ports:
      - "8123:8123"
      - "9000:9000"
      - "9009:9009"
    volumes:
      - clickhouse_data:/var/lib/clickhouse
      - clickhouse_logs:/var/log/clickhouse-server
      - ./config/config.xml:/etc/clickhouse-server/config.xml
      - ./config/users.xml:/etc/clickhouse-server/users.xml
      - ./config/config.d:/etc/clickhouse-server/config.d
    environment:
      - CLICKHOUSE_DB=analytics
      - CLICKHOUSE_USER=admin
      - CLICKHOUSE_PASSWORD=secure_password
    ulimits:
      nofile:
        soft: 262144
        hard: 262144
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:8123/ping"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    networks:
      - clickhouse-network

volumes:
  clickhouse_data:
    driver: local
  clickhouse_logs:
    driver: local

networks:
  clickhouse-network:
    driver: bridge
```

### 配置文件

创建自定义配置文件：

```xml
<!-- config/config.xml -->
<?xml version="1.0"?>
<clickhouse>
    <!-- 监听配置 -->
    <listen_host>0.0.0.0</listen_host>
    <http_port>8123</http_port>
    <tcp_port>9000</tcp_port>
    <interserver_http_port>9009</interserver_http_port>

    <!-- 数据目录 -->
    <path>/var/lib/clickhouse/</path>
    <tmp_path>/var/lib/clickhouse/tmp/</tmp_path>
    <user_files_path>/var/lib/clickhouse/user_files/</user_files_path>

    <!-- 日志配置 -->
    <logger>
        <level>information</level>
        <log>/var/log/clickhouse-server/clickhouse-server.log</log>
        <errorlog>/var/log/clickhouse-server/clickhouse-server.err.log</errorlog>
        <size>1000M</size>
        <count>10</count>
    </logger>

    <!-- 内存配置 -->
    <max_server_memory_usage_to_ram_ratio>0.8</max_server_memory_usage_to_ram_ratio>
    <max_memory_usage>10000000000</max_memory_usage>

    <!-- 并发配置 -->
    <max_concurrent_queries>100</max_concurrent_queries>
    <max_thread_pool_size>10000</max_thread_pool_size>

    <!-- 网络配置 -->
    <keep_alive_timeout>10</keep_alive_timeout>
    <max_connections>4096</max_connections>

    <!-- 压缩配置 -->
    <compression>
        <case>
            <method>lz4</method>
        </case>
    </compression>
</clickhouse>
```

```xml
<!-- config/users.xml -->
<?xml version="1.0"?>
<clickhouse>
    <users>
        <admin>
            <password_sha256_hex>e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855</password_sha256_hex>
            <networks>
                <ip>::/0</ip>
            </networks>
            <profile>default</profile>
            <quota>default</quota>
            <access_management>1</access_management>
        </admin>
        
        <readonly>
            <password>readonly_password</password>
            <networks>
                <ip>::/0</ip>
            </networks>
            <profile>readonly</profile>
            <quota>default</quota>
        </readonly>
    </users>

    <profiles>
        <default>
            <max_memory_usage>10000000000</max_memory_usage>
            <use_uncompressed_cache>0</use_uncompressed_cache>
            <load_balancing>random</load_balancing>
        </default>
        
        <readonly>
            <readonly>1</readonly>
            <max_memory_usage>5000000000</max_memory_usage>
        </readonly>
    </profiles>

    <quotas>
        <default>
            <interval>
                <duration>3600</duration>
                <queries>0</queries>
                <errors>0</errors>
                <result_rows>0</result_rows>
                <read_rows>0</read_rows>
                <execution_time>0</execution_time>
            </interval>
        </default>
    </quotas>
</clickhouse>
```

### 启动和管理

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f clickhouse

# 连接ClickHouse
docker-compose exec clickhouse clickhouse-client

# 停止服务
docker-compose down

# 完全清理（包括数据）
docker-compose down -v
```

## 高可用集群部署

### 集群架构设计

高可用 ClickHouse 集群通常采用以下架构：

- **分片（Shard）**：数据水平分割
- **副本（Replica）**：数据冗余备份
- **ZooKeeper**：集群协调服务

### ZooKeeper 集群部署

```yaml
# docker-compose-zk.yml
version: '3.8'

services:
  zookeeper1:
    image: zookeeper:3.8
    container_name: zookeeper1
    hostname: zookeeper1
    ports:
      - "2181:2181"
    environment:
      ZOO_MY_ID: 1
      ZOO_SERVERS: server.1=zookeeper1:2888:3888;2181 server.2=zookeeper2:2888:3888;2181 server.3=zookeeper3:2888:3888;2181
      ZOO_AUTOPURGE_PURGEINTERVAL: 1
      ZOO_AUTOPURGE_SNAPRETAINCOUNT: 3
    volumes:
      - zk1_data:/data
      - zk1_logs:/datalog
    networks:
      - clickhouse-cluster

  zookeeper2:
    image: zookeeper:3.8
    container_name: zookeeper2
    hostname: zookeeper2
    ports:
      - "2182:2181"
    environment:
      ZOO_MY_ID: 2
      ZOO_SERVERS: server.1=zookeeper1:2888:3888;2181 server.2=zookeeper2:2888:3888;2181 server.3=zookeeper3:2888:3888;2181
      ZOO_AUTOPURGE_PURGEINTERVAL: 1
      ZOO_AUTOPURGE_SNAPRETAINCOUNT: 3
    volumes:
      - zk2_data:/data
      - zk2_logs:/datalog
    networks:
      - clickhouse-cluster

  zookeeper3:
    image: zookeeper:3.8
    container_name: zookeeper3
    hostname: zookeeper3
    ports:
      - "2183:2181"
    environment:
      ZOO_MY_ID: 3
      ZOO_SERVERS: server.1=zookeeper1:2888:3888;2181 server.2=zookeeper2:2888:3888;2181 server.3=zookeeper3:2888:3888;2181
      ZOO_AUTOPURGE_PURGEINTERVAL: 1
      ZOO_AUTOPURGE_SNAPRETAINCOUNT: 3
    volumes:
      - zk3_data:/data
      - zk3_logs:/datalog
    networks:
      - clickhouse-cluster

volumes:
  zk1_data:
  zk1_logs:
  zk2_data:
  zk2_logs:
  zk3_data:
  zk3_logs:

networks:
  clickhouse-cluster:
    driver: bridge
```

### ClickHouse 集群部署

```yaml
# docker-compose-cluster.yml
version: '3.8'

services:
  # Shard 1 - Replica 1
  clickhouse-s1-r1:
    image: clickhouse/clickhouse-server:23.8
    container_name: clickhouse-s1-r1
    hostname: clickhouse-s1-r1
    ports:
      - "8123:8123"
      - "9000:9000"
    volumes:
      - ./cluster-config/config.xml:/etc/clickhouse-server/config.xml
      - ./cluster-config/users.xml:/etc/clickhouse-server/users.xml
      - ./cluster-config/cluster.xml:/etc/clickhouse-server/config.d/cluster.xml
      - ./cluster-config/macros/macros-s1-r1.xml:/etc/clickhouse-server/config.d/macros.xml
      - ch_s1_r1_data:/var/lib/clickhouse
    environment:
      - CLICKHOUSE_USER=admin
      - CLICKHOUSE_PASSWORD=secure_password
    depends_on:
      - zookeeper1
      - zookeeper2
      - zookeeper3
    ulimits:
      nofile:
        soft: 262144
        hard: 262144
    networks:
      - clickhouse-cluster

  # Shard 1 - Replica 2
  clickhouse-s1-r2:
    image: clickhouse/clickhouse-server:23.8
    container_name: clickhouse-s1-r2
    hostname: clickhouse-s1-r2
    ports:
      - "8124:8123"
      - "9001:9000"
    volumes:
      - ./cluster-config/config.xml:/etc/clickhouse-server/config.xml
      - ./cluster-config/users.xml:/etc/clickhouse-server/users.xml
      - ./cluster-config/cluster.xml:/etc/clickhouse-server/config.d/cluster.xml
      - ./cluster-config/macros/macros-s1-r2.xml:/etc/clickhouse-server/config.d/macros.xml
      - ch_s1_r2_data:/var/lib/clickhouse
    environment:
      - CLICKHOUSE_USER=admin
      - CLICKHOUSE_PASSWORD=secure_password
    depends_on:
      - zookeeper1
      - zookeeper2
      - zookeeper3
    ulimits:
      nofile:
        soft: 262144
        hard: 262144
    networks:
      - clickhouse-cluster

  # Shard 2 - Replica 1
  clickhouse-s2-r1:
    image: clickhouse/clickhouse-server:23.8
    container_name: clickhouse-s2-r1
    hostname: clickhouse-s2-r1
    ports:
      - "8125:8123"
      - "9002:9000"
    volumes:
      - ./cluster-config/config.xml:/etc/clickhouse-server/config.xml
      - ./cluster-config/users.xml:/etc/clickhouse-server/users.xml
      - ./cluster-config/cluster.xml:/etc/clickhouse-server/config.d/cluster.xml
      - ./cluster-config/macros/macros-s2-r1.xml:/etc/clickhouse-server/config.d/macros.xml
      - ch_s2_r1_data:/var/lib/clickhouse
    environment:
      - CLICKHOUSE_USER=admin
      - CLICKHOUSE_PASSWORD=secure_password
    depends_on:
      - zookeeper1
      - zookeeper2
      - zookeeper3
    ulimits:
      nofile:
        soft: 262144
        hard: 262144
    networks:
      - clickhouse-cluster

  # Shard 2 - Replica 2
  clickhouse-s2-r2:
    image: clickhouse/clickhouse-server:23.8
    container_name: clickhouse-s2-r2
    hostname: clickhouse-s2-r2
    ports:
      - "8126:8123"
      - "9003:9000"
    volumes:
      - ./cluster-config/config.xml:/etc/clickhouse-server/config.xml
      - ./cluster-config/users.xml:/etc/clickhouse-server/users.xml
      - ./cluster-config/cluster.xml:/etc/clickhouse-server/config.d/cluster.xml
      - ./cluster-config/macros/macros-s2-r2.xml:/etc/clickhouse-server/config.d/macros.xml
      - ch_s2_r2_data:/var/lib/clickhouse
    environment:
      - CLICKHOUSE_USER=admin
      - CLICKHOUSE_PASSWORD=secure_password
    depends_on:
      - zookeeper1
      - zookeeper2
      - zookeeper3
    ulimits:
      nofile:
        soft: 262144
        hard: 262144
    networks:
      - clickhouse-cluster

volumes:
  ch_s1_r1_data:
  ch_s1_r2_data:
  ch_s2_r1_data:
  ch_s2_r2_data:

networks:
  clickhouse-cluster:
    external: true
```

### 集群配置文件

```xml
<!-- cluster-config/cluster.xml -->
<?xml version="1.0"?>
<clickhouse>
    <!-- ZooKeeper配置 -->
    <zookeeper>
        <node>
            <host>zookeeper1</host>
            <port>2181</port>
        </node>
        <node>
            <host>zookeeper2</host>
            <port>2181</port>
        </node>
        <node>
            <host>zookeeper3</host>
            <port>2181</port>
        </node>
        <session_timeout_ms>30000</session_timeout_ms>
        <operation_timeout_ms>10000</operation_timeout_ms>
        <root>/clickhouse</root>
        <identity>user:password</identity>
    </zookeeper>

    <!-- 集群配置 -->
    <remote_servers>
        <cluster_2shards_2replicas>
            <!-- Shard 1 -->
            <shard>
                <replica>
                    <host>clickhouse-s1-r1</host>
                    <port>9000</port>
                </replica>
                <replica>
                    <host>clickhouse-s1-r2</host>
                    <port>9000</port>
                </replica>
            </shard>
            <!-- Shard 2 -->
            <shard>
                <replica>
                    <host>clickhouse-s2-r1</host>
                    <port>9000</port>
                </replica>
                <replica>
                    <host>clickhouse-s2-r2</host>
                    <port>9000</port>
                </replica>
            </shard>
        </cluster_2shards_2replicas>
    </remote_servers>

    <!-- 分布式DDL -->
    <distributed_ddl>
        <path>/clickhouse/task_queue/ddl</path>
    </distributed_ddl>
</clickhouse>
```

宏配置文件（每个节点不同）：

```xml
<!-- cluster-config/macros/macros-s1-r1.xml -->
<?xml version="1.0"?>
<clickhouse>
    <macros>
        <cluster>cluster_2shards_2replicas</cluster>
        <shard>01</shard>
        <replica>replica_1</replica>
    </macros>
</clickhouse>
```

```xml
<!-- cluster-config/macros/macros-s1-r2.xml -->
<?xml version="1.0"?>
<clickhouse>
    <macros>
        <cluster>cluster_2shards_2replicas</cluster>
        <shard>01</shard>
        <replica>replica_2</replica>
    </macros>
</clickhouse>
```

### 负载均衡器配置

使用 HAProxy 作为负载均衡器：

```yaml
# 添加到 docker-compose-cluster.yml
  haproxy:
    image: haproxy:2.8
    container_name: clickhouse-haproxy
    ports:
      - "8080:8080"  # 统计页面
      - "8123:8123"  # ClickHouse HTTP
      - "9000:9000"  # ClickHouse TCP
    volumes:
      - ./haproxy/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg
    depends_on:
      - clickhouse-s1-r1
      - clickhouse-s1-r2
      - clickhouse-s2-r1
      - clickhouse-s2-r2
    networks:
      - clickhouse-cluster
```

HAProxy 配置文件：

```haproxy
# haproxy/haproxy.cfg
global
    daemon
    maxconn 4096

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

# 统计页面
stats enable
stats uri /stats
stats refresh 30s
stats admin if TRUE

# HTTP负载均衡
frontend clickhouse_http
    bind *:8123
    default_backend clickhouse_http_backend

backend clickhouse_http_backend
    balance roundrobin
    option httpchk GET /ping
    server s1r1 clickhouse-s1-r1:8123 check
    server s1r2 clickhouse-s1-r2:8123 check
    server s2r1 clickhouse-s2-r1:8123 check
    server s2r2 clickhouse-s2-r2:8123 check

# TCP负载均衡
frontend clickhouse_tcp
    bind *:9000
    mode tcp
    default_backend clickhouse_tcp_backend

backend clickhouse_tcp_backend
    mode tcp
    balance roundrobin
    server s1r1 clickhouse-s1-r1:9000 check
    server s1r2 clickhouse-s1-r2:9000 check
    server s2r1 clickhouse-s2-r1:9000 check
    server s2r2 clickhouse-s2-r2:9000 check
```

## 集群部署和管理

### 部署脚本

```bash
#!/bin/bash
# deploy-cluster.sh

set -e

echo "=== ClickHouse 集群部署 ==="

# 创建网络
echo "创建 Docker 网络..."
docker network create clickhouse-cluster 2>/dev/null || true

# 启动 ZooKeeper 集群
echo "启动 ZooKeeper 集群..."
docker-compose -f docker-compose-zk.yml up -d

# 等待 ZooKeeper 启动
echo "等待 ZooKeeper 集群就绪..."
sleep 30

# 启动 ClickHouse 集群
echo "启动 ClickHouse 集群..."
docker-compose -f docker-compose-cluster.yml up -d

# 等待 ClickHouse 启动
echo "等待 ClickHouse 集群就绪..."
sleep 60

# 验证集群状态
echo "验证集群状态..."
docker exec clickhouse-s1-r1 clickhouse-client --query "SELECT * FROM system.clusters"

echo "=== 部署完成 ==="
echo "ClickHouse HTTP: http://localhost:8123"
echo "HAProxy 统计: http://localhost:8080/stats"
```

### 集群验证

连接集群并验证：

```sql
-- 检查集群状态
SELECT * FROM system.clusters;

-- 检查副本状态
SELECT * FROM system.replicas;

-- 创建分布式表
CREATE TABLE events ON CLUSTER cluster_2shards_2replicas (
    timestamp DateTime,
    user_id UInt32,
    event_type String,
    properties Map(String, String)
) ENGINE = ReplicatedMergeTree('/clickhouse/tables/{shard}/events', '{replica}')
PARTITION BY toYYYYMM(timestamp)
ORDER BY (user_id, timestamp);

-- 创建分布式表视图
CREATE TABLE events_distributed ON CLUSTER cluster_2shards_2replicas AS events
ENGINE = Distributed(cluster_2shards_2replicas, default, events, rand());

-- 插入测试数据
INSERT INTO events_distributed VALUES 
    ('2023-10-10 12:00:00', 1001, 'page_view', {'page': '/home'}),
    ('2023-10-10 12:01:00', 1002, 'click', {'button': 'signup'});

-- 查询数据
SELECT count(*) FROM events_distributed;
```

## 生产环境最佳实践

### 资源配置建议

```yaml
# 生产环境配置示例
services:
  clickhouse:
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '8'
        reservations:
          memory: 8G
          cpus: '4'
    environment:
      - CLICKHOUSE_MAX_MEMORY_USAGE=12884901888  # 12GB
      - CLICKHOUSE_MAX_CONCURRENT_QUERIES=100
```

### 安全配置

```xml
<!-- 安全配置 -->
<clickhouse>
    <!-- SSL配置 -->
    <openSSL>
        <server>
            <certificateFile>/etc/clickhouse-server/server.crt</certificateFile>
            <privateKeyFile>/etc/clickhouse-server/server.key</privateKeyFile>
            <dhParamsFile>/etc/clickhouse-server/dhparam.pem</dhParamsFile>
            <verificationMode>none</verificationMode>
            <loadDefaultCAFile>true</loadDefaultCAFile>
            <cacheSessions>true</cacheSessions>
            <disableProtocols>sslv2,sslv3</disableProtocols>
            <preferServerCiphers>true</preferServerCiphers>
        </server>
    </openSSL>
    
    <!-- HTTPS端口 -->
    <https_port>8443</https_port>
    
    <!-- 禁用HTTP端口（可选）-->
    <!-- <http_port remove="remove"/> -->
</clickhouse>
```

### 性能调优

```xml
<!-- 性能优化配置 -->
<clickhouse>
    <!-- 内存优化 -->
    <mark_cache_size>5368709120</mark_cache_size>
    <uncompressed_cache_size>8589934592</uncompressed_cache_size>
    
    <!-- I/O优化 -->
    <max_concurrent_queries>100</max_concurrent_queries>
    <max_thread_pool_size>10000</max_thread_pool_size>
    
    <!-- 网络优化 -->
    <tcp_keep_alive_timeout>10</tcp_keep_alive_timeout>
    <max_connections>4096</max_connections>
</clickhouse>
```

## 故障排查

### 常见问题和解决方案

1. **ZooKeeper 连接问题**

```bash
# 检查 ZooKeeper 状态
docker exec zookeeper1 zkServer.sh status

# 查看 ClickHouse 日志
docker logs clickhouse-s1-r1 | grep -i zookeeper
```

2. **副本同步问题**

```sql
-- 检查副本状态
SELECT * FROM system.replicas WHERE is_readonly = 1;

-- 修复副本
SYSTEM RESTART REPLICA table_name;
```

3. **分片数据不均衡**

```sql
-- 检查分片数据分布
SELECT 
    hostName() as host,
    database,
    table,
    count() as rows
FROM cluster('cluster_2shards_2replicas', system.parts)
GROUP BY host, database, table;
```
