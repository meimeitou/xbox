+++
title = 'Cilium网络插件'
+++

- [什么是 Cilium？](#什么是-cilium)
- [核心特性](#核心特性)
  - [1. 网络连接](#1-网络连接)
  - [2. 安全策略](#2-安全策略)
  - [3. 可观测性](#3-可观测性)
- [架构概览](#架构概览)
- [安装 Cilium](#安装-cilium)
  - [先决条件](#先决条件)
  - [1. 使用 Cilium CLI 安装](#1-使用-cilium-cli-安装)
  - [2. 使用 Helm 安装](#2-使用-helm-安装)
  - [3. 使用 Manifest 文件安装](#3-使用-manifest-文件安装)
- [配置选项](#配置选项)
  - [基本配置](#基本配置)
  - [高级配置](#高级配置)
- [网络策略配置](#网络策略配置)
  - [1. 基本网络策略](#1-基本网络策略)
  - [2. L7 HTTP 策略](#2-l7-http-策略)
  - [3. DNS 策略](#3-dns-策略)
- [可观测性配置](#可观测性配置)
  - [1. 启用 Hubble](#1-启用-hubble)
  - [2. 启用 Prometheus 监控](#2-启用-prometheus-监控)
- [故障排除](#故障排除)
  - [1. 检查 Cilium 状态](#1-检查-cilium-状态)
  - [2. 网络连接测试](#2-网络连接测试)
  - [3. 查看 Cilium 日志](#3-查看-cilium-日志)
  - [4. 调试 eBPF 程序](#4-调试-ebpf-程序)
- [性能优化](#性能优化)
  - [1. 数据平面优化](#1-数据平面优化)
  - [2. 资源优化](#2-资源优化)
- [监控和告警](#监控和告警)
  - [1. 关键指标监控](#1-关键指标监控)
- [总结](#总结)

## 什么是 Cilium？

Cilium 是一个开源的网络、可观测性和安全解决方案，专为云原生环境设计。它基于 eBPF (Extended Berkeley Packet Filter) 技术，为 Kubernetes 集群提供高性能的网络连接、负载均衡和安全策略。

Cilium 的核心优势在于利用 eBPF 在 Linux 内核中提供可编程的网络功能，相比传统的基于 iptables 的解决方案，具有更好的性能和更低的延迟。

## 核心特性

### 1. 网络连接

- **Pod 间通信**: 高效的容器间网络通信
- **跨节点通信**: 支持多种数据平面模式
- **负载均衡**: 内置高性能负载均衡器
- **多集群网络**: 跨集群服务发现和通信

### 2. 安全策略

- **网络策略**: 基于身份的 L3/L4 网络策略
- **应用层安全**: L7 策略支持（HTTP、gRPC、Kafka 等）
- **加密**: 透明的节点间通信加密
- **运行时安全**: 基于 eBPF 的运行时保护

### 3. 可观测性

- **网络流量监控**: 实时网络流量可视化
- **服务映射**: 自动服务依赖关系发现
- **性能指标**: 详细的网络性能监控
- **分布式跟踪**: 集成 Jaeger 和 Zipkin

## 架构概览

```txt
┌─────────────────────────────────────────────────────────────┐
│                    Cilium 架构                              │
├─────────────────────────────────────────────────────────────┤
│                 User Space                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Cilium    │  │   Hubble    │  │   Cilium    │        │
│  │   Agent     │  │   Server    │  │   Operator  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                 Kernel Space                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    eBPF     │  │    eBPF     │  │    eBPF     │        │
│  │  Programs   │  │    Maps     │  │   Hooks     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 安装 Cilium

### 先决条件

```bash
# 检查内核版本（推荐 4.19+）
uname -r

# 检查 eBPF 支持
mount | grep bpf

# 如果没有挂载，手动挂载
sudo mount bpffs -t bpf /sys/fs/bpf
```

### 1. 使用 Cilium CLI 安装

```bash
# 安装 Cilium CLI
curl -L --remote-name-all https://github.com/cilium/cilium-cli/releases/latest/download/cilium-linux-amd64.tar.gz{,.sha256sum}
sha256sum --check cilium-linux-amd64.tar.gz.sha256sum
sudo tar xzvfC cilium-linux-amd64.tar.gz /usr/local/bin
rm cilium-linux-amd64.tar.gz{,.sha256sum}

# 安装 Cilium 到 Kubernetes 集群
cilium install

# 验证安装
cilium status --wait
```

### 2. 使用 Helm 安装

```bash
# 添加 Cilium Helm 仓库
helm repo add cilium https://helm.cilium.io/
helm repo update

# 安装 Cilium
helm install cilium cilium/cilium --version 1.14.5 \
  --namespace kube-system \
  --set kubeProxyReplacement=partial \
  --set hostServices.enabled=false \
  --set externalIPs.enabled=true \
  --set nodePort.enabled=true \
  --set hostPort.enabled=true \
  --set bpf.masquerade=false \
  --set image.pullPolicy=IfNotPresent \
  --set ipam.mode=kubernetes
```

### 3. 使用 Manifest 文件安装

```bash
# 下载和应用清单文件
kubectl apply -f https://raw.githubusercontent.com/cilium/cilium/1.14.5/install/kubernetes/quick-install.yaml

# 等待 Pod 就绪
kubectl wait --for=condition=ready pod -l k8s-app=cilium -n kube-system
```

## 配置选项

### 基本配置

```yaml
# values.yaml
cluster:
  name: "my-cluster"
  id: 1

# IPAM 配置
ipam:
  mode: "kubernetes"  # 或 "cluster-pool", "multi-pool"

# 数据平面模式
datapath:
  mode: "vxlan"  # 或 "geneve", "native"

# 负载均衡
loadBalancer:
  algorithm: "random"  # 或 "round_robin", "source_hash"

# 网络策略
policyEnforcement: "default"  # 或 "always", "never"
```

### 高级配置

```yaml
# 启用 kube-proxy 替换
kubeProxyReplacement: "strict"

# 启用 Hubble 可观测性
hubble:
  enabled: true
  relay:
    enabled: true
  ui:
    enabled: true

# 启用透明加密
encryption:
  enabled: true
  type: "wireguard"  # 或 "ipsec"

# 启用 Cilium Operator
operator:
  replicas: 1
  resources:
    limits:
      cpu: 1000m
      memory: 1Gi
```

## 网络策略配置

### 1. 基本网络策略

```yaml
apiVersion: "cilium.io/v2"
kind: CiliumNetworkPolicy
metadata:
  name: "allow-frontend-to-backend"
spec:
  endpointSelector:
    matchLabels:
      role: backend
  ingress:
  - fromEndpoints:
    - matchLabels:
        role: frontend
    toPorts:
    - ports:
      - port: "8080"
        protocol: TCP
```

### 2. L7 HTTP 策略

```yaml
apiVersion: "cilium.io/v2"
kind: CiliumNetworkPolicy
metadata:
  name: "l7-http-policy"
spec:
  endpointSelector:
    matchLabels:
      app: web-server
  ingress:
  - fromEndpoints:
    - matchLabels:
        app: client
    toPorts:
    - ports:
      - port: "80"
        protocol: TCP
      rules:
        http:
        - method: "GET"
          path: "/api/.*"
        - method: "POST"
          path: "/api/users"
```

### 3. DNS 策略

```yaml
apiVersion: "cilium.io/v2"
kind: CiliumNetworkPolicy
metadata:
  name: "dns-policy"
spec:
  endpointSelector:
    matchLabels:
      app: client
  egress:
  - toEndpoints:
    - matchLabels:
        "k8s:io.kubernetes.pod.namespace": kube-system
        "k8s:k8s-app": kube-dns
    toPorts:
    - ports:
      - port: "53"
        protocol: UDP
      rules:
        dns:
        - matchName: "example.com"
        - matchPattern: "*.example.com"
```

## 可观测性配置

### 1. 启用 Hubble

```yaml
# hubble-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hubble-config
  namespace: kube-system
data:
  config.yaml: |
    peer-service: "hubble-peer.kube-system.svc.cluster.local:80"
    listen-address: ":4244"
    dial-timeout: "5s"
    retry-timeout: "30s"
    sort-buffer-len-max: 100
    sort-buffer-drain-timeout: "1s"
    tls-hubble-server-cert-file: "/var/lib/hubble-tls/server.crt"
    tls-hubble-server-key-file: "/var/lib/hubble-tls/server.key"
    tls-hubble-client-cert-file: "/var/lib/hubble-tls/client.crt"
    tls-hubble-client-key-file: "/var/lib/hubble-tls/client.key"
```

### 2. 启用 Prometheus 监控

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cilium-prometheus-config
  namespace: kube-system
data:
  prometheus.yaml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'cilium-agent'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_k8s_app]
        action: keep
        regex: cilium
      - source_labels: [__address__]
        action: replace
        regex: ([^:]+)(?::\d+)?
        target_label: __address__
        replacement: ${1}:9962
```

## 故障排除

### 1. 检查 Cilium 状态

```bash
# 检查 Cilium 整体状态
cilium status

# 检查 Cilium 连接性
cilium connectivity test

# 检查 Pod 状态
kubectl get pods -n kube-system -l k8s-app=cilium
```

### 2. 网络连接测试

```bash
# 创建测试 Pod
kubectl create -f https://raw.githubusercontent.com/cilium/cilium/HEAD/examples/minikube/http-sw-app.yaml

# 测试连接性
kubectl exec -it deployment/deathstar -- curl -s -XPOST deathstar.default.svc.cluster.local/v1/request-landing
```

### 3. 查看 Cilium 日志

```bash
# 查看 Cilium Agent 日志
kubectl logs -n kube-system -l k8s-app=cilium

# 查看 Cilium Operator 日志
kubectl logs -n kube-system -l name=cilium-operator

# 实时查看日志
kubectl logs -n kube-system -l k8s-app=cilium -f
```

### 4. 调试 eBPF 程序

```bash
# 进入 Cilium Pod
kubectl exec -it -n kube-system cilium-xxx -- bash

# 查看 eBPF 程序
cilium bpf endpoint list

# 查看 eBPF 映射
cilium bpf policy get

# 查看网络策略
cilium policy get
```

## 性能优化

### 1. 数据平面优化

```yaml
# 使用原生路由模式
tunnel: "disabled"
autoDirectNodeRoutes: true
ipv4NativeRoutingCIDR: "10.0.0.0/8"

# 启用 eBPF 主机路由
enableHostPort: true
hostServices:
  enabled: true
  protocols: tcp,udp

# 优化 eBPF 映射大小
bpf:
  mapDynamicSizeRatio: 0.25
  policyMapMax: 16384
  monitorAggregation: maximum
```

### 2. 资源优化

```yaml
# Cilium Agent 资源限制
resources:
  limits:
    cpu: 4000m
    memory: 4Gi
  requests:
    cpu: 100m
    memory: 512Mi

# 调整 eBPF 配置
bpf:
  preallocateMaps: true
  ctTcpTimeout: 21600
  ctAnyTimeout: 60
  natMax: 524288
  neigh:
    restoreOnStartup: true
```

## 监控和告警

### 1. 关键指标监控

```bash
# 网络策略违规
cilium_policy_l3_l4_denied_total

# 数据包丢失
cilium_drop_count_total

# 连接跟踪表使用率
cilium_bpf_map_ops_total

# 端点健康状态
cilium_endpoint_state
```

## 总结

Cilium 作为下一代网络插件，具有以下显著优势：

1. **高性能**: 基于 eBPF 技术，性能远超传统 iptables 方案
2. **强安全**: 提供 L3-L7 的网络安全策略
3. **强可观测性**: 内置 Hubble 提供网络流量可视化
4. **易扩展**: 支持多集群、多云部署
5. **功能丰富**: 包含负载均衡、加密、服务网格等功能

选择 Cilium 时需要考虑：

- 内核版本要求（推荐 4.19+）
- 团队的 eBPF 技术熟悉度
- 现有网络基础设施的兼容性
- 性能和功能需求

随着云原生技术的发展，Cilium 将成为 Kubernetes 网络的重要选择，特别适合对性能和安全有高要求的生产环境。

---

*作者：meimeitou*  
