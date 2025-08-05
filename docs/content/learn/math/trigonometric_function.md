+++
date = '2025-08-05T17:48:58+08:00'
title = '三角函数'
weight = 2
tags = ["math", "数学", "基础"]
categories = ["数学基础", "数学"]
description = '三角函数的基本概念和性质'
+++

三角函数是数学中重要的基本函数，广泛应用于几何、物理、工程等领域。

## 基本概念

### 角的度量

- **角度制**：一个圆周为 360°
- **弧度制**：一个圆周为 2π 弧度
- 转换关系：$180° = \pi$ 弧度，即 $1° = \frac{\pi}{180}$ 弧度

### 单位圆中的三角函数

在单位圆（半径为1的圆）中，对于角 $\theta$：

- $\sin \theta$ = y 坐标
- $\cos \theta$ = x 坐标  
- $\tan \theta = \frac{\sin \theta}{\cos \theta}$ （当 $\cos \theta \neq 0$ 时）

## 六个基本三角函数

1. **正弦函数**：$\sin \theta$
2. **余弦函数**：$\cos \theta$
3. **正切函数**：$\tan \theta = \frac{\sin \theta}{\cos \theta}$
4. **余切函数**：$\cot \theta = \frac{\cos \theta}{\sin \theta}$
5. **正割函数**：$\sec \theta = \frac{1}{\cos \theta}$
6. **余割函数**：$\csc \theta = \frac{1}{\sin \theta}$

## 基本性质

### 定义域和值域

- $\sin \theta$：定义域 $\mathbb{R}$，值域 $[-1, 1]$
- $\cos \theta$：定义域 $\mathbb{R}$，值域 $[-1, 1]$
- $\tan \theta$：定义域 $\mathbb{R} \setminus \{\frac{\pi}{2} + k\pi, k \in \mathbb{Z}\}$，值域 $\mathbb{R}$

### 周期性

- $\sin(\theta + 2\pi) = \sin \theta$
- $\cos(\theta + 2\pi) = \cos \theta$
- $\tan(\theta + \pi) = \tan \theta$

### 奇偶性

- $\sin(-\theta) = -\sin \theta$（奇函数）
- $\cos(-\theta) = \cos \theta$（偶函数）
- $\tan(-\theta) = -\tan \theta$（奇函数）

## 重要的三角恒等式

### 基本恒等式

$$\sin^2 \theta + \cos^2 \theta = 1$$

$$1 + \tan^2 \theta = \sec^2 \theta$$

$$1 + \cot^2 \theta = \csc^2 \theta$$

### 和差公式

$$\sin(A \pm B) = \sin A \cos B \pm \cos A \sin B$$

$$\cos(A \pm B) = \cos A \cos B \mp \sin A \sin B$$

$$\tan(A \pm B) = \frac{\tan A \pm \tan B}{1 \mp \tan A \tan B}$$

### 二倍角公式

$$\sin 2\theta = 2\sin \theta \cos \theta$$

$$\cos 2\theta = \cos^2 \theta - \sin^2 \theta = 2\cos^2 \theta - 1 = 1 - 2\sin^2 \theta$$

$$\tan 2\theta = \frac{2\tan \theta}{1 - \tan^2 \theta}$$

### 半角公式

$$\sin \frac{\theta}{2} = \pm\sqrt{\frac{1 - \cos \theta}{2}}$$

$$\cos \frac{\theta}{2} = \pm\sqrt{\frac{1 + \cos \theta}{2}}$$

$$\tan \frac{\theta}{2} = \frac{1 - \cos \theta}{\sin \theta} = \frac{\sin \theta}{1 + \cos \theta}$$

### 积化和差公式

$$\sin A \sin B = \frac{1}{2}[\cos(A-B) - \cos(A+B)]$$

$$\cos A \cos B = \frac{1}{2}[\cos(A-B) + \cos(A+B)]$$

$$\sin A \cos B = \frac{1}{2}[\sin(A+B) + \sin(A-B)]$$

### 和差化积公式

$$\sin A + \sin B = 2\sin\frac{A+B}{2}\cos\frac{A-B}{2}$$

$$\sin A - \sin B = 2\cos\frac{A+B}{2}\sin\frac{A-B}{2}$$

$$\cos A + \cos B = 2\cos\frac{A+B}{2}\cos\frac{A-B}{2}$$

$$\cos A - \cos B = -2\sin\frac{A+B}{2}\sin\frac{A-B}{2}$$

## 特殊角的三角函数值

| 角度 | 0° | 30° | 45° | 60° | 90° |
|------|----|----|----|----|-----|
| 弧度 | 0 | $\frac{\pi}{6}$ | $\frac{\pi}{4}$ | $\frac{\pi}{3}$ | $\frac{\pi}{2}$ |
| $\sin$ | 0 | $\frac{1}{2}$ | $\frac{\sqrt{2}}{2}$ | $\frac{\sqrt{3}}{2}$ | 1 |
| $\cos$ | 1 | $\frac{\sqrt{3}}{2}$ | $\frac{\sqrt{2}}{2}$ | $\frac{1}{2}$ | 0 |
| $\tan$ | 0 | $\frac{\sqrt{3}}{3}$ | 1 | $\sqrt{3}$ | 未定义 |

## 应用示例

### 三角形中的应用

在任意三角形 ABC 中：

- **正弦定理**：$\frac{a}{\sin A} = \frac{b}{\sin B} = \frac{c}{\sin C} = 2R$
- **余弦定理**：$c^2 = a^2 + b^2 - 2ab\cos C$

### 波动现象

三角函数常用于描述周期性现象，如：
$$y = A\sin(\omega t + \phi)$$
其中 A 为振幅，$\omega$ 为角频率，$\phi$ 为初相位。

## 学习要点

1. 熟记特殊角的三角函数值
2. 掌握基本恒等式的应用
3. 理解三角函数的几何意义
4. 练习三角恒等式的证明和化简
5. 学会解三角方程和三角不等式
