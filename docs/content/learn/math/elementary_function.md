+++
date = '2025-08-05T18:07:04+08:00'
title = '初等函数'
weight = 3
tags = ["math", "数学", "基础"]
categories = ["数学基础", "数学"]
description = '初等函数的基本概念和性质'
+++

## 定义

初等函数是由基本初等函数（常数函数、幂函数、指数函数、对数函数、三角函数、反三角函数）经过有限次的四则运算和有限次的函数复合步骤所构成并可用一个式子表示的函数。

## 一、基本初等函数

### 1. 常数函数

**函数形式：** $f(x) = c$ （$c$ 为常数）

**性质：**

- 定义域：$(-\infty, +\infty)$
- 值域：$\{c\}$
- 图像：平行于 $x$ 轴的直线

### 2. 幂函数

**函数形式：** $f(x) = x^{\alpha}$ （$\alpha$ 为实常数）

**常见幂函数及性质：**

| 函数 | 定义域 | 值域 | 性质 |
|------|--------|------|------|
| $y = x$ | $(-\infty, +\infty)$ | $(-\infty, +\infty)$ | 奇函数，单调递增 |
| $y = x^2$ | $(-\infty, +\infty)$ | $[0, +\infty)$ | 偶函数，$x \geq 0$ 时单调递增 |
| $y = x^3$ | $(-\infty, +\infty)$ | $(-\infty, +\infty)$ | 奇函数，单调递增 |
| $y = \sqrt{x}$ | $[0, +\infty)$ | $[0, +\infty)$ | 单调递增 |
| $y = \frac{1}{x}$ | $(-\infty, 0) \cup (0, +\infty)$ | $(-\infty, 0) \cup (0, +\infty)$ | 奇函数，在各区间内单调递减 |

### 3. 指数函数

**函数形式：** $f(x) = a^x$ （$a > 0$ 且 $a \neq 1$）

**性质：**

- 定义域：$(-\infty, +\infty)$
- 值域：$(0, +\infty)$
- 过点 $(0, 1)$
- 当 $a > 1$ 时，单调递增
- 当 $0 < a < 1$ 时，单调递减

**重要公式：**

- $a^0 = 1$
- $a^{m+n} = a^m \cdot a^n$
- $a^{mn} = (a^m)^n$
- $(ab)^n = a^n b^n$

**自然指数函数：** $f(x) = e^x$

- $e \approx 2.71828$
- $(e^x)' = e^x$

### 4. 对数函数

**函数形式：** $f(x) = \log_a x$ （$a > 0$ 且 $a \neq 1$）

**性质：**

- 定义域：$(0, +\infty)$
- 值域：$(-\infty, +\infty)$
- 过点 $(1, 0)$
- 当 $a > 1$ 时，单调递增
- 当 $0 < a < 1$ 时，单调递减

**重要公式：**

- $\log_a 1 = 0$
- $\log_a a = 1$
- $\log_a (MN) = \log_a M + \log_a N$
- $\log_a \frac{M}{N} = \log_a M - \log_a N$
- $\log_a M^n = n \log_a M$
- $\log_a b = \frac{\log_c b}{\log_c a}$ （换底公式）

**自然对数函数：** $f(x) = \ln x = \log_e x$

- $(\ln x)' = \frac{1}{x}$

**常用对数函数：** $f(x) = \lg x = \log_{10} x$

### 5. 三角函数

#### 正弦函数：$y = \sin x$

- 定义域：$(-\infty, +\infty)$
- 值域：$[-1, 1]$
- 周期：$2\pi$
- 奇函数：$\sin(-x) = -\sin x$

#### 余弦函数：$y = \cos x$

- 定义域：$(-\infty, +\infty)$
- 值域：$[-1, 1]$
- 周期：$2\pi$
- 偶函数：$\cos(-x) = \cos x$

#### 正切函数：$y = \tan x$

- 定义域：$x \neq \frac{\pi}{2} + k\pi, k \in \mathbb{Z}$
- 值域：$(-\infty, +\infty)$
- 周期：$\pi$
- 奇函数：$\tan(-x) = -\tan x$

**三角恒等式：**

- $\sin^2 x + \cos^2 x = 1$
- $\tan x = \frac{\sin x}{\cos x}$
- $\sin(x \pm y) = \sin x \cos y \pm \cos x \sin y$
- $\cos(x \pm y) = \cos x \cos y \mp \sin x \sin y$
- $\sin 2x = 2\sin x \cos x$
- $\cos 2x = \cos^2 x - \sin^2 x = 2\cos^2 x - 1 = 1 - 2\sin^2 x$

### 6. 反三角函数

#### 反正弦函数：$y = \arcsin x$

- 定义域：$[-1, 1]$
- 值域：$[-\frac{\pi}{2}, \frac{\pi}{2}]$
- 奇函数

#### 反余弦函数：$y = \arccos x$

- 定义域：$[-1, 1]$
- 值域：$[0, \pi]$

#### 反正切函数：$y = \arctan x$

- 定义域：$(-\infty, +\infty)$
- 值域：$(-\frac{\pi}{2}, \frac{\pi}{2})$
- 奇函数

**反三角函数关系：**

- $\arcsin x + \arccos x = \frac{\pi}{2}$
- $\arctan x + \arctan \frac{1}{x} = \frac{\pi}{2}$ （$x > 0$）

## 二、函数的基本性质

### 1. 奇偶性

- **偶函数：** $f(-x) = f(x)$，图像关于 $y$ 轴对称
- **奇函数：** $f(-x) = -f(x)$，图像关于原点对称

### 2. 单调性

- **单调递增：** $x_1 < x_2 \Rightarrow f(x_1) < f(x_2)$
- **单调递减：** $x_1 < x_2 \Rightarrow f(x_1) > f(x_2)$

### 3. 周期性

- **周期函数：** 存在正数 $T$，使得 $f(x+T) = f(x)$ 对定义域内所有 $x$ 成立
- 最小正周期：使上述等式成立的最小正数 $T$

### 4. 有界性

- **有上界：** 存在 $M$，使得 $f(x) \leq M$
- **有下界：** 存在 $m$，使得 $f(x) \geq m$
- **有界：** 既有上界又有下界

## 三、复合函数

设 $y = f(u)$，$u = g(x)$，则复合函数为 $y = f(g(x))$

**复合函数求导链式法则：**
$$[f(g(x))]' = f'(g(x)) \cdot g'(x)$$

## 四、反函数

如果函数 $y = f(x)$ 在定义域内单调，则存在反函数 $x = f^{-1}(y)$

**反函数性质：**

- $f(f^{-1}(x)) = x$
- $f^{-1}(f(x)) = x$
- $y = f(x)$ 与 $y = f^{-1}(x)$ 的图像关于直线 $y = x$ 对称

## 五、常用不等式

### 基本不等式

- **算术-几何平均不等式：** $\frac{a + b}{2} \geq \sqrt{ab}$ （$a, b \geq 0$）
- **当且仅当 $a = b$ 时等号成立**

### 三角不等式

- $|\sin x| \leq 1$, $|\cos x| \leq 1$
- $|a + b| \leq |a| + |b|$
- $||a| - |b|| \leq |a - b|$

## 六、重要极限

- $\lim_{x \to 0} \frac{\sin x}{x} = 1$
- $\lim_{x \to \infty} \left(1 + \frac{1}{x}\right)^x = e$
- $\lim_{x \to 0} \frac{e^x - 1}{x} = 1$
- $\lim_{x \to 0} \frac{\ln(1 + x)}{x} = 1$
