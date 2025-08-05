+++
date = '2025-08-06T06:57:36+08:00'
title = '微积分'
description = '微积分基础知识和公式'
+++

## 1. 极限 (Limits)

### 1.1 极限的定义

对于函数 $f(x)$，当 $x$ 趋近于 $a$ 时，如果 $f(x)$ 趋近于某个确定的值 $L$，则称 $L$ 为函数 $f(x)$ 当 $x \to a$ 时的极限。

$$\lim_{x \to a} f(x) = L$$

### 1.2 常用极限公式

#### 基本三角极限

- $\lim_{x \to 0} \frac{\sin x}{x} = 1$
- $\lim_{x \to 0} \frac{\tan x}{x} = 1$
- $\lim_{x \to 0} \frac{1 - \cos x}{x^2} = \frac{1}{2}$
- $\lim_{x \to 0} \frac{\arcsin x}{x} = 1$
- $\lim_{x \to 0} \frac{\arctan x}{x} = 1$

#### 指数与对数极限

- $\lim_{x \to 0} \frac{e^x - 1}{x} = 1$
- $\lim_{x \to 0} \frac{a^x - 1}{x} = \ln a$ (a > 0, a ≠ 1)
- $\lim_{x \to 0} \frac{\ln(1 + x)}{x} = 1$
- $\lim_{x \to 0} \frac{\log_a(1 + x)}{x} = \frac{1}{\ln a}$ (a > 0, a ≠ 1)

#### 重要极限

- $\lim_{x \to \infty} \left(1 + \frac{1}{x}\right)^x = e$
- $\lim_{x \to 0} (1 + x)^{\frac{1}{x}} = e$
- $\lim_{x \to 0} \frac{(1 + x)^n - 1}{x} = n$
- $\lim_{x \to 0} \frac{(1 + x)^{\alpha} - 1}{x} = \alpha$ (α为任意实数)

#### 无穷大与无穷小

- $\lim_{x \to \infty} \frac{x^n}{e^x} = 0$ (n为任意正整数)
- $\lim_{x \to \infty} \frac{\ln x}{x^n} = 0$ (n > 0)
- $\lim_{x \to +\infty} x^n e^{-x} = 0$ (n为任意正数)
- $\lim_{n \to \infty} \sqrt[n]{n} = 1$
- $\lim_{n \to \infty} \sqrt[n]{a} = 1$ (a > 0)

#### 数列极限

- $\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e$
- $\lim_{n \to \infty} \frac{n!}{n^n e^{-n} \sqrt{2\pi n}} = 1$ (Stirling公式)
- $\lim_{n \to \infty} \frac{a^n}{n!} = 0$ (a为任意常数)
- $\lim_{n \to \infty} \frac{n^k}{a^n} = 0$ (a > 1, k为任意正数)

#### 等价无穷小 (x → 0)

- $\sin x \sim x$
- $\tan x \sim x$
- $\arcsin x \sim x$
- $\arctan x \sim x$
- $e^x - 1 \sim x$
- $\ln(1 + x) \sim x$
- $a^x - 1 \sim x \ln a$
- $(1 + x)^{\alpha} - 1 \sim \alpha x$
- $1 - \cos x \sim \frac{x^2}{2}$

### 1.3 L'Hôpital法则

当遇到 $\frac{0}{0}$ 或 $\frac{\infty}{\infty}$ 型未定式时：

$$\lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)}$$

## 2. 导数 (Derivatives)

### 2.1 导数的定义

函数 $f(x)$ 在点 $x_0$ 处的导数定义为：

$$f'(x_0) = \lim_{h \to 0} \frac{f(x_0 + h) - f(x_0)}{h}$$

### 2.2 基本导数公式

| 函数 | 导数 |
|------|------|
| $c$ (常数) | $0$ |
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $a^x$ | $a^x \ln a$ |
| $\ln x$ | $\frac{1}{x}$ |
| $\log_a x$ | $\frac{1}{x \ln a}$ |
| $\sin x$ | $\cos x$ |
| $\cos x$ | $-\sin x$ |
| $\tan x$ | $\sec^2 x$ |
| $\cot x$ | $-\csc^2 x$ |
| $\sec x$ | $\sec x \tan x$ |
| $\csc x$ | $-\csc x \cot x$ |
| $\arcsin x$ | $\frac{1}{\sqrt{1-x^2}}$ |
| $\arccos x$ | $-\frac{1}{\sqrt{1-x^2}}$ |
| $\arctan x$ | $\frac{1}{1+x^2}$ |

### 2.3 求导法则

#### 线性法则

$(af(x) + bg(x))' = af'(x) + bg'(x)$

#### 乘积法则 (Product Rule)

$(f(x)g(x))' = f'(x)g(x) + f(x)g'(x)$

#### 商法则 (Quotient Rule)

$$\left(\frac{f(x)}{g(x)}\right)' = \frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2}$$

#### 链式法则 (Chain Rule)

$(f(g(x)))' = f'(g(x)) \cdot g'(x)$

#### 反函数求导

如果 $y = f(x)$ 的反函数为 $x = f^{-1}(y)$，则：

$$\frac{dx}{dy} = \frac{1}{\frac{dy}{dx}}$$

### 2.4 高阶导数

- 二阶导数：$f''(x) = \frac{d^2f}{dx^2}$
- n阶导数：$f^{(n)}(x) = \frac{d^nf}{dx^n}$

## 3. 积分 (Integrals)

### 3.1 不定积分

#### 定义

如果 $F'(x) = f(x)$，则称 $F(x)$ 为 $f(x)$ 的一个原函数，记作：

$$\int f(x)dx = F(x) + C$$

#### 基本积分公式

| 函数 | 不定积分 |
|------|----------|
| $0$ | $C$ |
| $1$ | $x + C$ |
| $x^n$ (n≠-1) | $\frac{x^{n+1}}{n+1} + C$ |
| $\frac{1}{x}$ | $\ln\|x\| + C$ |
| $e^x$ | $e^x + C$ |
| $a^x$ | $\frac{a^x}{\ln a} + C$ |
| $\sin x$ | $-\cos x + C$ |
| $\cos x$ | $\sin x + C$ |
| $\tan x$ | $-\ln\|\cos x\| + C$ |
| $\cot x$ | $\ln\|\sin x\| + C$ |
| $\sec x$ | $\ln\|\sec x + \tan x\| + C$ |
| $\csc x$ | $-\ln\|\csc x + \cot x\| + C$ |
| $\sec^2 x$ | $\tan x + C$ |
| $\csc^2 x$ | $-\cot x + C$ |
| $\frac{1}{\sqrt{1-x^2}}$ | $\arcsin x + C$ |
| $\frac{1}{1+x^2}$ | $\arctan x + C$ |
| $\frac{1}{x\sqrt{x^2-1}}$ | $\sec^{-1}\|x\| + C$ |

### 3.2 积分法则

#### 线性性质

$$\int [af(x) + bg(x)]dx = a\int f(x)dx + b\int g(x)dx$$

#### 分部积分法

$$\int u dv = uv - \int v du$$

#### 换元积分法

$$\int f(g(x))g'(x)dx = \int f(u)du \quad (u = g(x))$$

### 3.3 定积分

#### 定义（Riemann积分）

$$\int_a^b f(x)dx = \lim_{n \to \infty} \sum_{i=1}^n f(x_i)\Delta x$$

#### 牛顿-莱布尼茨公式

$$\int_a^b f(x)dx = F(b) - F(a)$$

其中 $F'(x) = f(x)$

#### 定积分性质

- $\int_a^a f(x)dx = 0$
- $\int_a^b f(x)dx = -\int_b^a f(x)dx$
- $\int_a^b f(x)dx = \int_a^c f(x)dx + \int_c^b f(x)dx$

## 4. 微积分的应用

### 4.1 几何应用

#### 切线方程

在点 $(x_0, f(x_0))$ 处的切线方程：

$$y - f(x_0) = f'(x_0)(x - x_0)$$

#### 曲线的凹凸性

- 若 $f''(x) > 0$，曲线向上凹（凹函数）
- 若 $f''(x) < 0$，曲线向下凸（凸函数）

#### 面积计算

曲线 $y = f(x)$ 与 x 轴在区间 $[a,b]$ 围成的面积：

$$S = \int_a^b |f(x)|dx$$

#### 弧长计算

曲线 $y = f(x)$ 在区间 $[a,b]$ 的弧长：

$$L = \int_a^b \sqrt{1 + [f'(x)]^2}dx$$

#### 体积计算（旋转体）

绕 x 轴旋转：$V = \pi \int_a^b [f(x)]^2 dx$

绕 y 轴旋转：$V = 2\pi \int_a^b x \cdot f(x) dx$

### 4.2 物理应用

#### 速度与加速度

- 位置函数：$s(t)$
- 速度：$v(t) = s'(t)$  
- 加速度：$a(t) = v'(t) = s''(t)$

#### 功与能量

变力做功：$W = \int_a^b F(x)dx$

## 5. 常用技巧与方法

### 5.1 三角函数积分

- $\int \sin^m x \cos^n x dx$ 的积分技巧
- 万能公式：$t = \tan(\frac{x}{2})$

### 5.2 有理函数积分

部分分式分解法

### 5.3 无理函数积分

三角替换、根式替换

### 5.4 Taylor级数展开

$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots$$

常用展开：

- $e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$
- $\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots$
- $\cos x = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \cdots$
- $\ln(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots$

## 6. 重要定理

### 6.1 微分中值定理

#### Rolle定理

如果函数 $f(x)$ 在 $[a,b]$ 连续，在 $(a,b)$ 可导，且 $f(a) = f(b)$，则存在 $\xi \in (a,b)$ 使得 $f'(\xi) = 0$。

#### Lagrange中值定理

$$\frac{f(b) - f(a)}{b - a} = f'(\xi), \quad \xi \in (a,b)$$

### 6.2 积分中值定理

如果 $f(x)$ 在 $[a,b]$ 连续，则存在 $\xi \in [a,b]$ 使得：

$$\int_a^b f(x)dx = f(\xi)(b-a)$$
