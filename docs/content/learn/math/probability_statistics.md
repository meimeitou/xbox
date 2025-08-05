+++
date = '2025-08-05T18:36:34+08:00'
title = '概率统计基础'
description = '概率统计的基本概念和方法'
tags = ["math", "数学", "基础"]
categories = ["数学基础", "数学"]
+++

## 概率统计基础

### 一、基本概念

#### 1. 随机试验与样本空间

**随机试验（Random Experiment）**：满足以下条件的试验

- 可以在相同条件下重复进行
- 每次试验的可能结果不止一个，且能事先明确所有可能结果
- 进行一次试验之前不能确定会出现哪一个结果

**样本空间（Sample Space）**：随机试验所有可能结果组成的集合，记作 $\Omega$

**随机事件（Random Event）**：样本空间的子集，通常用大写字母 $A$、$B$、$C$ 等表示

#### 2. 事件的运算

- **并事件**：$A \cup B$ 表示事件 $A$ 或事件 $B$ 发生
- **交事件**：$A \cap B$ 表示事件 $A$ 和事件 $B$ 同时发生
- **对立事件**：$\overline{A}$ 表示事件 $A$ 不发生
- **差事件**：$A - B = A \cap \overline{B}$ 表示事件 $A$ 发生但事件 $B$ 不发生

### 二、概率的基本性质

#### 1. 概率的公理化定义

设 $\Omega$ 为样本空间，$P$ 为定义在事件域上的实值函数，如果满足：

1. **非负性**：对任意事件 $A$，有 $P(A) \geq 0$
2. **规范性**：$P(\Omega) = 1$
3. **可列可加性**：对于两两不相容的事件 $A_1, A_2, \ldots$，有
   $$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

#### 2. 概率的基本性质

- $P(\emptyset) = 0$（不可能事件的概率为0）
- $P(\overline{A}) = 1 - P(A)$（对立事件概率）
- $P(A - B) = P(A) - P(A \cap B)$
- **加法公式**：$P(A \cup B) = P(A) + P(B) - P(A \cap B)$
- **三事件加法公式**：
  $$P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)$$

#### 3. 条件概率

**定义**：设 $A$，$B$ 是两个事件，且 $P(B) > 0$，则称
$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$
为在事件 $B$ 发生的条件下事件 $A$ 发生的条件概率。

**乘法公式**：

- $P(A \cap B) = P(A|B)P(B) = P(B|A)P(A)$
- $P(A \cap B \cap C) = P(A)P(B|A)P(C|A \cap B)$

### 三、重要公式和定理

#### 1. 全概率公式

设 $B_1, B_2, \ldots, B_n$ 是样本空间 $\Omega$ 的一个划分，且 $P(B_i) > 0$，$i = 1, 2, \ldots, n$，则对任意事件 $A$：
$$P(A) = \sum_{i=1}^{n} P(A|B_i)P(B_i)$$

#### 2. 贝叶斯公式

在全概率公式的条件下，对于任意 $k \in \{1, 2, \ldots, n\}$：
$$P(B_k|A) = \frac{P(A|B_k)P(B_k)}{\sum_{i=1}^{n} P(A|B_i)P(B_i)}$$

#### 3. 事件的独立性

**两事件独立**：事件 $A$ 与 $B$ 独立当且仅当 $P(A \cap B) = P(A)P(B)$

**多事件独立**：事件 $A_1, A_2, \ldots, A_n$ 相互独立当且仅当对于任意的 $1 \leq i_1 < i_2 < \cdots < i_k \leq n$：
$$P(A_{i_1} \cap A_{i_2} \cap \cdots \cap A_{i_k}) = P(A_{i_1})P(A_{i_2})\cdots P(A_{i_k})$$

### 四、随机变量及其分布

#### 1. 随机变量的定义

**随机变量**：定义在样本空间 $\Omega$ 上的实值函数，通常用 $X$、$Y$、$Z$ 等表示。

#### 2. 分布函数

**分布函数**：$F(x) = P(X \leq x)$，$x \in \mathbb{R}$

**性质**：

- $F(x)$ 单调不减
- $0 \leq F(x) \leq 1$
- $F(-\infty) = 0$，$F(+\infty) = 1$
- $F(x)$ 右连续

#### 3. 离散型随机变量

**概率质量函数**：$p_i = P(X = x_i)$，$i = 1, 2, \ldots$

**性质**：

- $p_i \geq 0$
- $\sum_{i} p_i = 1$

**常见离散分布**：

| 分布 | 记号 | 概率质量函数 | 期望 | 方差 |
|------|------|-------------|------|------|
| 伯努利分布 | $B(1,p)$ | $P(X=k)=p^k(1-p)^{1-k}$, $k=0,1$ | $p$ | $p(1-p)$ |
| 二项分布 | $B(n,p)$ | $P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| 泊松分布 | $P(\lambda)$ | $P(X=k)=\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ |
| 几何分布 | $G(p)$ | $P(X=k)=(1-p)^{k-1}p$ | $\frac{1}{p}$ | $\frac{1-p}{p^2}$ |

#### 4. 连续型随机变量

**概率密度函数**：$f(x)$ 满足

- $f(x) \geq 0$
- $\int_{-\infty}^{+\infty} f(x)dx = 1$
- $P(a < X \leq b) = \int_a^b f(x)dx$

**常见连续分布**：

| 分布 | 记号 | 概率密度函数 | 期望 | 方差 |
|------|------|-------------|------|------|
| 均匀分布 | $U(a,b)$ | $f(x)=\frac{1}{b-a}$, $a \leq x \leq b$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| 正态分布 | $N(\mu,\sigma^2)$ | $f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ |
| 指数分布 | $E(\lambda)$ | $f(x)=\lambda e^{-\lambda x}$, $x \geq 0$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |

### 五、数字特征

#### 1. 数学期望

**离散型**：$E(X) = \sum_{i} x_i p_i$

**连续型**：$E(X) = \int_{-\infty}^{+\infty} x f(x)dx$

**性质**：

- $E(C) = C$（常数的期望等于常数）
- $E(X + Y) = E(X) + E(Y)$（线性性）
- $E(aX + b) = aE(X) + b$
- 若 $X$ 与 $Y$ 独立，则 $E(XY) = E(X)E(Y)$

#### 2. 方差

**定义**：$\text{Var}(X) = E[(X - E(X))^2] = E(X^2) - [E(X)]^2$

**性质**：

- $\text{Var}(C) = 0$
- $\text{Var}(aX + b) = a^2\text{Var}(X)$
- 若 $X$ 与 $Y$ 独立，则 $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

**标准差**：$\sigma(X) = \sqrt{\text{Var}(X)}$

#### 3. 协方差和相关系数

**协方差**：$\text{Cov}(X,Y) = E[(X-E(X))(Y-E(Y))] = E(XY) - E(X)E(Y)$

**相关系数**：$\rho(X,Y) = \frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}}$

**性质**：

- $-1 \leq \rho(X,Y) \leq 1$
- $\rho(X,Y) = 0$ 时，称 $X$ 和 $Y$ 不相关
- $|\rho(X,Y)| = 1$ 时，$X$ 和 $Y$ 线性相关

### 六、大数定律和中心极限定理

#### 1. 大数定律

**弱大数定律（辛钦大数定律）**：设 $X_1, X_2, \ldots$ 是独立同分布的随机变量序列，且 $E(X_i) = \mu$ 存在，则对任意 $\varepsilon > 0$：
$$\lim_{n \to \infty} P\left(\left|\frac{1}{n}\sum_{i=1}^n X_i - \mu\right| < \varepsilon\right) = 1$$

#### 2. 中心极限定理

**独立同分布中心极限定理**：设 $X_1, X_2, \ldots$ 是独立同分布的随机变量序列，$E(X_i) = \mu$，$\text{Var}(X_i) = \sigma^2 > 0$，则：
$$\lim_{n \to \infty} P\left(\frac{\sum_{i=1}^n X_i - n\mu}{\sigma\sqrt{n}} \leq x\right) = \Phi(x)$$

其中 $\Phi(x)$ 是标准正态分布的分布函数。

**棣莫弗-拉普拉斯定理**：设 $X_n \sim B(n,p)$，则当 $n$ 足够大时：
$$\frac{X_n - np}{\sqrt{np(1-p)}} \stackrel{d}{\to} N(0,1)$$

### 七、统计量和抽样分布

#### 1. 基本统计量

设 $X_1, X_2, \ldots, X_n$ 是来自总体 $X$ 的样本：

**样本均值**：$\overline{X} = \frac{1}{n}\sum_{i=1}^n X_i$

**样本方差**：$S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \overline{X})^2$

**样本标准差**：$S = \sqrt{S^2}$

#### 2. 重要抽样分布

**$\chi^2$ 分布**：若 $X_1, \ldots, X_n$ 独立且都服从 $N(0,1)$，则 $\sum_{i=1}^n X_i^2 \sim \chi^2(n)$

**$t$ 分布**：若 $X \sim N(0,1)$，$Y \sim \chi^2(n)$ 且相互独立，则 $\frac{X}{\sqrt{Y/n}} \sim t(n)$

**$F$ 分布**：若 $X \sim \chi^2(m)$，$Y \sim \chi^2(n)$ 且相互独立，则 $\frac{X/m}{Y/n} \sim F(m,n)$

#### 3. 正态总体的抽样分布

设 $X_1, \ldots, X_n$ 是来自 $N(\mu, \sigma^2)$ 的样本：

- $\overline{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right)$
- $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)$
- $\frac{\overline{X} - \mu}{S/\sqrt{n}} \sim t(n-1)$
