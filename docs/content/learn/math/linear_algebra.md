+++
date = '2025-08-05T18:39:26+08:00'
title = '线性代数基础'
description = '线性代数的基本概念和方法'
tags = ["math", "数学", "基础"]
categories = ["数学基础", "数学"]
+++

## 线性代数基础

### 一、向量与向量空间

#### 1. 向量的基本概念

**n维向量**：$\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}$ 或 $\mathbf{x} = (x_1, x_2, \ldots, x_n)^T$

**零向量**：$\mathbf{0} = (0, 0, \ldots, 0)^T$

**单位向量**：$\|\mathbf{e}\| = 1$

#### 2. 向量的运算

**向量加法**：$\mathbf{x} + \mathbf{y} = \begin{pmatrix} x_1 + y_1 \\ x_2 + y_2 \\ \vdots \\ x_n + y_n \end{pmatrix}$

**数量乘法**：$k\mathbf{x} = \begin{pmatrix} kx_1 \\ kx_2 \\ \vdots \\ kx_n \end{pmatrix}$

**内积（点积）**：$\mathbf{x} \cdot \mathbf{y} = \mathbf{x}^T\mathbf{y} = \sum_{i=1}^n x_i y_i$

**向量的模（长度）**：$\|\mathbf{x}\| = \sqrt{\mathbf{x} \cdot \mathbf{x}} = \sqrt{\sum_{i=1}^n x_i^2}$

#### 3. 向量的性质

- **交换律**：$\mathbf{x} + \mathbf{y} = \mathbf{y} + \mathbf{x}$
- **结合律**：$(\mathbf{x} + \mathbf{y}) + \mathbf{z} = \mathbf{x} + (\mathbf{y} + \mathbf{z})$
- **分配律**：$k(\mathbf{x} + \mathbf{y}) = k\mathbf{x} + k\mathbf{y}$
- **柯西-施瓦茨不等式**：$|\mathbf{x} \cdot \mathbf{y}| \leq \|\mathbf{x}\| \|\mathbf{y}\|$
- **三角不等式**：$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$

#### 4. 线性相关性

**线性组合**：$\mathbf{v} = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k$

**线性相关**：存在不全为零的系数 $c_1, c_2, \ldots, c_k$ 使得
$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0}$$

**线性无关**：只有当 $c_1 = c_2 = \cdots = c_k = 0$ 时，上式才成立

### 二、矩阵与矩阵运算

#### 1. 矩阵的基本概念

**m×n矩阵**：$A = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix} = (a_{ij})_{m \times n}$

**特殊矩阵**：

- **零矩阵**：$O$，所有元素都为0
- **单位矩阵**：$I_n = \begin{pmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{pmatrix}$
- **对角矩阵**：除主对角线外其他元素都为0
- **上三角矩阵**：主对角线下方元素都为0
- **下三角矩阵**：主对角线上方元素都为0

#### 2. 矩阵运算

**矩阵加法**：$(A + B)_{ij} = a_{ij} + b_{ij}$

**数量乘法**：$(kA)_{ij} = ka_{ij}$

**矩阵乘法**：$(AB)_{ij} = \sum_{k=1}^p a_{ik}b_{kj}$（其中 $A$ 是 $m \times p$ 矩阵，$B$ 是 $p \times n$ 矩阵）

**矩阵转置**：$(A^T)_{ij} = a_{ji}$

#### 3. 矩阵运算的性质

- $(A + B) + C = A + (B + C)$
- $A + B = B + A$
- $(AB)C = A(BC)$
- $A(B + C) = AB + AC$
- $(A + B)C = AC + BC$
- $(AB)^T = B^T A^T$
- $(A^T)^T = A$

#### 4. 逆矩阵

**定义**：对于 $n$ 阶方阵 $A$，如果存在 $n$ 阶方阵 $B$ 使得 $AB = BA = I$，则称 $A$ 可逆，$B$ 为 $A$ 的逆矩阵，记作 $A^{-1}$

**性质**：

- $(A^{-1})^{-1} = A$
- $(AB)^{-1} = B^{-1}A^{-1}$
- $(A^T)^{-1} = (A^{-1})^T$
- $(kA)^{-1} = \frac{1}{k}A^{-1}$（$k \neq 0$）

**二阶矩阵逆矩阵公式**：
$$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}, \quad A^{-1} = \frac{1}{ad-bc}\begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

### 三、行列式

#### 1. 行列式的定义

**二阶行列式**：$\begin{vmatrix} a & b \\ c & d \end{vmatrix} = ad - bc$

**三阶行列式**：$\begin{vmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{vmatrix} = a_{11}\begin{vmatrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{vmatrix} - a_{12}\begin{vmatrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{vmatrix} + a_{13}\begin{vmatrix} a_{21} & a_{22} \\ a_{31} & a_{32} \end{vmatrix}$

**n阶行列式**：$\det(A) = \sum_{\sigma} \text{sgn}(\sigma) \prod_{i=1}^n a_{i,\sigma(i)}$

#### 2. 行列式的性质

- $\det(A^T) = \det(A)$
- $\det(AB) = \det(A)\det(B)$
- $\det(kA) = k^n\det(A)$（对于 $n$ 阶矩阵）
- $\det(A^{-1}) = \frac{1}{\det(A)}$
- 交换两行（列），行列式变号
- 某行（列）乘以常数 $k$，行列式乘以 $k$
- 某行（列）的倍数加到另一行（列），行列式不变

#### 3. 克拉默法则

对于线性方程组 $A\mathbf{x} = \mathbf{b}$，当 $\det(A) \neq 0$ 时，有唯一解：
$$x_i = \frac{\det(A_i)}{\det(A)}$$

其中 $A_i$ 是将 $A$ 的第 $i$ 列替换为 $\mathbf{b}$ 得到的矩阵。

### 四、线性方程组

#### 1. 齐次线性方程组

**形式**：$A\mathbf{x} = \mathbf{0}$

**解的性质**：

- 总有零解 $\mathbf{x} = \mathbf{0}$
- 当 $\det(A) \neq 0$ 时，只有零解
- 当 $\det(A) = 0$ 时，有无穷多解

#### 2. 非齐次线性方程组

**形式**：$A\mathbf{x} = \mathbf{b}$（$\mathbf{b} \neq \mathbf{0}$）

**解的判定**：

- 当 $\text{rank}(A) = \text{rank}([A|\mathbf{b}]) = n$ 时，有唯一解
- 当 $\text{rank}(A) = \text{rank}([A|\mathbf{b}]) < n$ 时，有无穷多解
- 当 $\text{rank}(A) < \text{rank}([A|\mathbf{b}])$ 时，无解

#### 3. 解空间

**齐次方程组解空间**：所有解向量构成的向量空间
**基础解系**：解空间的一组基
**通解**：基础解系的线性组合

### 五、特征值与特征向量

#### 1. 定义

对于 $n$ 阶方阵 $A$，如果存在非零向量 $\mathbf{v}$ 和数 $\lambda$ 使得：
$$A\mathbf{v} = \lambda\mathbf{v}$$

则称 $\lambda$ 为 $A$ 的**特征值**，$\mathbf{v}$ 为对应的**特征向量**。

#### 2. 特征多项式

**特征方程**：$\det(A - \lambda I) = 0$

**特征多项式**：$p(\lambda) = \det(A - \lambda I)$

#### 3. 特征值的性质

- $\sum_{i=1}^n \lambda_i = \text{tr}(A)$（迹等于特征值之和）
- $\prod_{i=1}^n \lambda_i = \det(A)$（行列式等于特征值之积）
- 不同特征值对应的特征向量线性无关

#### 4. 矩阵对角化

**对角化条件**：$n$ 阶矩阵 $A$ 可对角化当且仅当 $A$ 有 $n$ 个线性无关的特征向量

**对角化过程**：$P^{-1}AP = D$，其中 $P$ 的列向量是特征向量，$D$ 是对角矩阵

### 六、二次型

#### 1. 二次型的定义

**二次型**：$f(x_1, x_2, \ldots, x_n) = \sum_{i=1}^n \sum_{j=1}^n a_{ij}x_i x_j = \mathbf{x}^T A \mathbf{x}$

其中 $A = (a_{ij})$ 是对称矩阵。

#### 2. 二次型的标准形

通过正交变换 $\mathbf{x} = P\mathbf{y}$，可将二次型化为标准形：
$$f = \lambda_1 y_1^2 + \lambda_2 y_2^2 + \cdots + \lambda_n y_n^2$$

其中 $\lambda_1, \lambda_2, \ldots, \lambda_n$ 是矩阵 $A$ 的特征值。

#### 3. 二次型的分类

- **正定**：对所有非零向量 $\mathbf{x}$，有 $\mathbf{x}^T A \mathbf{x} > 0$
- **负定**：对所有非零向量 $\mathbf{x}$，有 $\mathbf{x}^T A \mathbf{x} < 0$
- **半正定**：对所有向量 $\mathbf{x}$，有 $\mathbf{x}^T A \mathbf{x} \geq 0$
- **半负定**：对所有向量 $\mathbf{x}$，有 $\mathbf{x}^T A \mathbf{x} \leq 0$
- **不定**：既有正值又有负值

**判定准则**：

- 正定 ⟺ 所有特征值都大于0 ⟺ 所有主子式都大于0
- 负定 ⟺ 所有特征值都小于0 ⟺ 奇数阶主子式小于0，偶数阶主子式大于0

### 七、向量空间

#### 1. 向量空间的定义

向量空间 $V$ 是满足以下条件的集合：

- **加法封闭性**：$\mathbf{u}, \mathbf{v} \in V \Rightarrow \mathbf{u} + \mathbf{v} \in V$
- **数乘封闭性**：$\mathbf{v} \in V, k \in \mathbb{R} \Rightarrow k\mathbf{v} \in V$
- **加法交换律**：$\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$
- **加法结合律**：$(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
- **零元存在**：存在 $\mathbf{0} \in V$ 使得 $\mathbf{v} + \mathbf{0} = \mathbf{v}$
- **逆元存在**：对每个 $\mathbf{v} \in V$，存在 $-\mathbf{v} \in V$ 使得 $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$

#### 2. 子空间

**子空间**：向量空间 $V$ 的非空子集 $W$，如果对加法和数乘运算封闭，则 $W$ 是 $V$ 的子空间。

#### 3. 基与维数

**基**：向量空间 $V$ 的一组线性无关的向量，且 $V$ 中任意向量都可表示为这组向量的线性组合

**维数**：基中向量的个数，记作 $\dim(V)$

#### 4. 内积空间

**内积**：满足以下性质的运算 $\langle \cdot, \cdot \rangle$：

- **正定性**：$\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$，等号成立当且仅当 $\mathbf{v} = \mathbf{0}$
- **对称性**：$\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$
- **线性性**：$\langle a\mathbf{u} + b\mathbf{v}, \mathbf{w} \rangle = a\langle \mathbf{u}, \mathbf{w} \rangle + b\langle \mathbf{v}, \mathbf{w} \rangle$

**正交**：$\langle \mathbf{u}, \mathbf{v} \rangle = 0$

**标准正交基**：基中向量两两正交且都是单位向量

### 八、线性变换

#### 1. 线性变换的定义

映射 $T: V \rightarrow W$ 称为线性变换，如果：

- $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
- $T(k\mathbf{v}) = kT(\mathbf{v})$

#### 2. 线性变换的矩阵表示

如果 $T: \mathbb{R}^n \rightarrow \mathbb{R}^m$ 是线性变换，则存在矩阵 $A$ 使得：
$$T(\mathbf{x}) = A\mathbf{x}$$

#### 3. 核与像

**核（零空间）**：$\ker(T) = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}\}$

**像（值域）**：$\text{Im}(T) = \{T(\mathbf{v}) : \mathbf{v} \in V\}$

**维数定理**：$\dim(V) = \dim(\ker(T)) + \dim(\text{Im}(T))$
