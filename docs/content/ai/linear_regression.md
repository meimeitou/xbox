+++
title = '线性回归'
math = true
+++

线性回归：用线性函数预测生活

- [引言](#引言)
- [什么是线性回归？](#什么是线性回归)
  - [数学表达式](#数学表达式)
  - [生活中的线性关系](#生活中的线性关系)
- [从散点图到直线：直观理解](#从散点图到直线直观理解)
- [线性回归的核心思想](#线性回归的核心思想)
  - [1. 最佳拟合直线](#1-最佳拟合直线)
  - [2. 最小二乘法](#2-最小二乘法)
- [多元线性回归：更复杂的现实](#多元线性回归更复杂的现实)
- [手动实现线性回归](#手动实现线性回归)
- [梯度下降法求解线性回归](#梯度下降法求解线性回归)
- [模型评估与诊断](#模型评估与诊断)
  - [1. 评估指标](#1-评估指标)
  - [2. 过拟合与正则化](#2-过拟合与正则化)
- [实际应用案例](#实际应用案例)
  - [房价预测完整案例](#房价预测完整案例)
- [线性回归的假设与局限性](#线性回归的假设与局限性)
  - [基本假设](#基本假设)
  - [假设检验](#假设检验)
- [总结：线性回归的核心要点](#总结线性回归的核心要点)
  - [🎯 核心思想](#-核心思想)
  - [📊 关键概念](#-关键概念)
  - [💡 实用技巧](#-实用技巧)
  - [🎪 应用场景](#-应用场景)
  - [🔧 记忆口诀](#-记忆口诀)

## 引言

想象一下，你正在买房，房产经纪人告诉你："这个区域，房子每大10平方米，价格大约多10万元。"这个简单的描述，其实就包含了线性回归的核心思想！

线性回归是机器学习中最基础、最重要的算法之一。它简单易懂，却威力无穷，是通往复杂算法的必经之路。

## 什么是线性回归？

线性回归（Linear Regression）是一种**预测**方法，它假设目标变量与输入特征之间存在**线性关系**。

### 数学表达式

最简单的线性回归方程：

```
y = wx + b
```

其中：

- `y`：我们要预测的目标（房价、成绩、销量等）
- `x`：输入特征（面积、学习时间、广告费用等）
- `w`：权重/斜率（每单位x对y的影响）
- `b`：偏置/截距（基础值）

### 生活中的线性关系

1. **房价与面积** 🏠

   ```
   房价 = 单价 × 面积 + 基础费用
   ```

2. **考试成绩与学习时间** 📚

   ```
   成绩 = 效率 × 学习时间 + 基础水平
   ```

3. **工资与工作年限** 💼

   ```
   工资 = 年增长率 × 工作年限 + 起薪
   ```

4. **汽车油耗与速度** 🚗

   ```
   油耗 = 速度系数 × 速度 + 基础油耗
   ```

## 从散点图到直线：直观理解

让我们从一个简单的例子开始：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建示例数据：房屋面积与价格
np.random.seed(42)
area = np.random.uniform(50, 200, 100)  # 面积：50-200平方米
noise = np.random.normal(0, 10, 100)    # 随机噪声
price = 2 * area + 50 + noise           # 价格 = 2万/平米 × 面积 + 50万基础 + 噪声

# 可视化数据
plt.figure(figsize=(12, 8))

# 子图1：散点图
plt.subplot(2, 2, 1)
plt.scatter(area, price, alpha=0.6, color='blue')
plt.xlabel('房屋面积 (平方米)')
plt.ylabel('房价 (万元)')
plt.title('房价与面积的关系（散点图）')
plt.grid(True, alpha=0.3)

# 子图2：添加拟合直线
plt.subplot(2, 2, 2)
plt.scatter(area, price, alpha=0.6, color='blue', label='实际数据')

# 拟合线性回归
model = LinearRegression()
area_2d = area.reshape(-1, 1)  # sklearn需要2D数组
model.fit(area_2d, price)

# 预测并绘制直线
area_line = np.linspace(50, 200, 100)
price_pred = model.predict(area_line.reshape(-1, 1))
plt.plot(area_line, price_pred, 'r-', linewidth=2, label=f'拟合直线: y={model.coef_[0]:.2f}x+{model.intercept_:.2f}')

plt.xlabel('房屋面积 (平方米)')
plt.ylabel('房价 (万元)')
plt.title('线性回归拟合结果')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图3：残差图
plt.subplot(2, 2, 3)
predictions = model.predict(area_2d)
residuals = price - predictions
plt.scatter(area, residuals, alpha=0.6, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('房屋面积 (平方米)')
plt.ylabel('残差 (实际值 - 预测值)')
plt.title('残差分析')
plt.grid(True, alpha=0.3)

# 子图4：预测效果
plt.subplot(2, 2, 4)
plt.scatter(price, predictions, alpha=0.6)
plt.plot([price.min(), price.max()], [price.min(), price.max()], 'r--', linewidth=2)
plt.xlabel('实际房价 (万元)')
plt.ylabel('预测房价 (万元)')
plt.title('预测 vs 实际')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"拟合结果:")
print(f"斜率(每平米价格): {model.coef_[0]:.2f} 万元/平米")
print(f"截距(基础价格): {model.intercept_:.2f} 万元")
print(f"R²分数: {model.score(area_2d, price):.3f}")
```

## 线性回归的核心思想

### 1. 最佳拟合直线

线性回归的目标是找到一条**最佳拟合直线**，使得：

- 直线尽可能接近所有数据点
- 预测误差最小

### 2. 最小二乘法

我们通过**最小化平方误差**来找到最佳直线：

```python
def visualize_least_squares():
    """可视化最小二乘法的原理"""
    
    # 简单数据
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])
    
    # 拟合直线
    model = LinearRegression()
    x_2d = x.reshape(-1, 1)
    model.fit(x_2d, y)
    y_pred = model.predict(x_2d)
    
    plt.figure(figsize=(12, 5))
    
    # 左图：显示误差
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, color='blue', s=100, label='实际数据点')
    plt.plot(x, y_pred, 'r-', linewidth=2, label='拟合直线')
    
    # 绘制误差线
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y[i], y_pred[i]], 'g--', alpha=0.7)
        plt.text(x[i]+0.1, (y[i]+y_pred[i])/2, f'误差:{y[i]-y_pred[i]:.1f}', fontsize=8)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('最小二乘法：最小化误差平方和')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右图：不同直线的误差对比
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, color='blue', s=100, label='实际数据')
    
    # 几条不同的直线
    slopes = [0.5, 0.8, 1.0]  # 不同斜率
    colors = ['orange', 'green', 'red']
    
    for slope, color in zip(slopes, colors):
        y_line = slope * x + 1
        mse = np.mean((y - y_line)**2)
        plt.plot(x, y_line, color=color, label=f'斜率={slope}, MSE={mse:.2f}')
    
    # 最优直线
    mse_best = np.mean((y - y_pred)**2)
    plt.plot(x, y_pred, 'purple', linewidth=3, 
             label=f'最优直线, MSE={mse_best:.2f}')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('不同直线的误差对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("最小二乘法找到误差最小的直线!")

visualize_least_squares()
```

## 多元线性回归：更复杂的现实

现实中，目标往往受多个因素影响：

```python
def multiple_linear_regression_demo():
    """多元线性回归示例"""
    
    # 创建多特征数据：房价受面积、房间数、楼层影响
    np.random.seed(42)
    n_samples = 200
    
    area = np.random.uniform(50, 200, n_samples)        # 面积
    rooms = np.random.randint(1, 6, n_samples)          # 房间数
    floor = np.random.randint(1, 21, n_samples)         # 楼层
    
    # 真实的价格模型（我们预先设定的规律）
    true_price = (2.0 * area +           # 面积影响：2万/平米
                  5.0 * rooms +          # 房间影响：5万/间
                  0.2 * floor +          # 楼层影响：0.2万/层
                  30)                    # 基础价格：30万
    
    # 添加噪声
    noise = np.random.normal(0, 8, n_samples)
    observed_price = true_price + noise
    
    # 组织数据
    X = np.column_stack([area, rooms, floor])
    feature_names = ['面积', '房间数', '楼层']
    
    # 训练模型
    model = LinearRegression()
    model.fit(X, observed_price)
    
    # 预测
    predictions = model.predict(X)
    
    # 分析结果
    print("多元线性回归结果分析:")
    print("=" * 50)
    print("真实系数 vs 学习到的系数:")
    true_coeffs = [2.0, 5.0, 0.2]
    for i, name in enumerate(feature_names):
        print(f"{name:>6}: 真实={true_coeffs[i]:6.2f}, 学习={model.coef_[i]:6.2f}")
    
    print(f"截距项: 真实={30:6.2f}, 学习={model.intercept_:6.2f}")
    print(f"R²分数: {model.score(X, observed_price):.3f}")
    
    # 可视化各特征的影响
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (name, ax) in enumerate(zip(feature_names, axes)):
        ax.scatter(X[:, i], observed_price, alpha=0.6)
        
        # 绘制单变量趋势（固定其他变量）
        x_range = np.linspace(X[:, i].min(), X[:, i].max(), 100)
        
        # 创建预测用的数据（其他特征取均值）
        X_temp = np.zeros((100, 3))
        X_temp[:, i] = x_range
        for j in range(3):
            if j != i:
                X_temp[:, j] = np.mean(X[:, j])
        
        y_trend = model.predict(X_temp)
        ax.plot(x_range, y_trend, 'r-', linewidth=2)
        
        ax.set_xlabel(name)
        ax.set_ylabel('房价 (万元)')
        ax.set_title(f'{name}与房价的关系')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 特征重要性分析
    feature_importance = np.abs(model.coef_)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(feature_names, feature_importance, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.title('特征重要性（系数绝对值）')
    plt.ylabel('系数绝对值')
    
    # 在柱子上显示数值
    for bar, coef in zip(bars, model.coef_):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{coef:.2f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.show()

multiple_linear_regression_demo()
```

## 手动实现线性回归

让我们从头实现线性回归算法，理解其内部工作原理：

```python
class SimpleLinearRegression:
    """手动实现的简单线性回归"""
    
    def __init__(self):
        self.slope = None
        self.intercept = None
        
    def fit(self, X, y):
        """训练模型（最小二乘法）"""
        # 计算均值
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # 计算分子和分母
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean) ** 2)
        
        # 计算斜率和截距
        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean
        
        return self
    
    def predict(self, X):
        """预测"""
        return self.slope * X + self.intercept
    
    def score(self, X, y):
        """计算R²分数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)  # 残差平方和
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # 总平方和
        return 1 - (ss_res / ss_tot)

class MultipleLinearRegression:
    """手动实现的多元线性回归"""
    
    def __init__(self):
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        """使用正规方程训练模型"""
        # 添加偏置列（全1列）
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        # 正规方程: θ = (X^T X)^(-1) X^T y
        XtX = X_with_bias.T @ X_with_bias
        Xty = X_with_bias.T @ y
        theta = np.linalg.solve(XtX, Xty)
        
        self.intercept = theta[0]
        self.coefficients = theta[1:]
        
        return self
    
    def predict(self, X):
        """预测"""
        return X @ self.coefficients + self.intercept
    
    def score(self, X, y):
        """计算R²分数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# 测试手动实现
def test_manual_implementation():
    """测试手动实现的线性回归"""
    
    # 生成测试数据
    np.random.seed(42)
    X_simple = np.random.randn(100)
    y_simple = 2 * X_simple + 1 + 0.1 * np.random.randn(100)
    
    # 简单线性回归对比
    manual_simple = SimpleLinearRegression()
    manual_simple.fit(X_simple, y_simple)
    
    sklearn_simple = LinearRegression()
    sklearn_simple.fit(X_simple.reshape(-1, 1), y_simple)
    
    print("简单线性回归对比:")
    print(f"手动实现 - 斜率: {manual_simple.slope:.4f}, 截距: {manual_simple.intercept:.4f}")
    print(f"sklearn  - 斜率: {sklearn_simple.coef_[0]:.4f}, 截距: {sklearn_simple.intercept:.4f}")
    print(f"R²分数对比 - 手动: {manual_simple.score(X_simple, y_simple):.4f}, sklearn: {sklearn_simple.score(X_simple.reshape(-1, 1), y_simple):.4f}")
    
    # 多元线性回归对比
    X_multi = np.random.randn(100, 3)
    y_multi = X_multi @ [1, 2, -1] + 0.5 + 0.1 * np.random.randn(100)
    
    manual_multi = MultipleLinearRegression()
    manual_multi.fit(X_multi, y_multi)
    
    sklearn_multi = LinearRegression()
    sklearn_multi.fit(X_multi, y_multi)
    
    print(f"\n多元线性回归对比:")
    print(f"手动实现系数: {manual_multi.coefficients}")
    print(f"sklearn 系数: {sklearn_multi.coef_}")
    print(f"截距对比 - 手动: {manual_multi.intercept:.4f}, sklearn: {sklearn_multi.intercept:.4f}")

test_manual_implementation()
```

## 梯度下降法求解线性回归

除了正规方程，我们还可以用梯度下降法求解：

```python
class LinearRegressionGD:
    """使用梯度下降的线性回归"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.coefficients = None
        self.intercept = None
        self.cost_history = []
    
    def fit(self, X, y):
        """使用梯度下降训练模型"""
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.coefficients = np.zeros(n_features)
        self.intercept = 0
        
        # 梯度下降
        for i in range(self.max_iterations):
            # 预测
            y_pred = self.predict(X)
            
            # 计算成本（均方误差）
            cost = np.mean((y - y_pred) ** 2)
            self.cost_history.append(cost)
            
            # 计算梯度
            dw = -(2/n_samples) * X.T @ (y - y_pred)
            db = -(2/n_samples) * np.sum(y - y_pred)
            
            # 更新参数
            self.coefficients -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db
        
        return self
    
    def predict(self, X):
        return X @ self.coefficients + self.intercept

def demonstrate_gradient_descent():
    """演示梯度下降过程"""
    
    # 生成数据
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = X @ [3, -2] + 1 + 0.1 * np.random.randn(100)
    
    # 标准化特征
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # 训练模型
    model_gd = LinearRegressionGD(learning_rate=0.1, max_iterations=1000)
    model_gd.fit(X_scaled, y)
    
    # 对比sklearn
    model_sklearn = LinearRegression()
    model_sklearn.fit(X_scaled, y)
    
    print("梯度下降 vs sklearn对比:")
    print(f"梯度下降系数: {model_gd.coefficients}")
    print(f"sklearn系数: {model_sklearn.coef_}")
    print(f"梯度下降截距: {model_gd.intercept:.4f}")
    print(f"sklearn截距: {model_sklearn.intercept:.4f}")
    
    # 可视化成本函数收敛过程
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(model_gd.cost_history)
    plt.title('成本函数收敛过程')
    plt.xlabel('迭代次数')
    plt.ylabel('均方误差')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(model_gd.cost_history[:100])  # 前100次迭代
    plt.title('前100次迭代的收敛')
    plt.xlabel('迭代次数')
    plt.ylabel('均方误差')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

demonstrate_gradient_descent()
```

## 模型评估与诊断

### 1. 评估指标

```python
def regression_metrics():
    """回归模型的各种评估指标"""
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # 生成测试数据
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y_true = X @ [1, 2, -1] + 1 + 0.2 * np.random.randn(100)
    
    # 训练模型
    model = LinearRegression()
    model.fit(X, y_true)
    y_pred = model.predict(X)
    
    # 计算各种指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 手动计算
    residuals = y_true - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    print("回归模型评估指标:")
    print("=" * 40)
    print(f"均方误差 (MSE):     {mse:.4f}")
    print(f"均方根误差 (RMSE):   {rmse:.4f}")
    print(f"平均绝对误差 (MAE):  {mae:.4f}")
    print(f"决定系数 (R²):       {r2:.4f}")
    print(f"调整R² (n=100, p=3): {1 - (1-r2)*(100-1)/(100-3-1):.4f}")
    
    # 可视化评估
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 预测vs实际
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    axes[0, 0].set_xlabel('实际值')
    axes[0, 0].set_ylabel('预测值')
    axes[0, 0].set_title('预测 vs 实际')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 残差图
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('预测值')
    axes[0, 1].set_ylabel('残差')
    axes[0, 1].set_title('残差图')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 残差分布
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('残差')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].set_title('残差分布')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q图（正态性检验）
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q图（正态性检验）')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

regression_metrics()
```

### 2. 过拟合与正则化

```python
def regularization_demo():
    """演示正则化对过拟合的影响"""
    
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    
    # 生成非线性数据
    np.random.seed(42)
    X = np.linspace(0, 1, 100).reshape(-1, 1)
    y = np.sin(2 * np.pi * X).ravel() + 0.2 * np.random.randn(100)
    
    # 分割训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 创建不同复杂度的模型
    degrees = [1, 5, 10, 15]
    
    plt.figure(figsize=(16, 12))
    
    for i, degree in enumerate(degrees):
        # 普通线性回归
        poly_reg = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        
        # Ridge回归
        ridge_reg = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('ridge', Ridge(alpha=0.1))
        ])
        
        # Lasso回归
        lasso_reg = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('lasso', Lasso(alpha=0.01))
        ])
        
        # 训练模型
        poly_reg.fit(X_train, y_train)
        ridge_reg.fit(X_train, y_train)
        lasso_reg.fit(X_train, y_train)
        
        # 生成预测曲线
        X_plot = np.linspace(0, 1, 300).reshape(-1, 1)
        y_poly = poly_reg.predict(X_plot)
        y_ridge = ridge_reg.predict(X_plot)
        y_lasso = lasso_reg.predict(X_plot)
        
        # 绘图
        plt.subplot(3, 4, i + 1)
        plt.scatter(X_train, y_train, alpha=0.6, label='训练数据')
        plt.scatter(X_test, y_test, alpha=0.6, color='red', label='测试数据')
        plt.plot(X_plot, y_poly, label='普通回归')
        plt.title(f'普通回归 (度数={degree})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, i + 5)
        plt.scatter(X_train, y_train, alpha=0.6, label='训练数据')
        plt.scatter(X_test, y_test, alpha=0.6, color='red', label='测试数据')
        plt.plot(X_plot, y_ridge, color='green', label='Ridge回归')
        plt.title(f'Ridge回归 (度数={degree})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, i + 9)
        plt.scatter(X_train, y_train, alpha=0.6, label='训练数据')
        plt.scatter(X_test, y_test, alpha=0.6, color='red', label='测试数据')
        plt.plot(X_plot, y_lasso, color='orange', label='Lasso回归')
        plt.title(f'Lasso回归 (度数={degree})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 计算分数
        print(f"度数 {degree}:")
        print(f"  普通回归 - 训练R²: {poly_reg.score(X_train, y_train):.3f}, 测试R²: {poly_reg.score(X_test, y_test):.3f}")
        print(f"  Ridge回归 - 训练R²: {ridge_reg.score(X_train, y_train):.3f}, 测试R²: {ridge_reg.score(X_test, y_test):.3f}")
        print(f"  Lasso回归 - 训练R²: {lasso_reg.score(X_train, y_train):.3f}, 测试R²: {lasso_reg.score(X_test, y_test):.3f}")
        print()
    
    plt.tight_layout()
    plt.show()

regularization_demo()
```

## 实际应用案例

### 房价预测完整案例

```python
def house_price_prediction():
    """完整的房价预测案例"""
    
    # 创建更真实的房价数据
    np.random.seed(42)
    n_samples = 1000
    
    # 特征工程
    area = np.random.normal(120, 30, n_samples)  # 面积
    rooms = np.random.randint(1, 6, n_samples)   # 房间数
    age = np.random.randint(0, 30, n_samples)    # 房龄
    distance = np.random.exponential(5, n_samples)  # 距离市中心距离
    
    # 复杂的价格模型
    base_price = 50  # 基础价格
    area_effect = 2.5 * area  # 面积效应
    room_effect = 8 * rooms   # 房间效应
    age_effect = -0.8 * age   # 房龄效应（负效应）
    distance_effect = -2 * distance  # 距离效应（负效应）
    
    # 非线性效应
    luxury_effect = np.where(area > 150, (area - 150) * 1.5, 0)  # 豪宅效应
    
    true_price = (base_price + area_effect + room_effect + 
                  age_effect + distance_effect + luxury_effect)
    
    # 添加噪声
    noise = np.random.normal(0, 10, n_samples)
    observed_price = true_price + noise
    
    # 组织数据
    X = np.column_stack([area, rooms, age, distance])
    feature_names = ['面积(m²)', '房间数', '房龄(年)', '距离(km)']
    
    # 数据分割
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, observed_price, test_size=0.2, random_state=42)
    
    # 特征标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练多个模型
    models = {
        '线性回归': LinearRegression(),
        'Ridge回归': Ridge(alpha=1.0),
        'Lasso回归': Lasso(alpha=0.1)
    }
    
    results = {}
    
    print("房价预测模型对比:")
    print("=" * 60)
    
    for name, model in models.items():
        # 训练
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # 评估
        train_r2 = model.score(X_train_scaled, y_train)
        test_r2 = model.score(X_test_scaled, y_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        }
        
        print(f"{name}:")
        print(f"  训练R²: {train_r2:.3f}, 测试R²: {test_r2:.3f}")
        print(f"  训练RMSE: {train_rmse:.2f}, 测试RMSE: {test_rmse:.2f}")
        print(f"  特征系数: {model.coef_}")
        print()
    
    # 选择最佳模型
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_model = results[best_model_name]['model']
    
    print(f"最佳模型: {best_model_name}")
    
    # 特征重要性分析
    feature_importance = np.abs(best_model.coef_)
    
    plt.figure(figsize=(15, 10))
    
    # 特征重要性
    plt.subplot(2, 3, 1)
    bars = plt.bar(feature_names, feature_importance)
    plt.title('特征重要性')
    plt.xticks(rotation=45)
    for bar, coef in zip(bars, best_model.coef_):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{coef:.2f}', ha='center', va='bottom')
    
    # 预测vs实际
    plt.subplot(2, 3, 2)
    y_test_pred = best_model.predict(X_test_scaled)
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title('预测效果')
    
    # 残差分析
    plt.subplot(2, 3, 3)
    residuals = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测价格')
    plt.ylabel('残差')
    plt.title('残差分析')
    
    # 各特征与价格的关系
    for i in range(4):
        plt.subplot(2, 3, 4 + i if i < 2 else 5 + i)
        plt.scatter(X_test[:, i], y_test, alpha=0.6)
        plt.xlabel(feature_names[i])
        plt.ylabel('价格')
        plt.title(f'{feature_names[i]}与价格关系')
    
    plt.tight_layout()
    plt.show()
    
    # 实际预测示例
    print("\n实际预测示例:")
    print("-" * 30)
    
    # 几个示例房屋
    examples = [
        [100, 3, 5, 2],   # 100平米，3室，5年房龄，距离市中心2km
        [150, 4, 10, 1],  # 150平米，4室，10年房龄，距离市中心1km
        [80, 2, 20, 8],   # 80平米，2室，20年房龄，距离市中心8km
    ]
    
    for i, example in enumerate(examples):
        example_scaled = scaler.transform([example])
        predicted_price = best_model.predict(example_scaled)[0]
        
        print(f"房屋 {i+1}: {feature_names[0]}={example[0]}, {feature_names[1]}={example[1]}, "
              f"{feature_names[2]}={example[2]}, {feature_names[3]}={example[3]:.1f}")
        print(f"预测价格: {predicted_price:.1f}万元")
        print()

house_price_prediction()
```

## 线性回归的假设与局限性

### 基本假设

1. **线性关系**：特征与目标之间存在线性关系
2. **独立性**：观测值之间相互独立
3. **同方差性**：残差的方差是常数
4. **正态性**：残差服从正态分布
5. **无多重共线性**：特征之间不高度相关

### 假设检验

```python
def check_assumptions():
    """检验线性回归的基本假设"""
    
    # 生成违反假设的数据
    np.random.seed(42)
    n = 200
    x = np.linspace(0, 10, n)
    
    # 违反线性假设（真实关系是二次的）
    y_nonlinear = 0.5 * x**2 + 2 * x + 1 + np.random.normal(0, 2, n)
    
    # 违反同方差假设（方差随x增大）
    y_heteroskedastic = 2 * x + 1 + np.random.normal(0, 0.5 * x, n)
    
    # 正常数据
    y_normal = 2 * x + 1 + np.random.normal(0, 2, n)
    
    datasets = [
        ('正常数据', y_normal),
        ('非线性数据', y_nonlinear),
        ('异方差数据', y_heteroskedastic)
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for i, (name, y) in enumerate(datasets):
        # 拟合模型
        model = LinearRegression()
        X = x.reshape(-1, 1)
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # 原数据散点图
        axes[i, 0].scatter(x, y, alpha=0.6)
        axes[i, 0].plot(x, y_pred, 'r-', linewidth=2)
        axes[i, 0].set_title(f'{name} - 拟合结果')
        axes[i, 0].set_xlabel('x')
        axes[i, 0].set_ylabel('y')
        axes[i, 0].grid(True, alpha=0.3)
        
        # 残差图
        axes[i, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[i, 1].axhline(y=0, color='r', linestyle='--')
        axes[i, 1].set_title(f'{name} - 残差图')
        axes[i, 1].set_xlabel('预测值')
        axes[i, 1].set_ylabel('残差')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Q-Q图
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[i, 2])
        axes[i, 2].set_title(f'{name} - Q-Q图')
        axes[i, 2].grid(True, alpha=0.3)
        
        # 统计检验
        _, p_value = stats.shapiro(residuals)
        print(f"{name}:")
        print(f"  Shapiro-Wilk正态性检验 p值: {p_value:.4f}")
        print(f"  R²分数: {model.score(X, y):.4f}")
        print()
    
    plt.tight_layout()
    plt.show()

check_assumptions()
```

## 总结：线性回归的核心要点

线性回归就像是一个**数据关系的翻译官**：

### 🎯 核心思想

1. **寻找最佳直线**：用一条直线最好地描述数据关系
2. **最小化误差**：让预测值尽可能接近真实值
3. **量化影响**：每个特征对目标的影响程度

### 📊 关键概念

- **斜率/系数**：特征的重要性和影响方向
- **截距**：基础水平/起始值
- **R²**：模型解释数据变异的比例
- **残差**：预测与实际的差异

### 💡 实用技巧

1. **数据预处理**：标准化、处理缺失值
2. **特征工程**：创造有意义的特征
3. **模型诊断**：检查假设、分析残差
4. **正则化**：防止过拟合
5. **交叉验证**：评估模型泛化能力

### 🎪 应用场景

- **预测问题**：房价、销量、成绩等
- **特征重要性分析**：了解哪些因素最重要
- **趋势分析**：理解变量间的关系
- **基准模型**：作为复杂模型的对比基线

### 🔧 记忆口诀

**"找直线，算误差，调参数，验效果"**

线性回归虽然简单，却是机器学习的基石。掌握了线性回归，你就拿到了进入机器学习世界的第一把钥匙！

---

**作者**: meimeitou  
