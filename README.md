# Week 1 — Linear Regression

> **Theme:** Foundations of Supervised Learning — Regression  
> **Dataset:** California Housing (sklearn) | **Tools:** Python, statsmodels, scikit-learn, Seaborn, Plotly

---

## Table of Contents

1. [What is Linear Regression?](#1-what-is-linear-regression)
2. [Simple Linear Regression (SLR)](#2-simple-linear-regression-slr)
3. [Multiple Linear Regression (MLR)](#3-multiple-linear-regression-mlr)
4. [Ordinary Least Squares (OLS)](#4-ordinary-least-squares-ols)
5. [MSE Loss Function](#5-mse-loss-function)
6. [Gradient Descent](#6-gradient-descent)
7. [5 Assumptions of Linear Regression](#7-5-assumptions-of-linear-regression)
8. [Model Evaluation Metrics](#8-model-evaluation-metrics)
9. [Python Implementation](#9-python-implementation)

---

## 1. What is Linear Regression?

Linear Regression is a **supervised learning** algorithm used to model the relationship between one or more **input features** $X$ and a **continuous target variable** $y$.

It assumes the relationship is **linear** — meaning the target can be expressed as a weighted sum of the input features plus a bias term.

---

## 2. Simple Linear Regression (SLR)

SLR models the relationship between **one independent variable** $x$ and one dependent variable $y$.

### Equation

$$\hat{y} = mx + c$$

Where:
- $\hat{y}$ = predicted output
- $m$ = slope (weight / coefficient)
- $x$ = input feature
- $c$ = intercept (bias)

### Slope and Intercept Formulas

$$m = \frac{n \sum x_i y_i - \sum x_i \sum y_i}{n \sum x_i^2 - \left(\sum x_i\right)^2}$$

$$c = \bar{y} - m\bar{x}$$

Where $\bar{x}$ and $\bar{y}$ are the means of $x$ and $y$ respectively.

### Interpretation

- **Slope $m$**: For every 1-unit increase in $x$, $\hat{y}$ changes by $m$ units.
- **Intercept $c$**: The predicted value of $y$ when $x = 0$.

---

## 3. Multiple Linear Regression (MLR)

MLR extends SLR to **multiple independent variables** $x_1, x_2, \ldots, x_p$.

### Equation (Scalar Form)

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_p x_p$$

### Matrix Form

$$\hat{\mathbf{y}} = \mathbf{X} \mathbf{w}$$

Where:
- $\mathbf{X} \in \mathbb{R}^{n \times (p+1)}$ — design matrix (with a column of 1s prepended for the bias)
- $\mathbf{w} \in \mathbb{R}^{p+1}$ — weight vector $[w_0, w_1, \ldots, w_p]^T$
- $\hat{\mathbf{y}} \in \mathbb{R}^{n}$ — vector of predictions

$$\mathbf{X} = \begin{bmatrix} 1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_p^{(1)} \\ 1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_p^{(2)} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_1^{(n)} & x_2^{(n)} & \cdots & x_p^{(n)} \end{bmatrix}$$

---

## 4. Ordinary Least Squares (OLS)

OLS finds the weights $\mathbf{w}$ that **minimize the sum of squared residuals** between actual values $y_i$ and predicted values $\hat{y}_i$.

### Objective

$$\min_{\mathbf{w}} \sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2 = \min_{\mathbf{w}} \| \mathbf{y} - \mathbf{X}\mathbf{w} \|^2$$

### Closed-Form Solution (Normal Equation)

Setting the derivative of the cost with respect to $\mathbf{w}$ to zero:

$$\frac{\partial}{\partial \mathbf{w}} \| \mathbf{y} - \mathbf{X}\mathbf{w} \|^2 = 0$$

$$\Rightarrow \quad \hat{\mathbf{w}} = \left(\mathbf{X}^T \mathbf{X}\right)^{-1} \mathbf{X}^T \mathbf{y}$$

### Conditions for OLS

- $\mathbf{X}^T \mathbf{X}$ must be **invertible** (no perfect multicollinearity)
- Computationally expensive for very large $p$ (use Gradient Descent instead)

---

## 5. MSE Loss Function

The **Mean Squared Error (MSE)** is the primary loss function for regression.

$$\mathcal{L}(\mathbf{w}) = \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2$$

Where:
- $y_i$ = actual value
- $\hat{y}_i = \mathbf{x}_i^T \mathbf{w}$ = predicted value
- $n$ = number of samples

### Why Squared Error?

- Penalizes **large errors** more heavily than small ones
- Differentiable everywhere — enables gradient-based optimization
- The optimal solution under the **Gaussian noise** assumption (via MLE)

### Related Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Squared units |
| RMSE | $\sqrt{\text{MSE}}$ | Same units as $y$ |
| MAE | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Robust to outliers |

---

## 6. Gradient Descent

Gradient Descent is an **iterative optimization algorithm** that minimizes the MSE loss by updating weights in the direction of the negative gradient.

### Update Rule

$$w_j \leftarrow w_j - \alpha \frac{\partial \mathcal{L}}{\partial w_j}$$

Where $\alpha$ is the **learning rate** (step size).

### Gradient of MSE

$$\frac{\partial \mathcal{L}}{\partial w_j} = -\frac{2}{n} \sum_{i=1}^{n} \left(y_i - \hat{y}_i\right) x_j^{(i)}$$

### Matrix Form Update

$$\mathbf{w} \leftarrow \mathbf{w} - \frac{\alpha}{n} \mathbf{X}^T \left(\mathbf{X}\mathbf{w} - \mathbf{y}\right)$$

### Bias Update

$$w_0 \leftarrow w_0 - \frac{\alpha}{n} \sum_{i=1}^{n} \left(\hat{y}_i - y_i\right)$$

### Learning Rate $\alpha$

| Learning Rate | Effect |
|---------------|--------|
| Too large | Diverges — loss increases |
| Too small | Converges very slowly |
| Just right | Converges to global minimum |

### Variants

- **Batch GD** — uses all $n$ samples per update (slow but stable)
- **Stochastic GD (SGD)** — uses 1 sample per update (noisy but fast)
- **Mini-Batch GD** — uses a batch of $k$ samples (best of both)

---

## 7. Five Assumptions of Linear Regression

Violating these assumptions leads to **biased, inefficient, or invalid** estimates.

---

### Assumption 1 — Linearity

**Statement:** The relationship between $X$ and $y$ is linear.

$$y_i = \mathbf{x}_i^T \mathbf{w} + \varepsilon_i$$

**How to Check:**
- Scatter plot of each $x_j$ vs $y$
- Residual plot: residuals vs fitted values should show **no pattern**

---

### Assumption 2 — No Multicollinearity

**Statement:** Independent variables $x_1, x_2, \ldots, x_p$ must not be **perfectly correlated** with each other.

**Why It Matters:** Perfect multicollinearity makes $\mathbf{X}^T\mathbf{X}$ **singular** (non-invertible), and even near-multicollinearity inflates coefficient variance.

**Quantified by Variance Inflation Factor (VIF):**

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

Where $R_j^2$ is the $R^2$ from regressing $x_j$ on all other features.

| VIF Value | Interpretation |
|-----------|---------------|
| $\text{VIF} = 1$ | No multicollinearity |
| $1 < \text{VIF} < 5$ | Moderate (acceptable) |
| $\text{VIF} \geq 10$ | High — investigate / drop feature |

**How to Check:**
- Correlation heatmap
- VIF scores for each feature

---

### Assumption 3 — Normality of Residuals

**Statement:** Residuals $\varepsilon_i = y_i - \hat{y}_i$ must follow a **normal distribution** with mean zero.

$$\varepsilon \sim \mathcal{N}(0, \sigma^2)$$

**Why It Matters:** Required for valid hypothesis tests on coefficients (t-tests, F-test).

**How to Check:**
- **KDE plot** of residuals — should be bell-shaped, centered at 0
- **Q-Q (Quantile-Quantile) plot** — points should lie along the diagonal line

---

### Assumption 4 — Homoscedasticity

**Statement:** The **variance of residuals** is **constant** across all levels of $\hat{y}$ (fitted values).

$$\text{Var}(\varepsilon_i) = \sigma^2 \quad \forall \, i$$

**Opposite = Heteroscedasticity:** Variance increases or decreases with fitted values — makes OLS estimates inefficient.

**How to Check:**
- Scatter plot of residuals vs fitted values
- Should show a **random horizontal band** around zero (no funnel shape)

---

### Assumption 5 — No Autocorrelation

**Statement:** Residuals are **independent** of each other — knowing $\varepsilon_i$ tells you nothing about $\varepsilon_{i+1}$.

$$\text{Cov}(\varepsilon_i, \varepsilon_j) = 0 \quad \text{for } i \neq j$$

**Most relevant for:** Time series data where observations are ordered.

**How to Check:**
- **ACF (Autocorrelation Function) plot** — lags beyond lag 0 should fall within the confidence bands
- **Durbin-Watson statistic**: values close to 2 indicate no autocorrelation

| Durbin-Watson | Interpretation |
|---------------|---------------|
| $\approx 2$ | No autocorrelation |
| $< 1.5$ | Positive autocorrelation |
| $> 2.5$ | Negative autocorrelation |

---

## 8. Model Evaluation Metrics

### R² (Coefficient of Determination)

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- Range: $(-\infty, 1]$
- $R^2 = 1$ → perfect fit; $R^2 = 0$ → model no better than predicting the mean

### Adjusted R²

$$\bar{R}^2 = 1 - \left(1 - R^2\right) \frac{n - 1}{n - p - 1}$$

- Penalizes adding irrelevant features (unlike $R^2$ which always increases)
- Preferred over $R^2$ for MLR model comparison

### RMSE

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

- Same unit as $y$ — directly interpretable
- Lower is better

---

## 9. Python Implementation

### Using `statsmodels` (OLS)

```python
import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('data/housing.csv')
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']

# Add intercept (bias) column
X = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X).fit()

# View detailed statistical summary
print(model.summary())
```

### Using `scikit-learn` (Linear Regression)

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²   : {r2_score(y_test, y_pred):.4f}")
```

### VIF Calculation

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data.sort_values('VIF', ascending=False))
```

### Gradient Descent (From Scratch)

```python
import numpy as np

def gradient_descent(X, y, lr=0.01, epochs=1000):
    n, p = X.shape
    w = np.zeros(p)        # initialize weights
    losses = []

    for epoch in range(epochs):
        y_pred = X @ w
        residuals = y_pred - y
        gradient = (2 / n) * X.T @ residuals
        w -= lr * gradient
        loss = np.mean(residuals ** 2)
        losses.append(loss)

    return w, losses
```

---

## 📁 Project Structure

```
Week-01-Linear-Regression/
├── README.md               ← This file (theory + formulas)
├── Project_Notebook.ipynb  ← End-to-end Python project
├── data/                   ← Dataset files
└── images/                 ← Saved plots
```

---

## 📚 References

- Hastie, Tibshirani & Friedman — *The Elements of Statistical Learning*
- James, Witten et al. — *An Introduction to Statistical Learning*
- [statsmodels OLS documentation](https://www.statsmodels.org/stable/regression.html)
- [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

---

*Next: [Week 2 — Advanced Regression & Regularization →](../Week-02-Advanced-Regression/README.md)*
