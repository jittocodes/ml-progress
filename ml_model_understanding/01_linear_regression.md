# Linear Regression

---

## Like a 5-year old

Imagine you're tracking how tall your plant grows each day. You water it the same amount and notice: the more days pass, the taller it gets. You draw a straight line through all the dots on your chart. That line helps you **guess** how tall the plant will be on day 10, even if you haven't seen day 10 yet.

Linear regression draws the **best possible straight line** through your data to make guesses.

---

## Technical Understanding

Linear regression models the relationship between a **continuous target variable** `y` and one or more **input features** `X` by fitting a straight line (or hyperplane in multiple dimensions).

It assumes:
- The relationship between features and target is **linear**
- Errors (residuals) are **normally distributed** with constant variance
- Features are **independent** of each other (no multicollinearity)

The model learns the **weights** (coefficients) that minimize the difference between its predictions and the actual values.

**Types:**
- **Simple Linear Regression** — one feature: `y = mx + b`
- **Multiple Linear Regression** — many features: `y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`

---

## The Math

### Hypothesis function

```
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

In matrix form:
```
ŷ = Xw
```

### Cost Function — Mean Squared Error (MSE)

```
MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
```

We want to **minimize** this. The goal is to find weights `w` such that the squared distance between predictions and true values is as small as possible.

### Normal Equation (closed-form solution)

```
w = (XᵀX)⁻¹ Xᵀy
```

Directly computes optimal weights. Works well for small datasets. Expensive for large ones (matrix inversion is O(n³)).

### Gradient Descent (iterative solution)

```
w := w - α * ∂MSE/∂w
```

Where `α` is the **learning rate**. Repeatedly adjusts weights in the direction that reduces the error.

```
∂MSE/∂w = -(2/n) * Xᵀ(y - Xw)
```

### Visualization

```
      y
      |          * (actual)
      |        /
      |      / ← predicted line (ŷ = wx + b)
      |    *↑
      |   /  residual (error) = y - ŷ
      |  *
      | /
      |/_____________________ x

Residual = vertical distance from point to line
MSE = average of all (residual²)
We minimize MSE to find the best line
```

### Regularization

To prevent overfitting:

| Type | Penalty added to MSE | Effect |
|---|---|---|
| Ridge (L2) | `+ λ * Σwᵢ²` | Shrinks weights toward zero |
| Lasso (L1) | `+ λ * Σ\|wᵢ\|` | Can zero out weights (feature selection) |
| Elastic Net | Mix of L1 + L2 | Best of both |

---

## Technical Example — Fraud Transaction Amount Prediction

**Problem:** Predict the `transaction_amount` of a customer based on their `account_age_days` and `num_previous_transactions`.

This is a regression problem — the target is continuous (a money amount).

**Feature thinking:**
- Older accounts tend to have higher average transaction amounts
- More transactions = more established user = potentially higher amounts

---

## Pseudocode

```
FUNCTION linear_regression_train(X, y):
    n_samples, n_features = shape(X)
    
    # Add bias column (column of 1s)
    X_bias = add_column_of_ones(X)
    
    # Normal equation
    w = inverse(X_bias.T @ X_bias) @ X_bias.T @ y
    
    RETURN w

FUNCTION predict(X, w):
    X_bias = add_column_of_ones(X)
    RETURN X_bias @ w

FUNCTION mse(y_true, y_pred):
    RETURN mean((y_true - y_pred) ** 2)
```

---

## Python Implementation — From Scratch

```python
import numpy as np

class LinearRegressionScratch:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Add bias column
        X_b = np.c_[np.ones((n_samples, 1)), X]

        # Normal equation: w = (X^T X)^-1 X^T y
        self.weights_full = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.bias = self.weights_full[0]
        self.weights = self.weights_full[1:]

    def predict(self, X):
        return X @ self.weights + self.bias

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)


# --- Simulate fraud data ---
np.random.seed(42)
n = 500

account_age = np.random.randint(30, 3000, n)
num_txns = np.random.randint(1, 200, n)
noise = np.random.normal(0, 50, n)

# True relationship: amount ~ 0.05 * age + 1.2 * num_txns + noise
transaction_amount = 0.05 * account_age + 1.2 * num_txns + 20 + noise

X = np.column_stack([account_age, num_txns])
y = transaction_amount

# Train/test split
split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Fit and evaluate
model = LinearRegressionScratch()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Weights: account_age={model.weights[0]:.4f}, num_txns={model.weights[1]:.4f}")
print(f"Bias: {model.bias:.4f}")
print(f"MSE: {model.mse(y_test, y_pred):.2f}")
print(f"R² Score: {model.r2_score(y_test, y_pred):.4f}")
```

---

## Sklearn Implementation

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Data (same as above)
np.random.seed(42)
n = 500
account_age = np.random.randint(30, 3000, n)
num_txns = np.random.randint(1, 200, n)
noise = np.random.normal(0, 50, n)
transaction_amount = 0.05 * account_age + 1.2 * num_txns + 20 + noise

X = np.column_stack([account_age, num_txns])
y = transaction_amount

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (good practice)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Plain Linear Regression ---
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)

print("=== Linear Regression ===")
print(f"Coefficients: {lr.coef_}")
print(f"Intercept: {lr.intercept_:.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# --- Ridge (L2 regularization) ---
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)
print(f"\nRidge R²: {r2_score(y_test, y_pred_ridge):.4f}")

# --- Lasso (L1 regularization) ---
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
print(f"Lasso R²: {r2_score(y_test, y_pred_lasso):.4f}")
print(f"Lasso non-zero coefficients: {sum(lasso.coef_ != 0)}")
```

---

## Key Interview Questions

**Q: When does linear regression fail?**
When the relationship is non-linear, features are correlated, or outliers dominate.

**Q: What does R² tell you?**
The proportion of variance in y explained by the model. R²=1 is perfect, R²=0 means the model is no better than predicting the mean.

**Q: Why scale features?**
Gradient descent converges faster. Also helps interpret coefficients fairly.

**Q: Ridge vs Lasso?**
Lasso can zero out coefficients (feature selection). Ridge shrinks all weights but keeps them non-zero. Use Lasso when you suspect many features are irrelevant.

---

## When to Use

✅ Target is continuous  
✅ Relationship is roughly linear  
✅ You need interpretability (coefficients explain feature impact)  
✅ Fast baseline before trying complex models  

❌ Target is a category (use logistic regression)  
❌ Highly non-linear relationships  
❌ Many outliers in the data  
