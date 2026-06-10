# Logistic Regression

---

## Like a 5-year old

Imagine you're sorting your toys into two boxes: "like" and "don't like." You look at each toy and say yes or no. Logistic regression is like a machine that looks at clues about a toy (its color, size, if it makes noise) and tells you: **"I'm 80% sure you'll like this one."**

It doesn't give you a number like height. It gives you a **probability** — a number between 0 and 1 — and then picks a side.

---

## Technical Understanding

Logistic regression is a **binary classification** algorithm (extendable to multiclass). Despite the name, it's a classifier, not a regressor.

It models the **probability** that an input belongs to class 1:

```
P(y=1 | X) = σ(Xw)
```

Where `σ` is the **sigmoid function** that squashes any value into [0, 1].

**Key properties:**
- Output is a probability between 0 and 1
- Decision boundary is linear (a straight line or hyperplane)
- Assumes features and log-odds have a linear relationship
- Highly interpretable — coefficients map directly to feature impact on odds

**Why not linear regression for classification?**  
Linear regression can output values outside [0,1] and is sensitive to outliers. Logistic regression bounds output to a valid probability range.

---

## The Math

### Sigmoid Function

Converts raw score (logit) to probability:

```
σ(z) = 1 / (1 + e^(-z))
```

```
z = -5  →  σ = 0.007  (almost certainly class 0)
z = 0   →  σ = 0.500  (uncertain)
z = 5   →  σ = 0.993  (almost certainly class 1)
```

### Visualization of sigmoid

```
  P(y=1)
  1.0 |                          __________
      |                      __/
  0.5 |-------------------*-/------------------
      |               __/
  0.0 |______________/
      |__________________________________ z (raw score)
     -5    -3    -1    0    1    3    5

Decision threshold: if P(y=1) >= 0.5 → predict class 1
```

### Cost Function — Log Loss (Binary Cross-Entropy)

MSE doesn't work here (non-convex). Log loss is used instead:

```
L = -(1/n) * Σ [ yᵢ * log(ŷᵢ) + (1 - yᵢ) * log(1 - ŷᵢ) ]
```

- If y=1 and ŷ≈1: loss ≈ 0 (correct, no penalty)
- If y=1 and ŷ≈0: loss → ∞ (confident and wrong, huge penalty)

### Gradient for weight update

```
∂L/∂w = (1/n) * Xᵀ(ŷ - y)
```

Update rule (gradient descent):
```
w := w - α * ∂L/∂w
```

### Decision Boundary

```
Predict class 1 if: σ(Xw) >= 0.5
Which means:        Xw >= 0
```

For 2 features x₁, x₂:
```
Decision boundary is the line: w₁x₁ + w₂x₂ + b = 0

   x₂
    |    * * * (class 1)
    |   * * *
    |  /← boundary
    | / * (class 0)
    |/*  *
    |________ x₁
```

### Odds and Log-Odds (Logit)

The coefficient `w₁` means: for every 1-unit increase in `x₁`, the **log-odds** of class 1 increase by `w₁`.

```
Odds = P(y=1) / P(y=0)
Log-odds (logit) = log(Odds) = Xw
```

---

## Technical Example — Fraud Detection

**Problem:** Predict whether a transaction is fraudulent (1) or legitimate (0).

**Features:**
- `amount` — transaction amount
- `hour_of_day` — time of transaction (fraud spikes at 2–4am)
- `distance_from_home` — how far from the cardholder's home location

**Feature thinking:**
- High amount + unusual hour + far from home → higher fraud probability

---

## Pseudocode

```
FUNCTION sigmoid(z):
    RETURN 1 / (1 + exp(-z))

FUNCTION logistic_regression_train(X, y, learning_rate, epochs):
    n_samples, n_features = shape(X)
    w = zeros(n_features)
    b = 0

    FOR each epoch:
        z = X @ w + b
        y_pred = sigmoid(z)

        # Gradients
        dw = (1/n_samples) * X.T @ (y_pred - y)
        db = (1/n_samples) * sum(y_pred - y)

        # Update weights
        w = w - learning_rate * dw
        b = b - learning_rate * db

    RETURN w, b

FUNCTION predict_proba(X, w, b):
    RETURN sigmoid(X @ w + b)

FUNCTION predict_class(X, w, b, threshold=0.5):
    proba = predict_proba(X, w, b)
    RETURN (proba >= threshold).astype(int)
```

---

## Python Implementation — From Scratch

```python
import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.n_epochs):
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)

            # Gradients
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            # Log loss
            loss = -np.mean(
                y * np.log(y_pred + 1e-15) +
                (1 - y) * np.log(1 - y_pred + 1e-15)
            )
            self.losses.append(loss)

    def predict_proba(self, X):
        return self.sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# --- Simulate fraud data ---
np.random.seed(42)
n = 1000

amount = np.random.exponential(200, n)
hour = np.random.randint(0, 24, n)
distance = np.random.exponential(50, n)

# Fraud probability: high amount + late hour + far from home
logit = -3 + 0.003 * amount + 0.05 * (hour > 22).astype(float) + 0.01 * distance
prob = 1 / (1 + np.exp(-logit))
y = (np.random.rand(n) < prob).astype(int)

print(f"Fraud rate: {y.mean():.2%}")

X = np.column_stack([amount, hour, distance])

# Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

split = int(0.8 * n)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

model = LogisticRegressionScratch(learning_rate=0.1, n_epochs=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.4f}")
print(f"Final log loss: {model.losses[-1]:.4f}")
```

---

## Sklearn Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve
)
import numpy as np

# Data (same simulation as above)
np.random.seed(42)
n = 1000
amount = np.random.exponential(200, n)
hour = np.random.randint(0, 24, n)
distance = np.random.exponential(50, n)
logit = -3 + 0.003 * amount + 0.05 * (hour > 22).astype(float) + 0.01 * distance
prob = 1 / (1 + np.exp(-logit))
y = (np.random.rand(n) < prob).astype(int)
X = np.column_stack([amount, hour, distance])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Fit model ---
clf = LogisticRegression(
    C=1.0,           # Regularization strength (inverse of λ — lower C = more regularization)
    max_iter=1000,
    random_state=42
)
clf.fit(X_train_s, y_train)

y_pred = clf.predict(X_test_s)
y_proba = clf.predict_proba(X_test_s)[:, 1]

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Coefficients: {clf.coef_[0]}")
print(f"Intercept: {clf.intercept_[0]:.4f}")

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# --- Threshold tuning ---
# Default 0.5 is often wrong for fraud (imbalanced). Try lower threshold.
y_pred_low = (y_proba >= 0.3).astype(int)
print("\n=== With threshold=0.3 ===")
print(classification_report(y_test, y_pred_low, target_names=["Legit", "Fraud"]))
```

---

## Key Interview Questions

**Q: Why is it called "regression" if it classifies?**  
It models the log-odds as a linear function — that's regression. The classification comes from thresholding the output probability.

**Q: What is the C parameter in sklearn?**  
`C = 1/λ` — the inverse of regularization strength. Small C = strong regularization = simpler model. Large C = weak regularization = fits training data more closely.

**Q: Why not use accuracy for fraud detection?**  
If 99% of transactions are legit, a model that always predicts "legit" gets 99% accuracy. Use precision, recall, F1, and AUC-PR instead.

**Q: How do you handle the threshold?**  
Default 0.5 maximizes accuracy. For fraud detection, lower the threshold to catch more fraud (improve recall) at the cost of more false positives. Tune based on business cost of FP vs FN.

---

## When to Use

✅ Binary or multiclass classification  
✅ Need probability outputs (not just class labels)  
✅ Need interpretability (regulatory compliance in banking)  
✅ Fast baseline before trying complex models  
✅ Linearly separable data  

❌ Complex non-linear decision boundaries  
❌ Very high-dimensional sparse data (consider linear SVM or tree models)
