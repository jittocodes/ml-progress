# SGD Classifier (Stochastic Gradient Descent)

---

## Like a 5-year old

Imagine you're trying to find the lowest point in a hilly park while blindfolded. You take a small step downhill each time. **Gradient descent** does this — it takes one step at a time toward the lowest error.

**Stochastic** means you look at just **one example at a time** (instead of all of them) before taking each step. It's faster but a bit wobbly. Imagine getting directions from one person at a time instead of asking everyone at once.

SGD is **not a new model** — it's a **way to train** existing models (like logistic regression) much faster on large datasets.

---

## Technical Understanding

SGD Classifier is **not a standalone algorithm** — it's an optimization method applied to linear models. In sklearn, `SGDClassifier` trains:

- **Logistic Regression** (loss='log_loss')
- **SVM** (loss='hinge')
- **Linear SVM** (loss='modified_huber')
- **Perceptron** (loss='perceptron')

**Why SGD instead of normal gradient descent?**

| Method | Data used per update | Speed | Stability |
|---|---|---|---|
| Batch GD | All n samples | Slow | Very stable |
| Mini-batch GD | k samples (k < n) | Medium | Stable |
| Stochastic GD | 1 sample | Fast | Noisy |

For large datasets (millions of rows), computing gradients over all data per update is too expensive. SGD approximates by using one sample (or a small batch) at a time.

**Key properties:**
- Scales to very large datasets (online learning)
- Requires feature scaling (sensitive to feature magnitude)
- Noisy updates can escape local minima (sometimes helpful)
- Convergence is noisier but reaches good solutions

---

## The Math

### Standard Gradient Descent (Batch)

Update using gradient over ALL samples:

```
w := w - α * (1/n) * Σ ∂L(xᵢ, yᵢ, w)/∂w
```

Must loop over all n samples before one update. Expensive.

### Stochastic Gradient Descent

Update using gradient from ONE sample at a time:

```
For each sample i:
    w := w - α * ∂L(xᵢ, yᵢ, w)/∂w
```

n updates per epoch instead of 1. Much faster convergence.

### Loss Functions Available in SGDClassifier

| Loss | Model it trains | Use when |
|---|---|---|
| `log_loss` | Logistic Regression | Probabilistic classification |
| `hinge` | Linear SVM | Hard-margin classification |
| `modified_huber` | Smooth SVM | Tolerant to outliers, gives probabilities |
| `perceptron` | Perceptron | Simple linear classifier |

### Learning Rate Schedule

The learning rate `α` can decay over time:

```
α_t = α₀ / (1 + decay_rate * t)
```

This stabilizes convergence — large steps early, small steps later.

### Convergence Visualization

```
Loss
 |
 |  *
 | * *
 |*   *   *                    Batch GD (smooth)
 |     *-*-*----___
 |
 |  *                          SGD (noisy but fast)
 |*  * * *
 |      * **  *  *  *
 |               *-*-*__
 |
 |______________________________ Iterations
```

SGD is noisier but reaches near-optimal solution much faster in wall-clock time for large n.

---

## Technical Example — Fraud Detection on Large Dataset

**Why SGD here:** Imagine 10 million transactions. Logistic regression via Normal Equation needs to invert a matrix — expensive. SGD trains incrementally and handles this efficiently.

**Also useful for:** Online learning — the model can be updated as new fraud patterns emerge without retraining from scratch (`partial_fit`).

---

## Pseudocode

```
FUNCTION sgd_train(X, y, loss_fn, learning_rate, n_epochs):
    w = zeros(n_features)
    b = 0

    FOR epoch in range(n_epochs):
        # Shuffle data each epoch (important for SGD)
        indices = shuffle(range(n_samples))

        FOR i in indices:
            xᵢ = X[i]
            yᵢ = y[i]

            # Compute prediction
            ŷᵢ = predict(xᵢ, w, b)

            # Compute gradient of loss for this single sample
            dw, db = gradient(loss_fn, xᵢ, yᵢ, ŷᵢ)

            # Update weights
            w = w - learning_rate * dw
            b = b - learning_rate * db

    RETURN w, b


FUNCTION partial_fit(X_new, y_new, w, b):
    # Online update with new data — no retraining from scratch
    FOR i in range(len(X_new)):
        ŷ = predict(X_new[i], w, b)
        dw, db = gradient(loss_fn, X_new[i], y_new[i], ŷ)
        w = w - learning_rate * dw
        b = b - learning_rate * db
    RETURN w, b
```

---

## Python Implementation — From Scratch

```python
import numpy as np

class SGDClassifierScratch:
    """SGD-trained Logistic Regression"""

    def __init__(self, learning_rate=0.01, n_epochs=50, batch_size=1):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def log_loss(self, y, y_pred):
        return -np.mean(
            y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15)
        )

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.n_epochs):
            # Shuffle
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            # Mini-batch updates
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                Xb = X_shuffled[start:end]
                yb = y_shuffled[start:end]

                z = Xb @ self.weights + self.bias
                y_pred = self.sigmoid(z)

                # Gradients
                dw = Xb.T @ (y_pred - yb) / len(yb)
                db = np.mean(y_pred - yb)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

            # Track loss per epoch
            z_all = X @ self.weights + self.bias
            y_all = self.sigmoid(z_all)
            self.losses.append(self.log_loss(y, y_all))

        return self

    def partial_fit(self, X, y):
        """Online update — no full retrain"""
        if self.weights is None:
            self.weights = np.zeros(X.shape[1])
            self.bias = 0.0

        for i in range(len(X)):
            z = X[i] @ self.weights + self.bias
            y_pred = self.sigmoid(z)
            dw = X[i] * (y_pred - y[i])
            db = y_pred - y[i]
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X):
        return self.sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# --- Simulate large fraud dataset ---
np.random.seed(42)
n = 10000
amount = np.random.exponential(200, n)
hour = np.random.randint(0, 24, n)
distance = np.random.exponential(50, n)
logit = -3 + 0.003*amount + 0.1*(hour>22) + 0.01*distance
prob = 1 / (1 + np.exp(-logit))
y = (np.random.rand(n) < prob).astype(int)
X = np.column_stack([amount, hour, distance])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Mini-batch SGD
model = SGDClassifierScratch(learning_rate=0.1, n_epochs=20, batch_size=32)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {np.mean(y_pred == y_test):.4f}")
print(f"Final loss: {model.losses[-1]:.4f}")
```

---

## Sklearn Implementation

```python
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

# Data
np.random.seed(42)
n = 10000
amount = np.random.exponential(200, n)
hour = np.random.randint(0, 24, n)
distance = np.random.exponential(50, n)
logit = -3 + 0.003*amount + 0.1*(hour>22) + 0.01*distance
prob = 1 / (1 + np.exp(-logit))
y = (np.random.rand(n) < prob).astype(int)
X = np.column_stack([amount, hour, distance])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# IMPORTANT: Always scale before SGD
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- SGD as Logistic Regression ---
sgd_lr = SGDClassifier(
    loss='log_loss',        # Logistic regression
    alpha=0.0001,           # L2 regularization
    learning_rate='optimal', # Auto-tune schedule
    max_iter=100,
    random_state=42,
    class_weight='balanced'  # Handle imbalance
)
sgd_lr.fit(X_train_s, y_train)
print("=== SGD as Logistic Regression ===")
print(classification_report(y_test, sgd_lr.predict(X_test_s),
      target_names=["Legit", "Fraud"]))

# --- SGD as Linear SVM ---
sgd_svm = SGDClassifier(
    loss='hinge',
    alpha=0.0001,
    max_iter=100,
    random_state=42
)
sgd_svm.fit(X_train_s, y_train)
print("=== SGD as Linear SVM ===")
print(classification_report(y_test, sgd_svm.predict(X_test_s),
      target_names=["Legit", "Fraud"]))

# --- Online learning with partial_fit ---
print("\n=== Online Learning Demo ===")
online_model = SGDClassifier(loss='log_loss', random_state=42)

# Simulate streaming batches
batch_size = 500
for start in range(0, len(X_train_s), batch_size):
    Xb = X_train_s[start:start+batch_size]
    yb = y_train[start:start+batch_size]
    online_model.partial_fit(Xb, yb, classes=[0, 1])

print(classification_report(y_test, online_model.predict(X_test_s),
      target_names=["Legit", "Fraud"]))
```

---

## Key Interview Questions

**Q: Is SGDClassifier a different model from Logistic Regression?**  
No. `SGDClassifier(loss='log_loss')` is logistic regression trained with SGD. `LogisticRegression()` uses more sophisticated optimizers (LBFGS). Results are similar; SGD is preferred for very large data.

**Q: Why must you scale features before SGD?**  
SGD updates weights proportional to feature values. A feature in thousands (e.g. salary) will dominate updates over a feature in decimals (e.g. age in years). Scaling puts everything on equal footing.

**Q: What is `partial_fit` useful for?**  
Online/incremental learning — updating a model on new data without retraining from scratch. Critical for production systems where new fraud patterns emerge over time.

**Q: SGD vs Mini-batch GD?**  
Pure SGD = batch_size=1. Mini-batch = batch_size=32/64/128. Mini-batch is a practical middle ground — GPU-efficient, less noisy than pure SGD.

---

## When to Use

✅ Very large datasets (millions of rows)  
✅ Online/streaming learning (`partial_fit`)  
✅ When memory is limited (processes one batch at a time)  
✅ Fast approximate solution before full training  

❌ Small datasets (overkill, use regular LogisticRegression)  
❌ Without feature scaling (will not converge properly)  
❌ When you need exact probabilities (noisy convergence)  
