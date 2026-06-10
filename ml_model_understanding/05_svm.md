# Support Vector Machine (SVM)

---

## Like a 5-year old

Imagine you have red and blue marbles mixed on a table. You want to draw a line to separate them. But there are many possible lines — which one is best?

SVM finds the line that has the **biggest gap** between the nearest red and blue marbles. It picks the line that stays as far away from both sides as possible. Those nearest marbles that "define" the gap are called **support vectors** — they're the marbles doing all the work.

---

## Technical Understanding

SVM finds the **optimal hyperplane** that separates classes with the **maximum margin**.

**Key concepts:**
- **Hyperplane** — the decision boundary (a line in 2D, plane in 3D, hyperplane in nD)
- **Margin** — the distance between the hyperplane and the nearest data points of each class
- **Support vectors** — the data points closest to the hyperplane that define the margin
- **Maximum margin** — SVM maximizes this gap, which gives better generalization

**Two variants:**

| Type | When data is | Approach |
|---|---|---|
| Hard Margin SVM | Perfectly linearly separable | No points inside the margin |
| Soft Margin SVM | Not perfectly separable | Allows some violations with penalty `C` |

**Kernel trick:**  
When data is not linearly separable in the original space, SVM maps it to a higher-dimensional space where it becomes separable — without explicitly computing that transformation.

---

## The Math

### Linear SVM Decision Function

```
f(x) = wᵀx + b
Predict class +1 if f(x) >= 0
Predict class -1 if f(x) < 0
```

### Margin

For correctly classified points:
```
Class +1: wᵀxᵢ + b >= +1
Class -1: wᵀxᵢ + b <= -1
```

Margin width = `2 / ||w||`

### Optimization Objective (Hard Margin)

```
Minimize:   (1/2) * ||w||²
Subject to: yᵢ(wᵀxᵢ + b) >= 1  for all i
```

Maximizing margin = minimizing ||w||.

### Soft Margin (Slack Variables ξᵢ)

```
Minimize:   (1/2) * ||w||² + C * Σξᵢ
Subject to: yᵢ(wᵀxᵢ + b) >= 1 - ξᵢ,  ξᵢ >= 0
```

`C` controls the tradeoff:
- **Large C** — narrow margin, fewer violations, risk of overfit
- **Small C** — wide margin, more violations allowed, more generalizable

### Visualization

```
       x₂
        |     + + +
        |    +    +
        |   +  +   
        | ---- ← margin boundary (+1)
        | ====== ← decision boundary (hyperplane)
        | ---- ← margin boundary (-1)  
        |   - -
        |  -   -
        |   - -
        |___________________ x₁

Support vectors: the + and - points closest to the margin lines.
These are the ONLY points that matter for the decision boundary.
```

### Kernel Trick

When classes are not linearly separable:

```
Original space (1D):  ----  +++++  ----
Not separable with a line.

Map to 2D with φ(x) = (x, x²):

     x²
      |      *  *
      |    *      *         → now linearly separable
      |*            *
      |_____________________ x
      
SVM finds a line in 2D = parabola boundary in 1D
```

**Common kernels:**

| Kernel | Formula | Use when |
|---|---|---|
| Linear | `xᵀz` | Linearly separable data |
| RBF (Gaussian) | `exp(-γ\|\|x-z\|\|²)` | Non-linear, most common |
| Polynomial | `(xᵀz + c)^d` | Image features, NLP |

The kernel function computes similarity **without explicitly mapping** to higher dimensions. This is computationally efficient.

### RBF Kernel Parameter `γ`

Controls the influence radius of each training point:
- **Large γ** — each point has narrow influence → complex boundary → overfit
- **Small γ** — each point has wide influence → smoother boundary → underfit

---

## Technical Example — Fraud Detection

**Why SVM for fraud?**
- Works well on small-to-medium datasets with many features
- RBF kernel can capture complex fraud patterns
- Maximum margin means it generalizes well to unseen transactions

**Limitation:** Slow on millions of rows (O(n²) to O(n³)). Use `LinearSVC` or `SGDClassifier(loss='hinge')` for large data.

---

## Pseudocode

```
FUNCTION svm_train(X, y, C, kernel='rbf'):
    # SVM is solved as a quadratic programming problem
    # Dual form: find α such that:

    Maximize: Σαᵢ - (1/2) * ΣΣ αᵢαⱼyᵢyⱼ K(xᵢ, xⱼ)
    Subject to: 0 <= αᵢ <= C,  Σαᵢyᵢ = 0

    # Support vectors: points where αᵢ > 0
    support_vectors = X[α > 0]

    # Decision function
    f(x) = Σ αᵢyᵢ K(xᵢ, x) + b

    RETURN support_vectors, α, b


FUNCTION predict(x, support_vectors, α, y_sv, b, kernel):
    score = sum(αᵢ * yᵢ * kernel(svᵢ, x) for svᵢ in support_vectors)
    score += b
    RETURN +1 if score >= 0 else -1


FUNCTION rbf_kernel(x, z, gamma):
    RETURN exp(-gamma * ||x - z||²)
```

---

## Python Implementation — From Scratch (Linear SVM with Hinge Loss)

```python
import numpy as np

class LinearSVMScratch:
    """Linear SVM using gradient descent on hinge loss"""

    def __init__(self, C=1.0, learning_rate=0.001, n_epochs=1000):
        self.C = C
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Convert labels to -1, +1
        y_ = np.where(y == 0, -1, 1).astype(float)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                # Hinge loss condition
                if y_[i] * (X[i] @ self.w + self.b) >= 1:
                    # Correctly classified and outside margin
                    dw = self.w
                    db = 0
                else:
                    # Misclassified or inside margin (support vector)
                    dw = self.w - self.C * y_[i] * X[i]
                    db = -self.C * y_[i]

                self.w -= self.lr * dw
                self.b -= self.lr * db

        return self

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, 0)


# --- Test on fraud data ---
np.random.seed(42)
n = 500
amount = np.random.exponential(200, n)
hour = np.random.randint(0, 24, n)
logit = -3 + 0.004*amount + 0.15*(hour>22).astype(float)
prob = 1/(1 + np.exp(-logit))
y = (np.random.rand(n) < prob).astype(int)
X = np.column_stack([amount, hour])

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

svm = LinearSVMScratch(C=1.0, learning_rate=0.001, n_epochs=500)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(f"Linear SVM Accuracy: {np.mean(y_pred == y_test):.4f}")
```

---

## Sklearn Implementation

```python
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

# Data
np.random.seed(42)
n = 1000
amount = np.random.exponential(200, n)
hour = np.random.randint(0, 24, n)
distance = np.random.exponential(50, n)
logit = -3 + 0.003*amount + 0.1*(hour>22) + 0.01*distance
prob = 1/(1 + np.exp(-logit))
y = (np.random.rand(n) < prob).astype(int)
X = np.column_stack([amount, hour, distance])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# --- RBF SVM (handles non-linear boundaries) ---
svm_rbf = SVC(
    kernel='rbf',
    C=1.0,          # Soft margin penalty
    gamma='scale',  # Auto-set gamma = 1/(n_features * X.var())
    probability=True,
    random_state=42,
    class_weight='balanced'
)
svm_rbf.fit(X_train_s, y_train)
print("=== RBF SVM ===")
print(classification_report(y_test, svm_rbf.predict(X_test_s),
      target_names=["Legit","Fraud"]))
print(f"Support vectors per class: {svm_rbf.n_support_}")

# --- Linear SVM (fast, good for high-dim) ---
svm_lin = LinearSVC(
    C=1.0,
    max_iter=2000,
    random_state=42,
    class_weight='balanced'
)
svm_lin.fit(X_train_s, y_train)
print("\n=== Linear SVM ===")
print(classification_report(y_test, svm_lin.predict(X_test_s),
      target_names=["Legit","Fraud"]))

# --- Hyperparameter tuning ---
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1]
}
grid = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'),
    param_grid, cv=5, scoring='f1', n_jobs=-1
)
grid.fit(X_train_s, y_train)
print(f"\nBest params: {grid.best_params_}")
print(f"Best F1: {grid.best_score_:.4f}")
```

---

## Key Interview Questions

**Q: What are support vectors?**  
The training points closest to the decision boundary. Only these points matter for defining the boundary — removing other points doesn't change the model.

**Q: What does C control?**  
The tradeoff between margin width and classification errors. Large C = penalize errors heavily = narrow margin = potential overfit. Small C = allow more errors = wider margin = more robust.

**Q: How does the kernel trick work?**  
Instead of explicitly mapping data to higher dimensions (expensive), kernels compute the dot product in that high-dimensional space directly. `K(x, z) = φ(x)ᵀφ(z)` without ever computing `φ(x)`.

**Q: When would you use SVM vs Logistic Regression?**  
SVM for: small-medium datasets, high-dimensional features, non-linear boundaries with RBF kernel. Logistic Regression for: large datasets, need probability calibration, faster training.

**Q: Why is SVM slow on large datasets?**  
The QP solver's time complexity is O(n²) to O(n³) in number of samples. Use `LinearSVC` (O(n)) or `SGDClassifier(loss='hinge')` for large n.

---

## When to Use

✅ Small-to-medium datasets (< 100k rows)  
✅ High-dimensional feature spaces (text classification)  
✅ Non-linear boundaries (with RBF kernel)  
✅ Strong generalization needed  

❌ Large datasets (> 100k rows) — too slow  
❌ Need probability outputs directly (requires `probability=True` which is slow)  
❌ Many features with most irrelevant (use L1-regularized logistic instead)  
