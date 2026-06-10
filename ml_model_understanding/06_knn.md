# K-Nearest Neighbors (K-NN)

---

## Like a 5-year old

Imagine you move to a new neighborhood and don't know if you'll like living there. You ask the 5 nearest neighbors: "Do you like it here?" If 4 out of 5 say yes, you guess you'll like it too.

K-NN works exactly like that. To predict something about a new point, it finds the K closest points in the training data and takes a vote (for classification) or an average (for regression).

---

## Technical Understanding

K-NN is a **lazy learner** — it doesn't build an explicit model during training. Instead, it stores all training data and makes predictions at query time by looking at the nearest neighbors.

**Key properties:**
- **No training phase** — just memorize the data
- **Expensive at prediction time** — must compute distance to all training points
- **Non-parametric** — makes no assumptions about data distribution
- **Sensitive to feature scale** — always normalize before using K-NN
- **Curse of dimensionality** — degrades in very high dimensions (distances become meaningless)

**Two uses:**
- **Classification** — majority vote among K neighbors
- **Regression** — average value among K neighbors

---

## The Math

### Distance Metrics

The choice of distance metric defines what "nearest" means.

**Euclidean Distance (most common):**
```
d(x, z) = √[ Σ (xᵢ - zᵢ)² ]
```

**Manhattan Distance:**
```
d(x, z) = Σ |xᵢ - zᵢ|
```

**Minkowski Distance (generalization):**
```
d(x, z) = ( Σ |xᵢ - zᵢ|^p )^(1/p)

p=1 → Manhattan
p=2 → Euclidean
```

### Prediction

**Classification:**
```
ŷ = mode({ y(xᵢ) : xᵢ ∈ KNN(x) })
```

Take the most common class among K nearest neighbors.

**Weighted voting** (closer neighbors count more):
```
weight_i = 1 / d(x, xᵢ)
ŷ = argmax_c Σ weight_i * I(y_i == c)
```

### Visualization

```
         x₂
          |
        * |  +  +
          |   +
       *  |     +
    ------+--O--------  ← new point O, K=5
       *  |  ↑
          | find 5 nearest
          | 3 are * (class 0)
          | 2 are + (class 1)
          | predict: * (class 0)
          |___________________ x₁
```

### Effect of K

```
K=1: Very jagged boundary. Memorizes noise. Overfit.
K=n: Always predicts majority class. Underfit.
K≈√n: Good balance. Common heuristic.

Decision boundary complexity vs K:

K=1        K=5         K=20
|*|+|*|   |**|++|     |*** +++|
|+|*|+|   | * |+ |    |  **   |
|*|+|*|   |   |  |    |       |
(jagged)  (smoother)  (smooth)
```

### Time Complexity

| Phase | Brute Force | KD-Tree | Ball-Tree |
|---|---|---|---|
| Training | O(1) | O(n log n) | O(n log n) |
| Prediction | O(n·d) per query | O(log n) avg | O(log n) avg |

`n` = training samples, `d` = dimensions

---

## Technical Example — Fraud Detection

**Why K-NN for fraud (with caveats):**
- Good for finding anomalies — fraudulent transactions may cluster in unusual regions of feature space
- Easy to explain: "this transaction looks like these 5 similar known-fraud transactions"

**Caveats:**
- With millions of transactions, K-NN is too slow for production
- Better used as a baseline or for anomaly detection on smaller datasets

---

## Pseudocode

```
FUNCTION knn_predict(X_train, y_train, x_query, k, metric='euclidean'):
    distances = []

    FOR each (xᵢ, yᵢ) in training data:
        d = distance(x_query, xᵢ)
        distances.append((d, yᵢ))

    # Sort by distance ascending
    distances.sort(by first element)

    # Get K nearest labels
    k_labels = [label for (_, label) in distances[:k]]

    # Classification: majority vote
    RETURN most_common(k_labels)

    # Regression: mean
    # RETURN mean(k_labels)


FUNCTION euclidean_distance(x, z):
    RETURN sqrt(sum((xᵢ - zᵢ)² for each dimension i))
```

---

## Python Implementation — From Scratch

```python
import numpy as np
from collections import Counter

class KNNClassifierScratch:
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # KNN has no training — just store data
        self.X_train = X.copy()
        self.y_train = y.copy()
        return self

    def _distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _predict_one(self, x):
        # Compute distances to all training points
        distances = np.array([self._distance(x, xt) for xt in self.X_train])

        # Get K nearest indices
        k_idx = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_idx]

        # Majority vote
        return Counter(k_labels).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def predict_proba(self, X):
        """Fraction of neighbors in each class"""
        probas = []
        for x in X:
            distances = np.array([self._distance(x, xt) for xt in self.X_train])
            k_idx = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_idx]
            prob_1 = np.mean(k_labels)
            probas.append([1 - prob_1, prob_1])
        return np.array(probas)


# --- Test on fraud data ---
np.random.seed(42)
n = 500  # Small n — KNN is slow on large datasets
amount = np.random.exponential(200, n)
hour = np.random.randint(0, 24, n)
distance = np.random.exponential(50, n)
logit = -3 + 0.003*amount + 0.1*(hour>22) + 0.01*distance
prob = 1/(1 + np.exp(-logit))
y = (np.random.rand(n) < prob).astype(int)
X = np.column_stack([amount, hour, distance])

# CRITICAL: Scale before KNN
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

knn = KNNClassifierScratch(k=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(f"KNN (k=5) Accuracy: {np.mean(y_pred == y_test):.4f}")
```

---

## Sklearn Implementation

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np

# Data
np.random.seed(42)
n = 2000
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

# Scale — non-negotiable for KNN
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# --- Basic KNN ---
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',   # or 'distance' (closer neighbors weigh more)
    metric='euclidean',
    algorithm='auto',    # auto-selects brute/kd_tree/ball_tree
    n_jobs=-1
)
knn.fit(X_train_s, y_train)
print("=== KNN (k=5, uniform) ===")
print(classification_report(y_test, knn.predict(X_test_s),
      target_names=["Legit","Fraud"]))

# --- Distance-weighted KNN ---
knn_weighted = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',  # Closer neighbors vote more strongly
    n_jobs=-1
)
knn_weighted.fit(X_train_s, y_train)
print("=== KNN (k=5, distance-weighted) ===")
print(classification_report(y_test, knn_weighted.predict(X_test_s),
      target_names=["Legit","Fraud"]))

# --- Find best K ---
k_values = [1, 3, 5, 7, 10, 15, 20]
cv_scores = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='f1')
    cv_scores.append(scores.mean())
    print(f"K={k:2d}: CV F1 = {scores.mean():.4f}")

best_k = k_values[np.argmax(cv_scores)]
print(f"\nBest K: {best_k}")
```

---

## Key Interview Questions

**Q: Why must you scale features before KNN?**  
Distance is computed in raw feature space. A feature in thousands (e.g. salary) will dominate the distance calculation over a feature in units (e.g. number of transactions). Scaling puts all features on equal footing.

**Q: How do you choose K?**  
K=1 overfits. K=n underfits. Common approaches: use cross-validation to pick the K with best performance, or start with `K = √n` as a heuristic.

**Q: What is the curse of dimensionality?**  
In very high dimensions, all points become roughly equidistant from each other. "Nearest neighbor" loses meaning. KNN degrades badly beyond ~20 features.

**Q: KNN vs Logistic Regression?**  
KNN: no assumptions, captures complex boundaries, slow at prediction, can't explain feature impact. Logistic Regression: fast, interpretable, assumes linear boundary.

**Q: What is lazy learning?**  
The model does no computation at training time — all work happens at prediction time. Pros: adapts to new data instantly. Cons: slow predictions, large memory footprint.

---

## When to Use

✅ Small datasets (< 10k rows)  
✅ Non-linear boundaries  
✅ Anomaly detection (unusual points have no close neighbors)  
✅ Simple baseline that requires no hyperparameter tuning beyond K  

❌ Large datasets (prediction is O(n) per query)  
❌ High-dimensional data (curse of dimensionality)  
❌ Production systems needing fast inference  
❌ Missing values (can't compute distance easily)  
