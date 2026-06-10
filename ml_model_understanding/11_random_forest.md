# Random Forest

---

## Like a 5-year old

Imagine you want to decide whether to see a movie. Instead of asking one friend (who might have bad taste), you ask 100 different friends. Each friend looks at different things — one checks the director, one checks the actors, one checks the reviews. Then you go with whatever most friends recommend.

Random Forest builds hundreds of decision trees (your friends), each seeing a **random sample** of the data and a **random subset of features**. The final answer is the majority vote. Many imperfect trees together make a much better prediction than any single tree.

---

## Technical Understanding

Random Forest is an **ensemble of decision trees** using **bagging** (Bootstrap Aggregating) + **random feature selection**.

**Two sources of randomness:**
1. **Bootstrap sampling** — each tree trains on a random sample (with replacement) of the training data
2. **Feature randomness** — at each split, only a random subset of features is considered

**This decorrelates the trees** — if all trees saw the same data, they'd make similar errors. Randomness ensures their errors are independent, so averaging cancels them out.

**Key properties:**
- Robust to overfitting (unlike a single decision tree)
- Handles missing values, mixed types, and outliers well
- Built-in feature importance via Gini impurity reduction
- Out-of-bag (OOB) error — free validation without a separate validation set

---

## The Math

### Bagging (Bootstrap Aggregating)

For each tree `t = 1...T`:
```
1. Draw n samples WITH replacement from training set → bootstrap dataset Dₜ
   (~63.2% of original samples appear at least once, ~36.8% are OOB)

2. Train a full decision tree on Dₜ,
   but at each node, consider only m features (not all p)
   m = √p for classification, m = p/3 for regression

3. No pruning — trees grow fully
```

### Prediction

Classification (majority vote):
```
ŷ = mode({ tree_t(x) : t = 1...T })
```

Regression (average):
```
ŷ = (1/T) * Σₜ tree_t(x)
```

### Why Averaging Reduces Variance

Each tree has variance `σ²` and each pair of trees has correlation `ρ`.
Variance of averaged predictions:
```
Var(mean of T trees) = ρσ² + (1-ρ)σ²/T
```

- If trees are identical (`ρ=1`): no benefit from averaging
- If trees are uncorrelated (`ρ=0`): variance reduces by factor T
- Random features reduce `ρ`, making the ensemble much better

### Visualization

```
Training Data (n=1000)
        │
   ┌────┴─────┐
   │          │  Bootstrap samples (with replacement)
   ▼          ▼
  D₁         D₂  ...  D₁₀₀
   │          │
   ▼          ▼
 Tree₁      Tree₂  ...  Tree₁₀₀
(random    (random
features)  features)
   │          │
   └────┬─────┘
        │
        ▼
   Majority Vote
        │
        ▼
   Final Prediction
```

### Out-of-Bag (OOB) Error

The ~36.8% of samples not used in each bootstrap become a free validation set:

```
OOB score ≈ cross-validation score
            (computed without holding out data separately)

For each sample xᵢ:
  predict using only trees where xᵢ was NOT in their bootstrap sample
  compare to true yᵢ
```

### Feature Importance (Gini Importance)

```
Importance(feature j) = 
  Σ over all trees Σ over all nodes where j was used [
    (n_node / n_total) * Δ_gini_impurity
  ]
```

Higher = more total impurity reduction caused by this feature across all trees.

---

## Pseudocode

```
FUNCTION random_forest_fit(X, y, n_trees, max_features, max_depth):
    forest = []

    FOR t in range(n_trees):
        # Bootstrap sample
        bootstrap_idx = random_sample_with_replacement(range(n_samples), n_samples)
        X_boot = X[bootstrap_idx]
        y_boot = y[bootstrap_idx]

        # Train a tree with random feature selection at each split
        tree = build_decision_tree(
            X_boot, y_boot,
            max_depth=max_depth,
            max_features=max_features  # consider only sqrt(p) features per split
        )
        forest.append(tree)

    RETURN forest


FUNCTION random_forest_predict(forest, X):
    all_predictions = [tree.predict(X) for tree in forest]
    # Majority vote across all trees
    RETURN mode(all_predictions, axis=0)


# In each tree's split finding — the key change from single tree:
FUNCTION find_best_split_random(X, y, max_features):
    feature_subset = random_sample_without_replacement(
        range(n_features), size=max_features
    )
    # Only consider this random subset of features
    best_feat, best_thresh = find_best_split(X[:, feature_subset], y)
    RETURN feature_subset[best_feat], best_thresh
```

---

## Python Implementation — From Scratch

```python
import numpy as np
from collections import Counter

class SimpleDecisionTree:
    """Minimal decision tree for use inside Random Forest"""
    def __init__(self, max_depth=None, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None

    def gini(self, y):
        if len(y) == 0: return 0
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs**2)

    def best_split(self, X, y):
        n_features = X.shape[1]
        feature_idx = np.random.choice(n_features,
            size=self.max_features or n_features, replace=False)

        best_gini, best_feat, best_thresh = float('inf'), None, None
        for f in feature_idx:
            for thresh in np.unique(X[:, f]):
                left  = y[X[:, f] <= thresh]
                right = y[X[:, f] >  thresh]
                if not len(left) or not len(right): continue
                n = len(y)
                g = (len(left)/n)*self.gini(left) + (len(right)/n)*self.gini(right)
                if g < best_gini:
                    best_gini, best_feat, best_thresh = g, f, thresh
        return best_feat, best_thresh

    def build(self, X, y, depth=0):
        if (self.max_depth and depth >= self.max_depth) or len(np.unique(y)) == 1:
            return Counter(y).most_common(1)[0][0]
        f, t = self.best_split(X, y)
        if f is None:
            return Counter(y).most_common(1)[0][0]
        left_mask = X[:, f] <= t
        return {'feat': f, 'thresh': t,
                'left':  self.build(X[left_mask],  y[left_mask],  depth+1),
                'right': self.build(X[~left_mask], y[~left_mask], depth+1)}

    def fit(self, X, y): self.root = self.build(X, y); return self

    def _pred_one(self, node, x):
        if not isinstance(node, dict): return node
        if x[node['feat']] <= node['thresh']: return self._pred_one(node['left'], x)
        else: return self._pred_one(node['right'], x)

    def predict(self, X): return np.array([self._pred_one(self.root, x) for x in X])


class RandomForestScratch:
    def __init__(self, n_trees=100, max_depth=None, max_features='sqrt', random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.oob_indices = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        n, p = X.shape
        mf = int(np.sqrt(p)) if self.max_features == 'sqrt' else self.max_features

        for _ in range(self.n_trees):
            # Bootstrap
            idx = np.random.choice(n, n, replace=True)
            oob = np.setdiff1d(np.arange(n), idx)
            self.oob_indices.append(oob)

            tree = SimpleDecisionTree(max_depth=self.max_depth, max_features=mf)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict(self, X):
        votes = np.array([t.predict(X) for t in self.trees])
        return np.array([Counter(votes[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])


# --- Test ---
np.random.seed(42)
n = 500
amount = np.random.exponential(200, n)
hour = np.random.randint(0, 24, n)
distance = np.random.exponential(50, n)
logit = -3 + 0.003*amount + 0.1*(hour>22) + 0.01*distance
y = (np.random.rand(n) < 1/(1+np.exp(-logit))).astype(int)
X = np.column_stack([amount, hour, distance])

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)
split = int(0.8 * n)
X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

rf = RandomForestScratch(n_trees=50, max_depth=5)
rf.fit(X_tr, y_tr)
print(f"Random Forest Accuracy: {np.mean(rf.predict(X_te) == y_te):.4f}")
```

---

## Sklearn Implementation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

np.random.seed(42)
n = 2000
amount = np.random.exponential(200, n)
hour = np.random.randint(0, 24, n)
distance = np.random.exponential(50, n)
is_foreign = np.random.binomial(1, 0.1, n)
logit = -3 + 0.003*amount + 0.1*(hour>22) + 0.01*distance + 0.8*is_foreign
y = (np.random.rand(n) < 1/(1+np.exp(-logit))).astype(int)
X = np.column_stack([amount, hour, distance, is_foreign])
feature_names = ["amount", "hour", "distance", "is_foreign"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=200,       # number of trees
    max_depth=None,         # grow fully (controlled by min_samples_leaf)
    min_samples_leaf=5,     # min 5 samples per leaf
    max_features='sqrt',    # sqrt(p) features per split
    oob_score=True,         # compute OOB score
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)

print("=== Random Forest ===")
print(classification_report(y_test, rf.predict(X_test), target_names=["Legit","Fraud"]))
print(f"OOB Score: {rf.oob_score_:.4f}")

print("\n=== Feature Importances ===")
for name, imp in sorted(zip(feature_names, rf.feature_importances_),
                         key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {name:12s}: {imp:.4f}  {bar}")
```

---

## Key Interview Questions

**Q: Why does Random Forest outperform a single decision tree?**  
A single tree overfits. Many diverse trees average out each other's errors. Diversity comes from bootstrap sampling + random feature selection.

**Q: What is OOB error?**  
~36.8% of training samples are excluded from each bootstrap sample. These become a free test set for that tree. Averaging across all trees gives an unbiased estimate of generalization error.

**Q: How many trees is enough?**  
Performance improves until ~100–300 trees, then plateaus. More trees never hurt (no overfit), just slower.

**Q: Random Forest vs XGBoost?**  
Random Forest trains trees in **parallel**, each independently. XGBoost trains trees **sequentially**, each correcting the last. XGBoost usually wins on accuracy; Random Forest is faster to train and more robust to hyperparameters.

---

## When to Use

✅ Tabular data, classification or regression  
✅ When you want feature importances  
✅ Good default model — works well out of the box  
✅ Handles missing values and mixed types well  
✅ Less hyperparameter tuning than XGBoost  

❌ Very large datasets (many trees are slow)  
❌ When interpretability of a single model is needed (use one decision tree)  
❌ When XGBoost/LightGBM is already available (usually beats RF on accuracy)  
