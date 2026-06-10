# Decision Trees

---

## Like a 5-year old

Imagine playing "20 questions." You ask: "Is it an animal?" → Yes. "Does it have 4 legs?" → Yes. "Is it bigger than a dog?" → No. "Is it a cat?" → Yes!

A decision tree works exactly like that. It asks a series of yes/no questions about your data, and eventually lands on an answer. Each question splits the data into groups, and the groups get purer and purer until you can make a confident guess.

---

## Technical Understanding

A decision tree is a flowchart-like structure where:
- **Internal nodes** = a feature test (e.g. `amount > 500?`)
- **Branches** = outcome of the test (yes/no or value ranges)
- **Leaf nodes** = final prediction (class label or value)

The tree is built by **recursively splitting** the dataset on the feature and threshold that best separates the classes.

**Two main uses:**
- **Classification trees** — predict a category
- **Regression trees** — predict a continuous value

**Key properties:**
- Non-parametric — makes no assumptions about data distribution
- Handles mixed feature types (numeric + categorical)
- Highly interpretable — can be drawn and explained
- Prone to **overfitting** — memorizes training data if unconstrained

---

## The Math

### Splitting Criterion — Gini Impurity

Measures how "mixed" a node is. Lower = purer.

```
Gini(node) = 1 - Σ pᵢ²
```

Where `pᵢ` = proportion of class `i` in the node.

**Example:**
- Node with 50% fraud, 50% legit: Gini = 1 - (0.5² + 0.5²) = 0.5 → most impure
- Node with 100% fraud: Gini = 1 - (1² + 0²) = 0 → perfectly pure

### Information Gain (alternative to Gini)

```
Entropy(node) = -Σ pᵢ * log₂(pᵢ)
Information Gain = Entropy(parent) - Σ (|child|/|parent|) * Entropy(child)
```

Higher information gain = better split.

**Gini vs Entropy:** Gini is slightly faster (no log computation). Both usually give similar trees. Gini is sklearn's default.

### How a split is chosen

```
For each feature f:
    For each possible threshold t:
        Split data into left (f <= t) and right (f > t)
        Compute weighted Gini of the two children
        
Choose the (f, t) pair with the lowest weighted Gini
```

### Visualization of a split

```
              Root (all transactions)
              Gini = 0.18
              n = 1000
                   |
         amount > 500?
          /              \
        YES               NO
       /                    \
  Left node              Right node
  Gini = 0.05            Gini = 0.12
  n = 200                n = 800
  90% fraud              15% fraud
       |                      |
  PREDICT: Fraud         hour_of_day > 22?
                          /           \
                        YES            NO
                       /                 \
                 Gini = 0.04         Gini = 0.08
                 70% fraud           5% fraud
                 PREDICT: Fraud      PREDICT: Legit
```

### Regularization Hyperparameters

| Parameter | What it controls | Effect of increasing |
|---|---|---|
| `max_depth` | Max levels in the tree | Simpler tree, less overfit |
| `min_samples_split` | Min samples to split a node | Bigger nodes before splitting |
| `min_samples_leaf` | Min samples in a leaf | Smoother predictions |
| `max_features` | Features considered per split | Faster, more random |

---

## Technical Example — Fraud Detection

**Problem:** Classify transactions as fraud (1) or legit (0).

**Features:** `amount`, `hour_of_day`, `distance_from_home`, `is_foreign_country`

**Why decision trees work well here:**
- Rules like "amount > 1000 AND hour > 22 → fraud" are naturally captured
- Handles the binary `is_foreign_country` feature without encoding

---

## Pseudocode

```
FUNCTION build_tree(X, y, depth=0, max_depth=5):
    IF all y same class OR depth == max_depth:
        RETURN LeafNode(most_common_class(y))

    best_feature, best_threshold = find_best_split(X, y)

    left_mask  = X[:, best_feature] <= best_threshold
    right_mask = X[:, best_feature] >  best_threshold

    left_subtree  = build_tree(X[left_mask],  y[left_mask],  depth+1, max_depth)
    right_subtree = build_tree(X[right_mask], y[right_mask], depth+1, max_depth)

    RETURN InternalNode(best_feature, best_threshold, left_subtree, right_subtree)


FUNCTION find_best_split(X, y):
    best_gini = infinity
    best_feature = None
    best_threshold = None

    FOR each feature f in range(n_features):
        FOR each unique value t in X[:, f]:
            left  = y[X[:, f] <= t]
            right = y[X[:, f] >  t]

            gini = weighted_gini(left, right)

            IF gini < best_gini:
                best_gini = gini
                best_feature = f
                best_threshold = t

    RETURN best_feature, best_threshold


FUNCTION gini_impurity(y):
    classes, counts = unique_with_counts(y)
    probs = counts / len(y)
    RETURN 1 - sum(probs ** 2)


FUNCTION weighted_gini(left, right):
    n = len(left) + len(right)
    RETURN (len(left)/n) * gini(left) + (len(right)/n) * gini(right)


FUNCTION predict(node, x):
    IF node is LeafNode:
        RETURN node.class_label
    IF x[node.feature] <= node.threshold:
        RETURN predict(node.left, x)
    ELSE:
        RETURN predict(node.right, x)
```

---

## Python Implementation — From Scratch

```python
import numpy as np
from collections import Counter

class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # set for leaf nodes

    def is_leaf(self):
        return self.value is not None


class DecisionTreeScratch:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def gini(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def best_split(self, X, y):
        best_gini = float('inf')
        best_feat, best_thresh = None, None

        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_y  = y[X[:, feat] <= thresh]
                right_y = y[X[:, feat] >  thresh]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                n = len(y)
                g = (len(left_y)/n) * self.gini(left_y) + \
                    (len(right_y)/n) * self.gini(right_y)

                if g < best_gini:
                    best_gini = g
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def build(self, X, y, depth=0):
        # Stopping conditions
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):
            return DecisionNode(value=Counter(y).most_common(1)[0][0])

        feat, thresh = self.best_split(X, y)
        if feat is None:
            return DecisionNode(value=Counter(y).most_common(1)[0][0])

        left_mask  = X[:, feat] <= thresh
        right_mask = ~left_mask

        left  = self.build(X[left_mask],  y[left_mask],  depth + 1)
        right = self.build(X[right_mask], y[right_mask], depth + 1)

        return DecisionNode(feature=feat, threshold=thresh, left=left, right=right)

    def fit(self, X, y):
        self.root = self.build(X, y)

    def _predict_one(self, node, x):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(node.left, x)
        else:
            return self._predict_one(node.right, x)

    def predict(self, X):
        return np.array([self._predict_one(self.root, x) for x in X])


# --- Simulate fraud data ---
np.random.seed(42)
n = 1000
amount = np.random.exponential(200, n)
hour = np.random.randint(0, 24, n)
distance = np.random.exponential(50, n)
is_foreign = np.random.binomial(1, 0.1, n)

logit = -3 + 0.003*amount + 0.1*(hour>22) + 0.01*distance + 0.8*is_foreign
prob = 1 / (1 + np.exp(-logit))
y = (np.random.rand(n) < prob).astype(int)

X = np.column_stack([amount, hour, distance, is_foreign])
split = int(0.8 * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

tree = DecisionTreeScratch(max_depth=4)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print(f"Accuracy: {np.mean(y_pred == y_test):.4f}")
```

---

## Sklearn Implementation

```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Data
np.random.seed(42)
n = 1000
amount = np.random.exponential(200, n)
hour = np.random.randint(0, 24, n)
distance = np.random.exponential(50, n)
is_foreign = np.random.binomial(1, 0.1, n)
logit = -3 + 0.003*amount + 0.1*(hour>22) + 0.01*distance + 0.8*is_foreign
prob = 1 / (1 + np.exp(-logit))
y = (np.random.rand(n) < prob).astype(int)
X = np.column_stack([amount, hour, distance, is_foreign])
feature_names = ["amount", "hour", "distance", "is_foreign"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Fit tree ---
clf = DecisionTreeClassifier(
    criterion='gini',      # or 'entropy'
    max_depth=4,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

# --- Print the tree rules ---
print("\n=== Tree Rules ===")
print(export_text(clf, feature_names=feature_names))

# --- Feature importance ---
print("=== Feature Importances ===")
for name, imp in zip(feature_names, clf.feature_importances_):
    print(f"  {name}: {imp:.4f}")

# --- Visualize the tree (save to file) ---
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(clf, feature_names=feature_names,
          class_names=["Legit", "Fraud"],
          filled=True, rounded=True, ax=ax)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
print("\nTree diagram saved to decision_tree.png")
```

---

## Key Interview Questions

**Q: Gini vs Entropy — which to use?**  
Both give similar results. Gini is faster (no log). Entropy can sometimes produce more balanced trees. Default to Gini.

**Q: Why do trees overfit?**  
A fully grown tree memorizes every training point (1 sample per leaf). Fix with `max_depth`, `min_samples_leaf`, or pruning.

**Q: What are feature importances?**  
The total reduction in Gini impurity caused by splits on that feature, across all nodes. Higher = more useful for prediction.

**Q: When would you prefer a decision tree over logistic regression?**  
Non-linear boundaries, interactions between features, mixed data types, or when you need human-readable rules for compliance.

---

## When to Use

✅ Need interpretable rules ("if amount > 500 AND hour > 22 → fraud")  
✅ Mixed feature types  
✅ Non-linear decision boundaries  
✅ Fast training  

❌ Alone (overfit easily — prefer Random Forest or XGBoost)  
❌ Extrapolation beyond training data range  
❌ Small changes in data can drastically change the tree  
