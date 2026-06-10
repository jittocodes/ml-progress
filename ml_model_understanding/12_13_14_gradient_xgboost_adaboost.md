# Gradient Boosting

---

## Like a 5-year old

Imagine you're trying to throw darts at a target. You throw the first dart and it lands too far to the right. Your second dart aims specifically at correcting that — you aim a bit to the left. The third dart corrects whatever is still off. Each dart learns from where the previous ones missed.

Gradient Boosting builds trees **sequentially**, where each new tree specifically tries to fix the mistakes of all the previous trees combined.

---

## Technical Understanding

Gradient Boosting builds an ensemble by **adding trees one at a time**, where each new tree fits the **residuals** (errors) of the current ensemble.

**Key difference from Random Forest:**
- Random Forest: trees built in **parallel**, independently
- Gradient Boosting: trees built **sequentially**, each correcting the last

**Key properties:**
- Usually outperforms Random Forest on tabular data
- More sensitive to hyperparameters
- Prone to overfit if learning rate is too high or trees too deep
- Slower to train than Random Forest (sequential)

---

## The Math

### Residuals and Gradient

At each step, the new tree fits the **negative gradient** of the loss function — the direction that reduces error most.

For MSE loss (regression):
```
Residual = y - ŷ_current
```

For log-loss (classification):
```
Pseudo-residual = -∂L/∂ŷ = y - sigmoid(ŷ_current)
```

### Algorithm

```
Step 0: Initialize with a constant prediction
  F₀(x) = argmin_γ Σ L(yᵢ, γ)    (e.g. mean(y) for MSE)

For m = 1 to M:
  Step 1: Compute pseudo-residuals
    rᵢₘ = -[∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)]    for each sample i

  Step 2: Fit a shallow tree hₘ to the pseudo-residuals rᵢₘ

  Step 3: Update the model
    Fₘ(x) = Fₘ₋₁(x) + α * hₘ(x)

Where α = learning rate (shrinkage)
```

### Visualization — Residuals Decreasing

```
Iteration 0: Predict mean(y) = 200 for all
  Residuals: [+500, -150, +300, -200, +100...]

Iteration 1: Tree fits residuals → corrects big errors
  New predictions: [650, 60, 480, 10, 290...]
  Residuals: [+50, -10, +120, -10, -90...]

Iteration 2: Tree fits remaining residuals → smaller corrections
  ...

After 100 iterations: residuals are tiny
```

### Learning Rate vs Number of Trees

```
High learning rate (α=0.5): large steps, fast but unstable, overfit
Low learning rate  (α=0.01): small steps, slow but stable, need more trees

Best practice: small α (0.01-0.1) + many trees (100-1000)
  "More smaller steps is better than fewer bigger steps"
```

---

## Python + Sklearn Implementation

```python
from sklearn.ensemble import GradientBoostingClassifier
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

gb = GradientBoostingClassifier(
    n_estimators=200,        # number of trees
    learning_rate=0.1,       # shrinkage — smaller = more robust
    max_depth=3,             # shallow trees preferred
    subsample=0.8,           # stochastic GB: use 80% of data per tree
    min_samples_leaf=10,
    random_state=42
)
gb.fit(X_train, y_train)

print("=== Gradient Boosting ===")
print(classification_report(y_test, gb.predict(X_test), target_names=["Legit","Fraud"]))

print("\n=== Feature Importances ===")
for name, imp in sorted(zip(feature_names, gb.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:12s}: {imp:.4f}")

# Training curve — see when it stops improving
train_scores = [gb.train_score_[i] for i in range(0, 200, 10)]
print(f"\nInitial train loss: {gb.train_score_[0]:.4f}")
print(f"Final  train loss: {gb.train_score_[-1]:.4f}")
```

---

## When to Use

✅ Tabular data, best default after Random Forest  
✅ When you need to squeeze out maximum accuracy  
✅ Handles missing values and mixed types  

❌ Needs careful hyperparameter tuning  
❌ Slower than Random Forest  
❌ Use XGBoost/LightGBM in practice (faster, better regularization)  

---
---

# XGBoost (Extreme Gradient Boosting)

---

## Like a 5-year old

XGBoost is Gradient Boosting but a supercharged, optimized version. It adds:
- A **speed engine** (parallel computation, column sub-sampling)
- A **safety belt** (regularization to prevent overfit)
- A **smart split finder** (approximate histogram-based)

Think of Gradient Boosting as a regular car and XGBoost as a race car — same idea, built much better.

---

## Technical Understanding

XGBoost improves on Gradient Boosting with:

1. **Regularization** — L1 (Lasso) and L2 (Ridge) terms on tree weights → prevents overfitting
2. **Column subsampling** — like Random Forest, only consider a fraction of features per tree → faster + less overfit
3. **Approximate split finding** — uses quantile sketching instead of checking every value → much faster
4. **Handling missing values** — learns the best direction to send missing values
5. **Parallel computation** — can run column-level computations in parallel

---

## The Math

### XGBoost Objective

```
Obj = Σᵢ L(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)

Where Ω(f) = γT + (1/2)λ||w||²
  T = number of leaves
  w = leaf weights
  γ = penalty per leaf (controls tree complexity)
  λ = L2 penalty on leaf weights
```

This regularization term directly penalizes complex trees, unlike standard GBM.

### Taylor Expansion

XGBoost uses a 2nd-order Taylor approximation of the loss, enabling exact closed-form leaf weights:

```
w*_j = -Gⱼ / (Hⱼ + λ)

Where:
  Gⱼ = Σᵢ∈leaf_j gᵢ   (sum of first-order gradients)
  Hⱼ = Σᵢ∈leaf_j hᵢ   (sum of second-order gradients / Hessians)
```

### Split Gain Formula

```
Gain = (1/2) * [Gₗ²/(Hₗ+λ) + Gᵣ²/(Hᵣ+λ) - (Gₗ+Gᵣ)²/(Hₗ+Hᵣ+λ)] - γ
```

If Gain < 0, the split is rejected (tree is pruned).

---

## Python Implementation

```python
# pip install xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

np.random.seed(42)
n = 5000
amount = np.random.exponential(200, n)
hour = np.random.randint(0, 24, n)
distance = np.random.exponential(50, n)
is_foreign = np.random.binomial(1, 0.1, n)
num_prev_txns = np.random.randint(1, 100, n)
logit = -3 + 0.003*amount + 0.1*(hour>22) + 0.01*distance + 0.8*is_foreign - 0.01*num_prev_txns
y = (np.random.rand(n) < 1/(1+np.exp(-logit))).astype(int)
X = np.column_stack([amount, hour, distance, is_foreign, num_prev_txns])
feature_names = ["amount", "hour", "distance", "is_foreign", "num_prev_txns"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Scale pos weight for class imbalance ---
scale = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class ratio (scale_pos_weight): {scale:.2f}")

# --- Fit XGBoost ---
xgb_clf = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,          # small lr + many trees
    max_depth=4,                 # shallow
    subsample=0.8,               # row subsampling
    colsample_bytree=0.8,        # column subsampling (like RF)
    reg_alpha=0.1,               # L1 regularization
    reg_lambda=1.0,              # L2 regularization
    scale_pos_weight=scale,      # handle class imbalance
    eval_metric='logloss',
    early_stopping_rounds=30,    # stop if no improvement for 30 rounds
    random_state=42,
    n_jobs=-1
)

xgb_clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50
)

y_pred = xgb_clf.predict(X_test)
y_proba = xgb_clf.predict_proba(X_test)[:, 1]

print("\n=== XGBoost ===")
print(classification_report(y_test, y_pred, target_names=["Legit","Fraud"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Best iteration: {xgb_clf.best_iteration}")

print("\n=== Feature Importances ===")
for name, imp in sorted(zip(feature_names, xgb_clf.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {name:14s}: {imp:.4f}  {bar}")

# --- SHAP values ---
import shap
explainer = shap.TreeExplainer(xgb_clf)
shap_values = explainer.shap_values(X_test[:100])
print(f"\nMean |SHAP| per feature:")
for name, val in zip(feature_names, np.abs(shap_values).mean(0)):
    print(f"  {name:14s}: {val:.4f}")
```

---

## Key Interview Questions

**Q: XGBoost vs sklearn GradientBoosting?**  
XGBoost is faster (parallel column computation, histogram splits), has built-in regularization (L1/L2 + leaf count penalty), handles missing values natively, and has early stopping. Use XGBoost in practice.

**Q: What is `scale_pos_weight`?**  
For imbalanced datasets, set `scale_pos_weight = negative/positive`. This tells XGBoost to pay more attention to the minority class (fraud), equivalent to class weighting.

**Q: What is `colsample_bytree`?**  
The fraction of features randomly sampled for each tree (like Random Forest's `max_features`). Reduces correlation between trees, improves generalization.

**Q: Why small learning rate + more trees?**  
Many small corrections → smoother, more accurate final prediction. Early stopping prevents unnecessary trees.

---

## When to Use

✅ Best default model for tabular data in competitive settings  
✅ Handles missing values natively  
✅ SHAP values for explainability  
✅ Industry standard for fraud detection  

❌ Very large datasets (use LightGBM — faster)  
❌ Image/text/sequence data (use neural networks)  

---
---

# AdaBoost (Adaptive Boosting)

---

## Like a 5-year old

Imagine you have a quiz and you keep getting some questions wrong. A smart teacher keeps making you practice the questions you get wrong more often, and less the ones you already know. Each round, you focus on your weaknesses. After many rounds, you're good at everything.

AdaBoost works the same way. After each weak tree, it increases the **weight** of misclassified samples so the next tree focuses on getting those right.

---

## Technical Understanding

AdaBoost is the **original boosting algorithm** (1995). It combines many **weak classifiers** (stumps — trees with depth=1) into a strong one.

**Key difference from Gradient Boosting:**
- AdaBoost: reweights **samples** based on errors
- Gradient Boosting: fits **residuals** (errors themselves)
- AdaBoost uses stumps (1-level trees); GBM uses deeper trees

**Historical note:** AdaBoost was the first successful boosting algorithm. It proved theoretically that combining weak learners creates a strong learner. Gradient Boosting generalized this idea.

---

## The Math

### Sample Weights

```
Initial weights: wᵢ = 1/n  for all i

For each round m:
  1. Train stump hₘ on weighted samples
  2. Compute weighted error:
     εₘ = Σᵢ wᵢ * I(yᵢ ≠ hₘ(xᵢ)) / Σᵢ wᵢ

  3. Compute stump weight:
     αₘ = (1/2) * log((1 - εₘ) / εₘ)
     (high weight if error is low, negative weight if error > 0.5)

  4. Update sample weights:
     wᵢ ← wᵢ * exp(-αₘ * yᵢ * hₘ(xᵢ))
     Normalize so Σwᵢ = 1

     → Misclassified samples get HIGHER weight
     → Correctly classified get LOWER weight
```

### Final Prediction

```
F(x) = sign( Σₘ αₘ * hₘ(x) )
```

Weighted vote of all stumps. Stumps with lower error get more vote weight.

### Visualization

```
Round 1: Equal weights
  * * * * + + + + + + +
  Stump splits well on big groups
  Some * misclassified → increase their weights

Round 2: Misclassified * have bigger circles
  *(big) * * * +(big) + + + + + +
  Next stump focuses on the big circles

Round 3: Previous errors now handled
  ...

After M rounds: all regions well-covered by ensemble
```

---

## Python + Sklearn Implementation

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

np.random.seed(42)
n = 2000
amount = np.random.exponential(200, n)
hour = np.random.randint(0, 24, n)
distance = np.random.exponential(50, n)
logit = -3 + 0.003*amount + 0.1*(hour>22) + 0.01*distance
y = (np.random.rand(n) < 1/(1+np.exp(-logit))).astype(int)
X = np.column_stack([amount, hour, distance])
feature_names = ["amount", "hour", "distance"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- AdaBoost with decision stumps ---
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # stump
    n_estimators=200,
    learning_rate=0.5,
    algorithm='SAMME',
    random_state=42
)
ada.fit(X_train, y_train)

print("=== AdaBoost ===")
print(classification_report(y_test, ada.predict(X_test), target_names=["Legit","Fraud"]))

print("\n=== Feature Importances ===")
for name, imp in sorted(zip(feature_names, ada.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:12s}: {imp:.4f}")

# Stump weights (first 5)
print("\nFirst 5 stump weights (estimator_weights_):")
for i, w in enumerate(ada.estimator_weights_[:5]):
    print(f"  Stump {i+1}: weight = {w:.4f}")

# --- Compare all ensemble methods ---
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

models = {
    'AdaBoost':          AdaBoostClassifier(n_estimators=200, random_state=42),
    'Random Forest':     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost':           xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                                            eval_metric='logloss', verbosity=0),
}

print("\n=== Ensemble Method Comparison ===")
print(f"{'Model':<22} {'Accuracy':>10} {'F1 (Fraud)':>12}")
print("-" * 46)
from sklearn.metrics import accuracy_score, f1_score
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    print(f"{name:<22} {acc:>10.4f} {f1:>12.4f}")
```

---

## Key Interview Questions

**Q: AdaBoost vs Gradient Boosting?**  
AdaBoost reweights samples and uses stumps with equal-depth trees. Gradient Boosting fits residuals directly with adjustable tree depth. GBM is more general and usually better, but AdaBoost was the foundational algorithm.

**Q: Why do stumps (depth=1) work in AdaBoost?**  
Each stump is a weak learner — barely better than random. But the ensemble of hundreds of stumps, each focused on different errors, is very powerful.

**Q: What happens if a stump has error > 0.5?**  
`αₘ` becomes negative — the stump's vote is inverted. This can happen but typically means the feature selection was poor.

**Q: When would you use AdaBoost over XGBoost?**  
Rarely in practice. AdaBoost is historically important but XGBoost almost always outperforms it. AdaBoost is faster and simpler to understand conceptually.

---

## When to Use

✅ Understanding boosting conceptually  
✅ Simple datasets where stumps capture the key patterns  
✅ When training speed matters more than peak accuracy  

❌ Most real tasks — XGBoost/LightGBM will outperform  
❌ Very noisy data (AdaBoost is sensitive to outliers — they get high weights and dominate)  
