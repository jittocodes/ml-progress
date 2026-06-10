# K-Means Clustering

---

## Like a 5-year old

Imagine you have a bag of mixed candy — red, green, and yellow ones — but nobody told you the colors. You want to sort them into 3 piles based on how similar they look.

You start by randomly picking one candy for each pile. Then every other candy joins the pile whose starting candy it looks most like. After sorting, you pick new "center" candies for each pile that best represent that pile. You keep re-sorting until nothing changes.

That's K-Means. It groups things that look similar together, without being told what the groups should be.

---

## Technical Understanding

K-Means is an **unsupervised clustering** algorithm. It partitions `n` data points into `K` clusters, where each point belongs to the cluster with the **nearest centroid**.

**Key properties:**
- No labels needed — discovers structure in data
- Requires you to specify K (number of clusters) upfront
- Minimizes **inertia** — total squared distance from each point to its cluster centroid
- Assumes clusters are **spherical and equally sized** (its main weakness)
- Sensitive to outliers and initial centroid placement

**Common uses in fintech/data engineering:**
- Customer segmentation (high-value vs low-value users)
- Transaction behavior grouping
- Anomaly detection (points far from all centroids = outliers)

---

## The Math

### Objective Function — Inertia

```
Inertia = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²

Where:
  Cᵢ = cluster i
  μᵢ = centroid of cluster i
  ||x - μᵢ||² = squared Euclidean distance
```

Goal: **minimize total inertia** by finding the best centroids.

### The Algorithm (Lloyd's Algorithm)

```
Step 1: Initialize K centroids randomly (or K-Means++ strategy)

Step 2: Assignment step
  For each point x:
      Assign to cluster i* = argmin_i ||x - μᵢ||²

Step 3: Update step
  For each cluster i:
      μᵢ = (1/|Cᵢ|) * Σₓ∈Cᵢ x   ← new centroid = mean of assigned points

Step 4: Repeat steps 2-3 until centroids stop moving (convergence)
```

### Visualization of Iterations

```
Initial (random centroids X):

   * *         + +
  *  X *       + X +        Cluster assignment
    * *            +
       ? ? ?               "?" points are ambiguous
      ?  X ?

After iteration 1:

   * *         + +
  * X *        + X +        Centroids moved to cluster means
    * *           +

After iteration 3 (converged):

   * *         + +
  * [X] *      + [X] +      Centroids settled, clusters stable
    * *           +
```

### K-Means++ Initialization

Random initialization can converge to bad solutions. K-Means++ spreads initial centroids apart:

```
1. Choose first centroid randomly
2. For each subsequent centroid:
   - Probability of choosing point x ∝ d(x, nearest centroid)²
   - Points farther from existing centroids are more likely chosen
```

This gives consistently better results and is the default in sklearn.

### Choosing K — The Elbow Method

Plot inertia vs K:

```
Inertia
  |
  |*
  | *
  |  *
  |   *
  |    *──────────────   ← "elbow" here = optimal K
  |              
  |_______________________ K
  1  2  3  4  5  6  7
        ↑
     optimal K
```

Also use **Silhouette Score** (measures how well-separated clusters are):
- Range: [-1, 1]
- Close to 1 = well-separated clusters
- Close to 0 = overlapping clusters

---

## Technical Example — Customer Segmentation

**Problem:** Segment bank customers into groups by behavior, without predefined labels.

**Features:** `avg_monthly_balance`, `num_transactions_per_month`, `avg_transaction_amount`

**Expected clusters:**
- Cluster 0: Low balance, few txns (low-activity customers)
- Cluster 1: High balance, many txns (high-value customers)
- Cluster 2: Medium balance, high-value single transactions (occasional large spenders)

---

## Pseudocode

```
FUNCTION kmeans(X, k, max_iters=300, tol=1e-4):
    # Initialize centroids using K-Means++
    centroids = kmeans_plus_plus_init(X, k)

    FOR iteration in range(max_iters):
        # Assignment step
        labels = []
        FOR each point x in X:
            distances = [euclidean(x, c) for c in centroids]
            labels.append(argmin(distances))

        # Update step
        new_centroids = []
        FOR cluster_i in range(k):
            points_in_cluster = X[labels == cluster_i]
            new_centroids.append(mean(points_in_cluster, axis=0))

        # Check convergence
        shift = max(euclidean(c_old, c_new)
                    for c_old, c_new in zip(centroids, new_centroids))
        centroids = new_centroids

        IF shift < tol:
            BREAK

    RETURN labels, centroids


FUNCTION kmeans_plus_plus_init(X, k):
    centroids = [random choice from X]

    FOR _ in range(k-1):
        distances = [min(euclidean(x, c)² for c in centroids) for x in X]
        probs = distances / sum(distances)
        next_centroid = weighted_random_choice(X, probs)
        centroids.append(next_centroid)

    RETURN centroids
```

---

## Python Implementation — From Scratch

```python
import numpy as np

class KMeansScratch:
    def __init__(self, k=3, max_iters=300, tol=1e-4, init='kmeans++', random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        if self.init == 'random':
            idx = rng.choice(len(X), self.k, replace=False)
            return X[idx].copy()

        # K-Means++
        centroids = [X[rng.integers(len(X))]]
        for _ in range(self.k - 1):
            dists = np.array([
                min(np.sum((x - c)**2) for c in centroids) for x in X
            ])
            probs = dists / dists.sum()
            idx = rng.choice(len(X), p=probs)
            centroids.append(X[idx])
        return np.array(centroids)

    def fit(self, X):
        self.centroids = self._init_centroids(X)

        for iteration in range(self.max_iters):
            # Assignment
            dists = np.array([[np.sum((x - c)**2) for c in self.centroids] for x in X])
            labels = np.argmin(dists, axis=1)

            # Update
            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i)
                else self.centroids[i]  # keep old centroid if cluster is empty
                for i in range(self.k)
            ])

            # Convergence check
            shift = np.max(np.linalg.norm(self.centroids - new_centroids, axis=1))
            self.centroids = new_centroids
            self.labels_ = labels

            if shift < self.tol:
                print(f"Converged at iteration {iteration}")
                break

        # Compute inertia
        self.inertia_ = sum(
            np.sum((X[labels == i] - self.centroids[i])**2)
            for i in range(self.k)
        )
        return self

    def predict(self, X):
        dists = np.array([[np.sum((x - c)**2) for c in self.centroids] for x in X])
        return np.argmin(dists, axis=1)


# --- Customer segmentation example ---
np.random.seed(42)
n = 600

# Simulate 3 customer segments
seg0 = np.random.multivariate_normal([1000, 5, 100],   [[50000,0,0],[0,4,0],[0,0,200]],  200)
seg1 = np.random.multivariate_normal([20000, 30, 500], [[1e7,0,0],[0,50,0],[0,0,5000]], 200)
seg2 = np.random.multivariate_normal([5000, 10, 2000], [[500000,0,0],[0,10,0],[0,0,50000]], 200)

X = np.vstack([seg0, seg1, seg2])

from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

model = KMeansScratch(k=3, random_state=42)
model.fit(X_scaled)

print(f"Inertia: {model.inertia_:.2f}")
for i in range(3):
    n_pts = np.sum(model.labels_ == i)
    print(f"Cluster {i}: {n_pts} customers")
```

---

## Sklearn Implementation

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

# Data — customer segments
np.random.seed(42)
seg0 = np.random.multivariate_normal([1000, 5, 100],   [[50000,0,0],[0,4,0],[0,0,200]],  200)
seg1 = np.random.multivariate_normal([20000, 30, 500], [[1e7,0,0],[0,50,0],[0,0,5000]], 200)
seg2 = np.random.multivariate_normal([5000, 10, 2000], [[500000,0,0],[0,10,0],[0,0,50000]], 200)
X = np.vstack([seg0, seg1, seg2])
feature_names = ["avg_balance", "num_txns", "avg_txn_amount"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Fit K=3 ---
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
kmeans.fit(X_scaled)

print("=== K-Means (k=3) ===")
print(f"Inertia: {kmeans.inertia_:.2f}")
print(f"Silhouette Score: {silhouette_score(X_scaled, kmeans.labels_):.4f}")
for i in range(3):
    center_orig = scaler.inverse_transform([kmeans.cluster_centers_[i]])[0]
    n_pts = np.sum(kmeans.labels_ == i)
    print(f"\nCluster {i} ({n_pts} customers):")
    for name, val in zip(feature_names, center_orig):
        print(f"  {name}: {val:.1f}")

# --- Elbow method to find optimal K ---
print("\n=== Elbow Method ===")
inertias = []
sil_scores = []
k_range = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))
    print(f"K={k}: Inertia={km.inertia_:.1f}, Silhouette={sil_scores[-1]:.4f}")

best_k = k_range[np.argmax(sil_scores)]
print(f"\nBest K by silhouette: {best_k}")
```

---

## Key Interview Questions

**Q: How do you choose K?**  
Elbow method (plot inertia vs K, find the bend) + Silhouette score (maximize). Domain knowledge also helps — you might know you want 3 customer tiers.

**Q: What are K-Means' main weaknesses?**  
Assumes spherical clusters of equal size. Fails on elongated, irregular, or very different-sized clusters. Sensitive to outliers (they pull centroids). Must specify K in advance.

**Q: K-Means vs DBSCAN?**  
K-Means: need to specify K, assumes round clusters, fast. DBSCAN: discovers K automatically, handles arbitrary shapes, identifies outliers as noise, slower.

**Q: Why run K-Means multiple times (n_init)?**  
Different initializations can converge to different local minima. Running multiple times and picking the best (lowest inertia) reduces this risk. `n_init=10` is sklearn's default.

**Q: What is inertia?**  
The sum of squared distances from each point to its assigned centroid. Lower = tighter clusters. Used for the elbow method.

---

## When to Use

✅ Customer segmentation  
✅ Document/text clustering  
✅ Anomaly detection (outliers have high distance to nearest centroid)  
✅ Preprocessing for supervised learning (cluster as a feature)  
✅ Fast and scalable (O(n·K·d·iterations))  

❌ Non-spherical clusters  
❌ Clusters of very different sizes or densities  
❌ Unknown K (try DBSCAN or hierarchical instead)  
❌ Datasets with many outliers  
