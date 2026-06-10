# Hierarchical Clustering

---

## Like a 5-year old

Imagine sorting animals into groups. You start with every animal alone. Then you say: "a dog and a wolf look similar — put them together." Then "cats and tigers look similar — put them together." Then "dogs+wolves and cats+tigers are both mammals — put them all in one group."

You keep merging the most similar groups until everything is in one big group. This makes a family tree (a **dendrogram**) that shows all the relationships. You can cut the tree at any level to get any number of clusters you want.

---

## Technical Understanding

Hierarchical clustering builds a **tree of clusters** (dendrogram) by either:
- **Agglomerative (bottom-up):** Start with n clusters (each point alone), merge the two most similar clusters at each step
- **Divisive (top-down):** Start with 1 cluster (all points), recursively split into smaller clusters

**Key advantage over K-Means:**  
- No need to specify K upfront
- Produces a dendrogram showing all levels of clustering
- Can detect non-spherical clusters with the right linkage

**Linkage criteria** — how to measure similarity between two clusters:

| Linkage | Distance = | Shape of clusters |
|---|---|---|
| Single | min distance between any two points | Can detect elongated clusters, but "chains" |
| Complete | max distance between any two points | Compact, equal-sized clusters |
| Average | mean distance between all pairs | Balance between single and complete |
| Ward | minimizes total within-cluster variance | Most common, similar to K-Means |

---

## The Math

### Distance Matrix

```
Initial: compute pairwise distances between all n points
D[i][j] = euclidean(xᵢ, xⱼ)

Matrix (n=4 points):
        A    B    C    D
   A  [ 0   1.2  5.3  5.1 ]
   B  [1.2   0   4.8  4.9 ]
   C  [5.3  4.8   0   0.8 ]
   D  [5.1  4.9  0.8   0  ]
```

### Agglomerative Algorithm

```
Step 1: Each point is its own cluster
         {A}, {B}, {C}, {D}

Step 2: Find two closest clusters
         D(A,B) = 1.2 → merge → {A,B}, {C}, {D}

Step 3: Recompute distances with Ward linkage
         D({A,B},{C}) = ..., D({A,B},{D}) = ..., D(C,D) = 0.8 → merge C,D
         {A,B}, {C,D}

Step 4: Merge remaining
         {A,B,C,D}
```

### Dendrogram

```
Height
(merge distance)
    |
5.0 |            ┌────────────┐
    |             │            │
3.0 |        ┌───┘      ┌─────┘
    |         │          │
1.2 |    ┌───┘     ┌────┘
    |     │         │
0.8 |  ┌──┘     ┌──┘
    |  │         │
0   |  A  B     C  D

Cut the dendrogram at height 3.0 → 2 clusters: {A,B} and {C,D}
Cut at height 1.5 → 3 clusters: {A,B}, {C}, {D}
```

---

## Pseudocode

```
FUNCTION agglomerative_clustering(X, linkage='ward'):
    n = len(X)
    clusters = [{i} for i in range(n)]        # Start: each point is a cluster
    D = compute_pairwise_distances(X)          # n×n distance matrix
    history = []                               # Merge history for dendrogram

    WHILE len(clusters) > 1:
        i*, j* = find_two_closest_clusters(clusters, D, linkage)
        new_cluster = clusters[i*] ∪ clusters[j*]
        history.append((i*, j*, D[i*][j*]))

        clusters.remove(clusters[i*])
        clusters.remove(clusters[j*])
        clusters.append(new_cluster)

        # Recompute distances to the new merged cluster
        UPDATE D with new cluster distances

    RETURN history  # Used to build dendrogram
```

---

## Python + Sklearn Implementation

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

# --- Customer data ---
np.random.seed(42)
seg0 = np.random.multivariate_normal([1000, 5],   [[50000,0],[0,4]],  100)
seg1 = np.random.multivariate_normal([20000, 30], [[1e7,0],[0,50]],   100)
seg2 = np.random.multivariate_normal([5000, 10],  [[500000,0],[0,10]], 100)
X = np.vstack([seg0, seg1, seg2])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Agglomerative (sklearn) ---
agg = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'   # 'single', 'complete', 'average', 'ward'
)
labels = agg.fit_predict(X_scaled)
print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")
for i in range(3):
    print(f"Cluster {i}: {np.sum(labels==i)} points")

# --- Dendrogram (scipy) ---
# Compute linkage matrix
Z = linkage(X_scaled[:50], method='ward')  # small subset for speed
print("\nFirst 5 merges:")
print("Cluster1  Cluster2  Distance  Size")
for row in Z[:5]:
    print(f"  {int(row[0]):4d}      {int(row[1]):4d}   {row[2]:8.3f}    {int(row[3]):3d}")
```

---

## When to Use

✅ Don't know K upfront  
✅ Need to understand cluster hierarchy  
✅ Small-medium datasets (< 10k rows — O(n²) memory)  
✅ Non-spherical clusters (with single/average linkage)  

❌ Very large datasets (memory O(n²), time O(n² log n))  
❌ When you know K and just need fast clustering (use K-Means)  
