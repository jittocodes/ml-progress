# Apriori Algorithm

---

## Like a 5-year old

Imagine you run a grocery store and you notice that people who buy bread almost always buy butter too. And people who buy diapers often buy beer (a famous real-world finding!).

The Apriori algorithm looks through all your shopping receipts and finds these patterns: "things that get bought together." It does this smart — if "milk alone" is rare, then "milk + eggs + bread" will be even rarer, so it doesn't even bother checking those big combinations.

---

## Technical Understanding

Apriori is an **association rule learning** algorithm. It discovers relationships between items in a dataset — specifically, it finds rules like:

```
{bread, butter} → {jam}
"People who buy bread and butter also tend to buy jam"
```

**Unsupervised** — no labels, just transaction data.

**Main application:** Market basket analysis, recommendation systems, fraud pattern detection.

**Three key metrics:**

| Metric | Formula | Meaning |
|---|---|---|
| **Support** | freq(A∪B) / n_transactions | How often A and B appear together |
| **Confidence** | freq(A∪B) / freq(A) | Given A, how often does B appear? |
| **Lift** | confidence(A→B) / support(B) | How much more likely is B given A vs random? |

Lift > 1 means the association is genuine (not just because B is popular).

---

## The Math

### Support

```
support(A → B) = P(A ∪ B) = freq(A and B together) / total transactions
```

Minimum support threshold filters out rare itemsets.

### Confidence

```
confidence(A → B) = P(B | A) = P(A ∪ B) / P(A)
```

High confidence = B almost always follows A.

### Lift

```
lift(A → B) = confidence(A → B) / support(B)
            = P(A ∪ B) / (P(A) * P(B))
```

- Lift = 1: A and B are independent
- Lift > 1: A and B appear together more than expected (genuine association)
- Lift < 1: A and B appear together less than expected (negative association)

### The Apriori Principle

**"If an itemset is infrequent, all its supersets are also infrequent."**

```
{milk} → infrequent
Then: {milk, bread}, {milk, eggs}, {milk, bread, eggs} → all infrequent
→ prune entire branch — don't check them

This dramatically reduces search space.
```

### Algorithm Steps

```
Step 1: Find all frequent 1-itemsets (items meeting min_support)
  L1 = {bread, butter, milk, eggs, ...}

Step 2: Generate 2-itemset candidates from L1, prune using Apriori principle
  C2 = all pairs from L1
  L2 = C2 where support >= min_support

Step 3: Generate 3-itemset candidates from L2, prune
  C3 = all triples where all sub-pairs are in L2
  L3 = C3 where support >= min_support

Step 4: Continue until no new frequent itemsets found

Step 5: Generate association rules from all frequent itemsets
  For each itemset X, for each partition A | B = X:
      compute confidence(A → B)
      keep if >= min_confidence
```

### Visualization

```
Transactions:
T1: {bread, butter, jam}
T2: {bread, butter, milk}
T3: {butter, milk, eggs}
T4: {bread, butter, eggs}
T5: {bread, milk}

Support counts:
bread:  4/5 = 0.80
butter: 4/5 = 0.80
milk:   3/5 = 0.60
eggs:   2/5 = 0.40
jam:    1/5 = 0.20

With min_support=0.40: jam dropped (0.20 < 0.40)
L1 = {bread, butter, milk, eggs}

2-itemsets:
{bread,butter}: 3/5 = 0.60 ✓
{bread,milk}:   2/5 = 0.40 ✓
{butter,milk}:  2/5 = 0.40 ✓
{bread,eggs}:   1/5 = 0.20 ✗ pruned
{butter,eggs}:  1/5 = 0.20 ✗ pruned

Rules from {bread,butter} (support=0.60):
  bread → butter: confidence = 0.60/0.80 = 0.75, lift = 0.75/0.80 = 0.94
  butter → bread: confidence = 0.60/0.80 = 0.75, lift = 0.75/0.80 = 0.94
```

---

## Pseudocode

```
FUNCTION apriori(transactions, min_support, min_confidence):
    # Step 1: Find frequent 1-itemsets
    item_counts = count_all_items(transactions)
    L_current = {item for item, count in item_counts
                 if count/n_transactions >= min_support}

    all_frequent = [L_current]
    k = 2

    WHILE L_current is not empty:
        # Generate k-itemset candidates
        C_k = generate_candidates(L_current, k)

        # Prune using Apriori principle
        C_k = {c for c in C_k
               if all subsets of size k-1 are in L_current}

        # Count support in transactions
        item_counts = count_itemsets(transactions, C_k)
        L_current = {itemset for itemset, count in item_counts
                     if count/n_transactions >= min_support}

        IF L_current: all_frequent.append(L_current)
        k += 1

    # Generate rules
    rules = []
    FOR each frequent itemset X in all_frequent:
        FOR each non-empty proper subset A of X:
            B = X - A
            conf = support(X) / support(A)
            lift = conf / support(B)
            IF conf >= min_confidence:
                rules.append((A → B, support(X), conf, lift))

    RETURN rules
```

---

## Python + mlxtend Implementation

```python
# pip install mlxtend
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# --- Simulate supermarket transactions ---
np.random.seed(42)
item_pool = ['bread', 'butter', 'milk', 'eggs', 'jam', 'cheese', 'yogurt', 'juice']

# Create 1000 transactions with correlations
transactions = []
for _ in range(1000):
    basket = []
    if np.random.rand() < 0.7:   basket.append('bread')
    if np.random.rand() < 0.6:   basket.append('butter')
    if 'bread' in basket and np.random.rand() < 0.8:   basket.append('jam')  # bread → jam
    if 'butter' in basket and np.random.rand() < 0.7:  basket.append('bread')
    if np.random.rand() < 0.5:   basket.append('milk')
    if 'milk' in basket and np.random.rand() < 0.6:    basket.append('eggs')
    if np.random.rand() < 0.3:   basket.append('cheese')
    basket = list(set(basket))
    if basket:
        transactions.append(basket)

# --- Encode transactions ---
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

print(f"Transactions: {len(df)}")
print(f"Items: {list(df.columns)}")
print(f"\nItem frequencies:")
print(df.mean().sort_values(ascending=False).round(3))

# --- Run Apriori ---
frequent_itemsets = apriori(
    df,
    min_support=0.1,      # item must appear in 10%+ of transactions
    use_colnames=True,
    max_len=3             # up to 3-item combinations
)
print(f"\nFrequent itemsets found: {len(frequent_itemsets)}")
print(frequent_itemsets.sort_values('support', ascending=False).head(10))

# --- Generate rules ---
rules = association_rules(
    frequent_itemsets,
    metric='lift',
    min_threshold=1.2     # lift > 1.2 = genuine association
)

print(f"\nRules found: {len(rules)}")
print("\nTop rules by lift:")
top = rules.sort_values('lift', ascending=False).head(10)
for _, row in top.iterrows():
    ant = set(row['antecedents'])
    con = set(row['consequents'])
    print(f"  {ant} → {con}")
    print(f"    support={row['support']:.3f}, "
          f"confidence={row['confidence']:.3f}, "
          f"lift={row['lift']:.3f}")

# --- Filter for high-confidence rules ---
print("\nHigh confidence rules (>= 0.8):")
high_conf = rules[rules['confidence'] >= 0.8].sort_values('confidence', ascending=False)
for _, row in high_conf.head(5).iterrows():
    print(f"  {set(row['antecedents'])} → {set(row['consequents'])}: "
          f"conf={row['confidence']:.3f}, lift={row['lift']:.3f}")
```

---

## Key Interview Questions

**Q: What does lift > 1 mean?**  
The items appear together more often than you'd expect by chance. Lift = 1 means independence. Lift = 2 means they appear together twice as often as random chance would predict.

**Q: What is the Apriori principle?**  
"If an itemset is infrequent, all its supersets are infrequent." This lets you prune the search space — if {milk} is rare, you don't need to check {milk, bread, eggs, butter}.

**Q: Apriori vs FP-Growth?**  
FP-Growth is much faster — it compresses the transaction database into an FP-tree and doesn't need to generate candidates. Apriori generates and tests candidates explicitly. Use `fpgrowth` in mlxtend for larger datasets.

**Q: What's the difference between confidence and lift?**  
Confidence `P(B|A)` can be high just because B is very popular overall. Lift corrects for this by dividing by `P(B)`. Always use lift to identify genuine associations.

---

## When to Use

✅ Market basket analysis (what items to recommend together)  
✅ Medical co-occurrence (symptoms that appear together)  
✅ Fraud pattern detection (transaction patterns that co-occur with fraud)  
✅ Web clickstream analysis  

❌ Very large item catalogs (exponential candidates)  
❌ Continuous features (requires discretization first)  
❌ When you need predictions, not just patterns (use supervised learning)  
