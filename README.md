# Lab 3: Contextual Bandit-Based News Article Recommendation System

**Student:** Raghav Sarna  
**Roll Number:** U20230106  
**Course:** Reinforcement Learning Fundamentals  
**Branch:** `raghav_U20230106`

---

## 1. Overview

This project implements a **Contextual Multi-Armed Bandit (CMAB)** framework for recommending news articles to users. The system:
1. Classifies users into one of three categories (User1, User2, User3) using a Random Forest classifier.
2. Uses the user category as **context** for a contextual bandit.
3. Selects the optimal **news category** (Entertainment, Education, Tech, Crime) as the bandit arm.
4. Recommends a randomly sampled article from the chosen category.

---

## 2. Approach & Design Decisions

### Data Pre-processing
- Missing `age` values (~35% missing in both train and test) are imputed with the column median.
- Categorical features (`region_code`, `subscriber`, `browser_version`) are label-encoded.
- `user_id` is dropped as a non-predictive identifier.

### User Classification (Context Detection)
- A **Random Forest Classifier** (200 trees, max_depth=15) is trained on 80% of `train_users.csv`.
- Achieves **~91% accuracy** on the 20% validation split.
- The full training set is then used to train the final classifier for predicting test user categories.

### Contextual Bandit Algorithms
Three strategies are implemented, each trained **separately per user context** (3 contexts × 4 arms = 12 total arms):

| Arm Index (j) | News Category  | User Context |
|:---:|:---:|:---:|
| 0–3 | Entertainment, Education, Tech, Crime | User1 |
| 4–7 | Entertainment, Education, Tech, Crime | User2 |
| 8–11 | Entertainment, Education, Tech, Crime | User3 |

1. **Epsilon-Greedy**: Tested with ε ∈ {0.01, 0.1, 0.3}
2. **Upper Confidence Bound (UCB)**: Tested with C ∈ {0.5, 1.0, 2.0}
3. **SoftMax (Boltzmann)**: Fixed temperature τ = 1.0

All simulations run for **T = 10,000 steps**.

---

## 3. Key Results

### Expected Reward Distribution (Estimated)

| Context | Entertainment | Education | Tech | Crime |
|:---:|:---:|:---:|:---:|:---:|
| User1 | -0.25 | **3.77** | -6.20 | -1.75 |
| User2 | -8.63 | -1.89 | **4.77** | 2.69 |
| User3 | -7.55 | -5.36 | **10.92** | -1.44 |

### Optimal Categories
- **User1** → Education (expected reward ≈ 3.77)
- **User2** → Tech (expected reward ≈ 4.77)
- **User3** → Tech (expected reward ≈ 10.92)

### Algorithm Performance (Mean Reward over T=10,000)

| Algorithm | User1 | User2 | User3 |
|:---|:---:|:---:|:---:|
| ε-Greedy (ε=0.1) | 3.31 | 4.19 | 9.74 |
| UCB (C=1.0) | 3.77 | 4.75 | 10.87 |
| SoftMax (τ=1.0) | 3.68 | 4.49 | 10.86 |

### Classification Accuracy
- Validation accuracy: **90.75%**
- Precision/Recall/F1 all above 0.84 for every class.

---

## 4. Observations

### Hyperparameter Sensitivity
- **Epsilon-Greedy**: Lower ε (0.01) converges faster to the optimal arm but explores less. Higher ε (0.3) explores more, reducing overall average reward by ~30%.
- **UCB**: Performance is relatively stable across C values (0.5, 1.0, 2.0) because UCB's exploration term naturally decays with more pulls. All C values converge to similar final rewards.
- **SoftMax**: With τ=1, it quickly concentrates probability on the best arm, especially when reward gaps are large (User3). Competitive with UCB.

### Comparative Analysis
- **UCB** achieves the highest mean reward across all contexts due to its adaptive exploration bonus.
- **SoftMax** is a close second, performing especially well when the best arm has a large reward gap.
- **ε-Greedy** is the simplest but wastes exploration budget uniformly across all arms regardless of their estimated values.

### Contextual Bandit Value
- Different user contexts have different optimal news categories (User1→Education, User2→Tech, User3→Tech), demonstrating that a **contextual** approach outperforms a single global policy.

---

## 5. How to Reproduce

### Prerequisites
- Python ≥ 3.12
- Install dependencies:
```bash
pip install rlcmab-sampler scikit-learn pandas numpy matplotlib
```

### Run
1. Open `lab3_results_U20230106.ipynb` in Jupyter or VS Code.
2. Run all cells top to bottom.
3. The notebook will produce all results, tables, and plots.

---

## 6. Files

| File | Description |
|:---|:---|
| `lab3_results_U20230106.ipynb` | Main notebook with all code, results, and visualizations |
| `data/news_articles.csv` | News articles dataset (209,527 articles) |
| `data/train_users.csv` | Training user data (2,000 users with labels) |
| `data/test_users.csv` | Test user data (2,000 users, no labels) |
| `README.md` | This project report |
