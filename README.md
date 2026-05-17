# Part 1 — Neural Network Analysis
## Customer Churn Prediction with a Feed-Forward Neural Network

---

## Problem Statement
Build and analyze a feed-forward neural network to predict whether a telecom customer will churn (leave the service), demonstrating the full ML pipeline: forward pass → loss → backpropagation → parameter updates.

---

## Dataset
| Property | Value |
|----------|-------|
| File | `customer_churn_nn.csv` |
| Rows | 2,000 |
| Features | 15 (after dropping `customer_id`) |
| Target | `churn` (0 = stays, 1 = churns) |
| Class balance | 98.45% No Churn / 1.55% Churn |

**Feature types:**
- **Categorical:** region, plan_type, contract_type, payment_method
- **Numerical:** tenure_months, monthly_charges_inr, avg_login_days_per_month, support_tickets_last_90_days, payment_delay_days, data_usage_gb, satisfaction_score, last_complaint_days_ago, discount_percent, autopay_enabled, referral_count

---

## Model Architecture
```
Input Layer       →  15 neurons
Hidden Layer 1    →  64 neurons  (ReLU)
Hidden Layer 2    →  32 neurons  (ReLU)
Output Layer      →  1 neuron    (Sigmoid → binary cross-entropy)
Optimizer         →  Adam  |  LR = 0.001  |  Batch size = 32
```

---

## Results

### Baseline Model
| Metric | Value |
|--------|-------|
| Train Accuracy | 100.00% |
| Test Accuracy  | 97.50%  |
| Final Loss     | 0.0009  |

### Hyperparameter Comparison

| Config | Hidden Layers | Activation | LR | Epochs | Train Acc | Test Acc | Loss |
|--------|--------------|------------|-------|--------|-----------|----------|------|
| Exp1 — Baseline | (64, 32) | relu | 0.001 | 200 | 1.0000 | 0.9750 | 0.0009 |
| Exp2 — Deeper Net | (128, 64, 32) | relu | 0.001 | 200 | 1.0000 | **0.9800** | 0.0005 |
| Exp3 — High LR | (64, 32) | relu | 0.01 | 200 | 1.0000 | 0.9725 | 0.0009 |
| Exp4 — Tanh Act. | (64, 32) | tanh | 0.001 | 200 | 1.0000 | 0.9700 | 0.0017 |
| Exp5 — Low LR | (64, 32) | relu | 0.0001 | 300 | 0.9981 | 0.9800 | 0.0109 |

**Best configuration:** Exp2 (Deeper Net) or Exp5 (Low LR) — both achieve 98.00% test accuracy.

---

## Repository Structure
```
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_comparison_table.csv
    ├── model_comparison_table.png
    └── evaluation_outputs.png
```

---

## How to Run

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd part-1-neural-network-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset in the same folder
#    customer_churn_nn.csv  →  part-1-neural-network-analysis/

# 4. Launch the notebook
jupyter notebook notebook.ipynb
```

---

## Key Reflections
- **Weights & biases** are learned parameters; weights scale connections, biases shift activations for flexibility.
- **Activation functions** (ReLU) introduce non-linearity — without them, the network collapses to a single linear transformation.
- **High LR** → overshoots, oscillates; **Low LR** → slow convergence, may underfit within epoch budget.
- This dataset shows **class imbalance** — high accuracy (~98%) is misleading. Minority-class recall needs SMOTE or class weighting.
