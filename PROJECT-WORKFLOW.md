# 🚀 Machine Learning Project Master Template

- **Project Name:** [Insert Project Name]  
- **Objective:** [e.g., Binary Classification of Fraud / Regression for Pricing]  
- **Target Variable:** `[Target Column Name]`

---

## 📊 Phase 1: Data Understanding & EDA

*Execute this phase to completely map out the data landscape before transforming anything.*

### ✅ Checklist

- [ ] **Data Source & Integrity Check**
  - Source: `[Synthetic / Scraped / Production DB]`
  - Total Rows: `[Count]`
  - Total Columns: `[Count]`
  - Checked for duplicate rows:

```python
df.duplicated().sum()
```

- [ ] **Target Leakage Verification**
  - Manually reviewed features to ensure no columns contain future or proxy information about the target.

- [ ] **Missingness Pattern Analysis**
  - Calculated exact null percentages per column.
  - Determined missingness strategy:
    - **Numerical:** Median (skewed) / Mean (normal)
    - **Categorical:** Mode / `'Missing'`

```python
(df.isnull().sum() / len(df)) * 100
```

- [ ] **Univariate & Distribution Analysis**
  - Checked categorical feature cardinality:

```python
df[col].nunique()
```

  - Mark heavily skewed columns:

```python
df.skew()
```

  - Outlier boundaries (IQR Method):

```text
Lower Bound = Q1 - 1.5 * IQR
Upper Bound = Q3 + 1.5 * IQR
```

- [ ] **Multivariate & Target Analysis**
  - Correlation matrix for multicollinearity:

```python
df.corr()
```

  - Flag correlations > `0.85`
  - Check target imbalance ratio:

```python
df[target].value_counts(normalize=True)
```

---

## ⚙️ Phase 2: Preprocessing & Feature Engineering

*The transformation sandbox. Keep train and validation completely separated here.*

### ✅ Checklist

- [ ] **Data Splitting Strategy**
  - [ ] Standard Split (`80/10/10`)
  - [ ] Stratified Split (Imbalanced)
  - [ ] Time-Series Split (Chronological)

```python
from sklearn.model_selection import train_test_split
```

- [ ] **Data Leakage Guard Active**
  - `.fit()` on training only
  - `.transform()` on validation/test

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
```

- [ ] **Categorical Encoding**
  - Low Cardinality (<10) → One-Hot Encoding
  - High Cardinality (>10) → Target Encoding / Embeddings
  - Ordinal → Explicit Mapping

- [ ] **Feature Scaling**
  - [ ] `StandardScaler`
  - [ ] `MinMaxScaler`
  - [ ] `RobustScaler`

- [ ] **Feature Dropping**
  - Remove:
    - IDs
    - Zero-variance constants
    - Raw text blobs
    - Columns with >50% missing values

---

## 🏗️ Phase 3: Architecture Design & Compilation

*Building the model engine systematically.*

### ✅ Checklist

- [ ] **Establish Naive Baseline**

```python
from sklearn.dummy import DummyClassifier, DummyRegressor
```

Compare baseline before building complex models.

- [ ] **Set Global Reproducibility**

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

- [ ] **Layer Shape Verification**
  - Input Layer = `X_train.shape[1]`

Output Layer Rules:

| Task | Nodes | Activation |
|---|---:|---|
| Regression | 1 | Linear |
| Binary Classification | 1 | Sigmoid |
| Multi-Class Classification | N | Softmax |

- [ ] **Weights Initialization**
  - ReLU / LeakyReLU → He/Kaiming
  - Tanh / Sigmoid → Xavier/Glorot

- [ ] **Regularization Integration**
  - Dropout (`0.2 – 0.5`)
  - Batch Normalization

- [ ] **Loss & Optimizer Compilation**

Examples:

```python
# Classification
BCEWithLogitsLoss()
CrossEntropyLoss()

# Regression
MSELoss()
HuberLoss()
```

Optimizer:

```python
torch.optim.AdamW(model.parameters(), lr=3e-4)
```

- [ ] **Training Callbacks Configured**
  - `ReduceLROnPlateau`
  - `EarlyStopping`
  - Gradient Clipping (`max_norm=1.0`)

---

## 📊 Phase 4: Training Diagnostics & Evaluation

*How to interpret live outputs.*

### 🛠️ Live Loss Curve Diagnostic Cheat Sheet

| Pattern | Meaning | Action |
|---|---|---|
| Train ↓, Val ↑ | Overfitting | Add dropout, regularization, more data |
| Train ↑, Val ↑ (flat high) | Underfitting | Increase model capacity |
| Sawtooth Val Loss | LR too high | Lower learning rate |
| Flatline from start | Vanishing gradients | Use LeakyReLU / BatchNorm |

---

## 📈 Phase 5: Final Metric Scorecard

*Do not rely on accuracy alone.*

### Classification Tasks

| Metric | Value | Interpretation |
|---|---|---|
| Accuracy | `______` | Only reliable if balanced dataset |
| Precision | `______` | Important when FP is costly |
| Recall | `______` | Important when FN is dangerous |
| F1 Score | `______` | Best for imbalance |
| PR-AUC | `______` | Strong minority-class metric |

---

### Regression Tasks

| Metric | Value | Interpretation |
|---|---|---|
| MAE | `______` | Average absolute error |
| RMSE | `______` | High = large outlier mistakes |
| R² Score | `______` | Must be > 0 |

---

## 🔍 Model Interpretation Quick Guide

### Classification

#### Precision vs Recall

```text
High Precision + Low Recall → Conservative model
Low Precision + High Recall → Aggressive model
High Both → Strong model
Low Both → Weak model
```

#### F1 Score

```text
0.90+ → Excellent
0.80–0.89 → Strong
0.70–0.79 → Decent
<0.70 → Needs improvement
```

---

### Regression

#### MAE vs RMSE

```text
RMSE ≈ MAE → Errors are consistent
RMSE >> MAE → Large outliers exist
```

#### R² Score

```text
1.0 → Perfect fit
0.8+ → Strong
0.5–0.8 → Moderate
<0.5 → Weak
<0 → Worse than mean predictor
```

---

## 🚀 Production Readiness Check

- [ ] Model inference latency tested under mock production load.
- [ ] Exported model weights.
- [ ] Version controlled.
- [ ] Quantized for deployment (if edge device).
- [ ] Saved preprocessing pipeline.
- [ ] Input schema validated.
- [ ] Logging & monitoring prepared.

---

## 📂 Recommended Project Structure

```text
project/
│── data/
│── notebooks/
│── src/
│   │── preprocessing.py
│   │── model.py
│   │── train.py
│   │── evaluate.py
│── models/
│── results/
│── README.md
│── requirements.txt
```

---

## 📝 Notes

Standard checklist Template to ensure:

- Proper data handling
- Zero leakage
- Better diagnostics
- Faster debugging
- Production-ready workflows