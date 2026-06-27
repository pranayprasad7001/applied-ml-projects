# 🚀 Machine Learning Project Workflow

* **Project Name:** [Insert Project Name]
* **Objective:** [e.g., Binary Classification of Fraud / Regression for Pricing]
* **Target Variable:** `[Target Column Name]`

---

## 📊 Phase 1: Data Understanding & EDA

*Map the data landscape before transforming anything.*

### ✅ Checklist

* [ ] **Data Source & Integrity Check**

  * Source: `[Synthetic / Scraped / Production DB]`
  * Total Rows: `[Count]`
  * Total Columns: `[Count]`

```python
df.duplicated().sum()
```

* [ ] **Target Leakage Verification**

  * Ensure no feature contains future or proxy target information.

* [ ] **Missingness Pattern Analysis**

```python
(df.isnull().sum() / len(df)) * 100
```

Decision Rules:

* Numerical → Median (skewed), Mean (normal)

* Categorical → Mode / `'Missing'`

* [ ] **Univariate Analysis**

```python
df[col].nunique()
df.skew()
```

Skew Rule:

* Between -1 and 1 → okay

* > 1 or <-1 → transformation candidate

* [ ] **Outlier Detection (IQR)**

```text
Lower = Q1 - 1.5 × IQR
Upper = Q3 + 1.5 × IQR
```

* [ ] **Multivariate Analysis**

```python
df.corr()
```

Flag:

* Correlation > 0.85 → multicollinearity risk

* [ ] **Target Imbalance Check**

```python
df[target].value_counts(normalize=True)
```

---

## ⚙️ Phase 2: Preprocessing & Feature Engineering

*Transform carefully. Split first.*

### ✅ Checklist

* [ ] **Data Splitting Strategy**

  * Standard Split (`80/10/10`)
  * Stratified Split (imbalanced)
  * Time-Series Split (chronological)

```python
from sklearn.model_selection import train_test_split
```

* [ ] **Leakage Prevention**

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
```

Rule:

* Fit only on train

* Transform on validation/test

* [ ] **Categorical Encoding**

  * Low cardinality → OneHot
  * High cardinality → Target Encoding / Embeddings
  * Ordinal → Explicit Mapping

* [ ] **Feature Scaling**

  * StandardScaler → Normal distributions
  * MinMaxScaler → Neural Networks
  * RobustScaler → Heavy outliers

* [ ] **Feature Selection**

  * Remove high correlation (>0.85)
  * Remove low variance features
  * Remove irrelevant features

* [ ] **Feature Dropping**

  * IDs
  * Raw text blobs
  * Constants
  * > 50% missing columns

---

## 🏗️ Phase 3: Architecture Design & Compilation

*Build systematically.*

### ✅ Checklist

* [ ] **Naive Baseline**

```python
from sklearn.dummy import DummyClassifier, DummyRegressor
```

* [ ] **Reproducibility**

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

* [ ] **Input/Output Layer Verification**

Input:

```python
X_train.shape[1]
```

Output Rules:

| Task                  | Nodes | Activation |
| --------------------- | ----: | ---------- |
| Regression            |     1 | Linear     |
| Binary Classification |     1 | Sigmoid    |
| Multi-Class           |     N | Softmax    |

* [ ] **Weight Initialization**

  * ReLU/LeakyReLU → He
  * Tanh/Sigmoid → Xavier

* [ ] **Regularization**

  * Dropout (`0.2–0.5`)
  * BatchNorm
  * L2 Weight Decay

* [ ] **Loss Selection**

```python
# Classification
BCEWithLogitsLoss()
CrossEntropyLoss()

# Regression
MSELoss()
HuberLoss()
```

* [ ] **Optimizer**

```python
torch.optim.AdamW(model.parameters(), lr=3e-4)
```

* [ ] **Callbacks**

  * ReduceLROnPlateau
  * EarlyStopping
  * Gradient Clipping (`max_norm=1.0`)

---

## 📊 Phase 4: Training Diagnostics & Evaluation

*Monitor learning behavior.*

### Validation Strategy Check

* [ ] K-Fold Cross Validation
* [ ] Stratified K-Fold
* [ ] TimeSeriesSplit

---

### 🛠️ Loss Curve Diagnostic Cheat Sheet

| Pattern                    | Meaning               | Action                        |
| -------------------------- | --------------------- | ----------------------------- |
| Train ↓, Val ↑             | Overfitting           | More regularization           |
| Train ↑, Val ↑ (flat high) | Underfitting          | Bigger model                  |
| Sawtooth Val Loss          | LR too high           | Lower LR                      |
| Flatline from start        | No effective learning | Check LR / init / activations |
| Sudden spikes              | Gradient explosion    | Gradient clipping             |

---

## 📈 Phase 5: Final Metric Interpretation

*Understand performance beyond raw numbers.*

### Classification Metrics

| Metric    | Good Sign         | Warning Sign    | Interpretation           |
| --------- | ----------------- | --------------- | ------------------------ |
| Accuracy  | High + F1 aligned | High but low F1 | May be misleading        |
| Precision | >0.80             | Low             | Too many false positives |
| Recall    | >0.80             | Low             | Missing positives        |
| F1 Score  | Balanced with P/R | Low             | Poor balance             |
| PR-AUC    | >0.70             | <0.50           | Minority class quality   |
| ROC-AUC   | >0.80             | ~0.50           | Separability             |

#### Quick Rules

```text
High Accuracy + Low Recall = Minority ignored
High Precision + Low Recall = Conservative model
Low Precision + High Recall = Aggressive model
High Both = Strong model
Low Both = Weak model
```

---

### Regression Metrics

| Metric      | Good Sign    | Warning Sign | Interpretation              |
| ----------- | ------------ | ------------ | --------------------------- |
| MAE         | Low          | High         | Average error magnitude     |
| RMSE        | Close to MAE | Much higher  | Outlier mistakes            |
| R² Score    | >0.80        | <0.50        | Explained variance          |
| Adjusted R² | Close to R²  | Much lower   | Irrelevant features present |

#### Quick Rules

```text
RMSE ≈ MAE → Stable predictions
RMSE >> MAE → Outlier issues
R² = 1 → Perfect fit
R² > 0.8 → Strong
R² < 0 → Worse than baseline
```

---

### Loss Relationship Check

| Scenario                      | Meaning                   |
| ----------------------------- | ------------------------- |
| Train Loss ↓ + Val Loss ↓     | Healthy learning          |
| Train Loss ↓ + Val Loss ↑     | Overfitting               |
| Both High + Flat              | Underfitting              |
| Val Loss unstable             | LR too high               |
| Train Loss low + metrics poor | Leakage / threshold issue |

---

## 🚀 Production Readiness Check

* [ ] Inference latency tested

* [ ] Model version controlled

* [ ] Preprocessing pipeline saved

* [ ] Input schema validated

* [ ] Logging & monitoring prepared

* [ ] **Saved deployment format**

  * `.pkl` → Scikit-learn
  * `.pt` → PyTorch
  * `.onnx` → Cross-platform
  * `.tflite` → Mobile/Edge

* [ ] Quantized / pruned if required

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

This workflow ensures:

* Proper data handling
* Zero leakage
* Better diagnostics
* Faster debugging
* Reproducibility
* Production readiness