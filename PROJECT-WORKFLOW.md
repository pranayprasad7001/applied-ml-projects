# 🚀 Machine Learning Project Workflow

* **Project Name:** [Insert Project Name]
* **Objective:** [e.g., Binary Classification / Regression / Forecasting]
* **Target Variable:** `[Target Column Name]`

---

# 📊 Phase 1: Data Understanding & EDA

*Understand the dataset before transformations.*

## ✅ Checklist

* [ ] Verify dataset source
* [ ] Check duplicates

```python
df.duplicated().sum()
```

* [ ] Analyze missing values

```python
(df.isnull().sum()/len(df))*100
```

Decision Rules:

* Numerical → Mean / Median
* Categorical → Mode / Missing Label

---

* [ ] Check feature distributions

```python
df.describe()
df.skew()
```

Skew Rules:

```text
-1 to 1 → Safe
>1 → Log transform candidate
<-1 → Reflect + Transform
```

---

* [ ] Detect outliers

```text
IQR Method:
Lower = Q1 - 1.5*IQR
Upper = Q3 + 1.5*IQR
```

---

* [ ] Correlation analysis

```python
df.corr()
```

Rules:

```text
Correlation > 0.85 → Potential multicollinearity
```

---

* [ ] Target balance check

```python
df[target].value_counts(normalize=True)
```

Rules:

```text
60:40 → Usually okay
80:20 → Watch F1
90:10+ → Use balancing methods
```

---

# ⚙️ Phase 2: Preprocessing & Feature Engineering

*Split first. Transform later.*

## ✅ Checklist

* [ ] Data split strategy chosen

```python
from sklearn.model_selection import train_test_split
```

Choose:

* Standard split
* Stratified split
* TimeSeries split

---

* [ ] Prevent data leakage

```python
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
```

Rule:

```text
Never fit on full dataset
```

---

* [ ] Encoding selected

| Scenario         | Method          |
| ---------------- | --------------- |
| Low cardinality  | OneHot          |
| High cardinality | Target Encoding |
| Deep learning    | Embeddings      |
| Ordinal          | Mapping         |

---

* [ ] Scaling selected

| Scenario        | Method         |
| --------------- | -------------- |
| Normal data     | StandardScaler |
| Neural networks | MinMaxScaler   |
| Outliers        | RobustScaler   |

---

* [ ] Feature selection

Remove:

* High correlation
* Low variance
* IDs
* Irrelevant columns
* Constants

---

# 🏗️ Phase 3: Model Building & Compilation

*Build baseline first.*

## ✅ Checklist

* [ ] Create dummy baseline

```python
from sklearn.dummy import DummyClassifier, DummyRegressor
```

---

* [ ] Fix random seeds

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

---

* [ ] Verify input/output shapes

| Task        | Output      |
| ----------- | ----------- |
| Regression  | 1 (Linear)  |
| Binary      | 1 (Sigmoid) |
| Multi-Class | N (Softmax) |

---

* [ ] Initialize weights

| Activation | Initialization |
| ---------- | -------------- |
| ReLU       | He             |
| LeakyReLU  | He             |
| Sigmoid    | Xavier         |
| Tanh       | Xavier         |

---

* [ ] Add regularization

* Dropout

* BatchNorm

* Weight Decay

---

* [ ] Select loss function

```python
# Classification
BCEWithLogitsLoss()
CrossEntropyLoss()

# Regression
MSELoss()
HuberLoss()
```

---

* [ ] Select optimizer

```python
torch.optim.AdamW(model.parameters(), lr=3e-4)
```

---

# 📊 Phase 4: Training Diagnostics

*Watch learning behavior.*

## Validation Strategy

Choose:

* KFold
* StratifiedKFold
* TimeSeriesSplit

---

## Loss Diagnostics

| Pattern       | Meaning            | Action             |
| ------------- | ------------------ | ------------------ |
| Train ↓ Val ↑ | Overfitting        | Add regularization |
| Both high     | Underfitting       | Bigger model       |
| Sawtooth      | LR too high        | Lower LR           |
| Flatline      | No learning        | Check LR/init      |
| Sudden spikes | Gradient explosion | Clip gradients     |

---

# 🎛️ Phase 4.5: Hyperparameter Tuning

*Optimize systematically.*

---

## Tree Models

Tune:

| Parameter         | Meaning         |
| ----------------- | --------------- |
| n_estimators      | Number of trees |
| max_depth         | Tree complexity |
| min_samples_split | Split control   |
| min_samples_leaf  | Leaf smoothness |
| max_features      | Randomness      |

Example:

```python
param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5, 10]
}
```

---

## Neural Networks

Tune:

| Parameter     | Range       |
| ------------- | ----------- |
| Learning Rate | 1e-5 → 1e-2 |
| Batch Size    | 16 → 256    |
| Dropout       | 0.2 → 0.5   |
| Hidden Units  | 32 → 1024   |
| Layers        | 1 → 10      |
| Weight Decay  | 1e-6 → 1e-2 |

---

## Search Methods

| Method        | Use Case            |
| ------------- | ------------------- |
| Grid Search   | Small space         |
| Random Search | Large space         |
| Bayesian      | Expensive models    |
| Hyperband     | Deep learning       |
| Optuna        | General best choice |

---

### Grid Search Example

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="f1"
)

grid.fit(X_train, y_train)
```

---

### Tuning Rules

## Overfitting

```text
↓ max_depth
↑ dropout
↑ regularization
↓ learning rate
```

---

## Underfitting

```text
↑ layers
↑ hidden units
↑ max_depth
↓ dropout
↑ epochs
```

---

## Unstable Training

```text
↓ learning rate
↑ batch size
Add BatchNorm
Use gradient clipping
```

---

# 📈 Phase 5: Final Metric Interpretation

*Interpret model behavior.*

---

## Classification Metrics

| Metric    | Good              | Warning               |
| --------- | ----------------- | --------------------- |
| Accuracy  | High + F1 aligned | Misleading imbalance  |
| Precision | >0.80             | Too many FP           |
| Recall    | >0.80             | Missing positives     |
| F1        | Balanced          | Weak precision/recall |
| PR-AUC    | >0.70             | Weak minority         |
| ROC-AUC   | >0.80             | Poor separability     |

Quick Rules:

```text
High Accuracy + Low Recall = Minority ignored
High Precision + Low Recall = Conservative
Low Precision + High Recall = Aggressive
High Both = Strong model
```

---

## Regression Metrics

| Metric      | Good         | Warning             |
| ----------- | ------------ | ------------------- |
| MAE         | Low          | High                |
| RMSE        | Close to MAE | Outlier mistakes    |
| R²          | >0.80        | Weak fit            |
| Adjusted R² | Close to R²  | Irrelevant features |

Quick Rules:

```text
RMSE ≈ MAE → Stable
RMSE >> MAE → Outlier issues
R² < 0 → Worse than baseline
```

---

## Loss Relationship Check

| Scenario                     | Meaning                    |
| ---------------------------- | -------------------------- |
| Train ↓ Val ↓                | Healthy                    |
| Train ↓ Val ↑                | Overfitting                |
| Both high                    | Underfitting               |
| Val unstable                 | LR too high                |
| Low train loss + bad metrics | Leakage or threshold issue |

---

# 🚀 Phase 6: Production Readiness

## Checklist

* [ ] Model saved
* [ ] Artifacts saved
* [ ] Logging active
* [ ] Inference tested
* [ ] Input schema validated
* [ ] Monitoring setup ready

Saved formats:

```text
.pkl → Scikit-learn
.pt → PyTorch
.onnx → Cross-platform
.tflite → Edge/Mobile
```

Optimization:

* Quantization
* Pruning
* ONNX conversion

---

# 📂 Recommended Project Structure

```text
project/
│── artifacts/
│   │── scaler.pkl
│   │── encoder.pkl
│   │── pipeline.pkl
│
│── data/
│   │── raw/
│   │── processed/
│
│── logs/
│   │── fit/
│
│── models/
│   │── model.pkl
│   │── best_model.pkl
│   │── model_metadata.json
│
│── notebooks/
│   │── experimentation.ipynb
│
│── src/
│   │── data_ingestion.py
│   │── preprocessing.py
│   │── train.py
│   │── tune.py
│   │── evaluate.py
│   │── predict.py
│
│── README.md
│── app.py
│── requirements.txt
│── runtime.txt
```

---

# 📝 Notes

This workflow ensures:

* Proper EDA
* Zero leakage
* Better preprocessing decisions
* Better hyperparameter tuning
* Better debugging
* Better reproducibility
* Better deployment readiness