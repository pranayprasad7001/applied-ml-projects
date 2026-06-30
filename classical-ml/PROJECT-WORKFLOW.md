# 🚀 Machine Learning Project Workflow

* **Project Name:** [Insert Project Name]
* **Objective:** [Classification / Regression / Forecasting / Segmentation]
* **Target Variable:** `[Target Column Name]`

---

# 📊 Phase 1: Data Understanding & EDA

*Understand the dataset before transformations.*

## ✅ Checklist

* [ ] Verify source integrity
* [ ] Remove duplicates

```python
df.duplicated().sum()
```

* [ ] Missing value analysis

```python
(df.isnull().sum() / len(df)) * 100
```

Decision:

* Numerical → Mean / Median
* Categorical → Mode / Missing Label

---

* [ ] Distribution analysis

```python
df.describe()
df.skew()
```

Rules:

```text
-1 to 1 → Safe
>1 → Transformation candidate
<-1 → Reflect + Transform
```

---

* [ ] Outlier detection

```text
Lower = Q1 - 1.5*IQR
Upper = Q3 + 1.5*IQR
```

---

* [ ] Correlation analysis

```python
df.corr()
```

Rule:

```text
>0.85 → Consider removing one feature
```

---

* [ ] Target balance check

```python
df[target].value_counts(normalize=True)
```

Rule:

```text
90:10+ → Use imbalance strategies
```

---

# ⚙️ Phase 2: Preprocessing & Feature Engineering

*Split first. Transform second.*

## ✅ Checklist

* [ ] Select split strategy:

* Standard Split

* Stratified Split

* TimeSeriesSplit

---

* [ ] Prevent leakage

```python
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
```

Rule:

```text
Never fit preprocessing on full data
```

---

* [ ] Encoding

| Scenario         | Method          |
| ---------------- | --------------- |
| Low cardinality  | OneHot          |
| High cardinality | Target Encoding |
| Ordered          | Mapping         |
| Deep Learning    | Embeddings      |

---

* [ ] Scaling

| Scenario            | Method         |
| ------------------- | -------------- |
| Normal distribution | StandardScaler |
| Neural Networks     | MinMaxScaler   |
| Heavy outliers      | RobustScaler   |

---

* [ ] Feature selection

Remove:

* IDs
* Constants
* High correlation
* Low variance
* Irrelevant columns

---

# 🏗️ Phase 3: Model Building & Compilation

*Build baseline first.*

## ✅ Checklist

* [ ] Baseline model

```python
from sklearn.dummy import DummyClassifier, DummyRegressor
```

---

* [ ] Reproducibility

```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

---

* [ ] Verify output layer

| Task        | Output  |
| ----------- | ------- |
| Regression  | Linear  |
| Binary      | Sigmoid |
| Multi-Class | Softmax |

---

* [ ] Weight initialization

| Activation | Init   |
| ---------- | ------ |
| ReLU       | He     |
| LeakyReLU  | He     |
| Sigmoid    | Xavier |
| Tanh       | Xavier |

---

* [ ] Regularization

* Dropout

* BatchNorm

* Weight Decay

---

* [ ] Loss function

```python
# Classification
BCEWithLogitsLoss()
CrossEntropyLoss()

# Regression
MSELoss()
HuberLoss()
```

---

* [ ] Optimizer

```python
torch.optim.AdamW(model.parameters(), lr=3e-4)
```

---

# 📊 Phase 4: Training Diagnostics

*Monitor training behavior.*

## Validation Strategy

Choose:

* KFold
* StratifiedKFold
* TimeSeriesSplit

---

## Loss Diagnostics

| Pattern       | Meaning            | Action            |
| ------------- | ------------------ | ----------------- |
| Train ↓ Val ↑ | Overfitting        | Regularize more   |
| Both high     | Underfitting       | Increase capacity |
| Sawtooth      | LR too high        | Reduce LR         |
| Flatline      | No learning        | Check init/LR     |
| Sudden spikes | Gradient explosion | Clip gradients    |

---

# 🎛️ Phase 4.5: Hyperparameter Tuning

*Tune model-specific parameters.*

---

# Classical ML

## Logistic Regression

Tune:

| Parameter | Meaning                 |
| --------- | ----------------------- |
| C         | Regularization strength |
| penalty   | L1 / L2                 |
| solver    | Optimization algorithm  |

Rules:

```text
Overfitting → decrease C
Underfitting → increase C
```

---

## Random Forest

Tune:

| Parameter         | Meaning         |
| ----------------- | --------------- |
| n_estimators      | Number of trees |
| max_depth         | Tree complexity |
| min_samples_split | Split threshold |
| min_samples_leaf  | Leaf smoothness |
| max_features      | Randomness      |

Rules:

```text
Overfitting:
↓ max_depth
↑ min_samples_leaf

Underfitting:
↑ max_depth
↑ n_estimators
```

---

## XGBoost / LightGBM / CatBoost

Tune:

| Parameter        | Meaning          |
| ---------------- | ---------------- |
| n_estimators     | Boosting rounds  |
| learning_rate    | Step size        |
| max_depth        | Complexity       |
| subsample        | Row sampling     |
| colsample_bytree | Feature sampling |
| gamma            | Split constraint |
| reg_alpha        | L1               |
| reg_lambda       | L2               |

Important interactions:

```text
Lower learning_rate → More n_estimators needed
Higher max_depth → More regularization needed
```

---

## SVM

Tune:

| Parameter | Meaning            |
| --------- | ------------------ |
| C         | Margin flexibility |
| kernel    | Decision boundary  |
| gamma     | Influence radius   |

Rules:

```text
High C = lower bias, higher variance
Low C = higher bias, lower variance
```

---

# Deep Learning

## Optimizer Parameters

| Parameter     | Typical Range        |
| ------------- | -------------------- |
| Learning Rate | 1e-5 → 1e-2          |
| Weight Decay  | 1e-6 → 1e-2          |
| Betas         | Default usually fine |

---

## Architecture Parameters

| Parameter    | Range                   |
| ------------ | ----------------------- |
| Layers       | 1 → 10                  |
| Hidden Units | 32 → 1024               |
| Dropout      | 0.2 → 0.5               |
| Activation   | ReLU / GELU / LeakyReLU |

---

## Scheduler Parameters

| Scheduler         | Tune             |
| ----------------- | ---------------- |
| ReduceLROnPlateau | patience, factor |
| StepLR            | step_size, gamma |
| CosineAnnealing   | T_max            |

---

## Early Stopping

Tune:

| Parameter | Meaning             |
| --------- | ------------------- |
| patience  | Stop tolerance      |
| min_delta | Minimum improvement |

---

# Search Strategies

| Method                | Best For                          |
| --------------------- | --------------------------------- |
| Grid Search           | Small search space                |
| Random Search         | Large search space                |
| Bayesian Optimization | Expensive models                  |
| Hyperband             | Early stopping based optimization |
| Optuna                | Best general-purpose tuning       |

---

## Grid Search Example

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

## Tuning Decision Rules

### Overfitting

```text
Reduce model complexity
Increase regularization
Increase dropout
Use early stopping
Increase min_samples_leaf
Reduce max_depth
```

---

### Underfitting

```text
Increase capacity
Reduce regularization
Increase features
Train longer
Improve architecture
```

---

### Unstable Training

```text
Lower LR
Increase batch size
Use BatchNorm
Use gradient clipping
```

---

# 📈 Phase 5: Final Metric Interpretation

*Interpret model behavior.*

---

## Classification Metrics

| Metric    | Good              | Warning                  |
| --------- | ----------------- | ------------------------ |
| Accuracy  | High + F1 aligned | Misleading in imbalance  |
| Precision | >0.80             | Too many false positives |
| Recall    | >0.80             | Missing positives        |
| F1        | Balanced P/R      | Weak balance             |
| PR-AUC    | >0.70             | Weak minority class      |
| ROC-AUC   | >0.80             | Poor separation          |

---

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
| RMSE        | Close to MAE | Outlier issues      |
| R²          | >0.80        | Weak fit            |
| Adjusted R² | Close to R²  | Irrelevant features |

---

Quick Rules:

```text
RMSE ≈ MAE → Stable
RMSE >> MAE → Outlier problems
R² < 0 → Worse than baseline
```

---

# 🚀 Phase 6: Production Readiness

## Checklist

* [ ] Save model
* [ ] Save artifacts
* [ ] Validate inputs
* [ ] Enable logging
* [ ] Test inference
* [ ] Setup monitoring

Formats:

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

* Better EDA
* Better preprocessing
* Better model selection
* Better hyperparameter tuning
* Better debugging
* Better reproducibility
* Better deployment