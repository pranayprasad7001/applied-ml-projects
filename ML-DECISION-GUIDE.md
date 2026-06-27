# 🧠 Machine Learning Solution Guide (What to Use in Different Scenarios)

A quick decision framework for choosing preprocessing, models, metrics, regularization, and debugging strategies based on dataset behavior.

---

# 📊 1. Missing Values

## Numerical Features

| Scenario | Use |
|---|---|
| Data is normally distributed | Mean Imputation |
| Data is skewed | Median Imputation |
| Missing values carry meaning | Create separate "Missing Flag" column |
| Sequential/Time-series data | Forward Fill / Backward Fill |

---

## Categorical Features

| Scenario | Use |
|---|---|
| Few missing values | Mode Imputation |
| Missing has meaning | `"Unknown"` / `"Missing"` label |
| Too many missing values (>50%) | Drop column |

---

# 🔍 2. Feature Distribution

## If feature is highly skewed:

| Skewness | Action |
|---|---|
| Between -1 and 1 | Usually okay |
| >1 | Log Transform / Box-Cox |
| <-1 | Reflect + Log Transform |

Use:

```python
np.log1p(feature)
```

Best for:

- Income
- Prices
- Population
- Counts

---

# ⚡ 3. Feature Scaling

| Scenario | Use |
|---|---|
| Normal distribution | StandardScaler |
| Data bounded or neural networks | MinMaxScaler |
| Heavy outliers exist | RobustScaler |
| Sparse matrix data | MaxAbsScaler |

---

# 🏷️ 4. Encoding Categorical Variables

| Scenario | Use |
|---|---|
| No order, low cardinality | OneHotEncoder |
| Ordered categories | Label Encoding / Mapping |
| High cardinality | Target Encoding |
| Deep learning models | Embeddings |

---

# ⚖️ 5. Imbalanced Dataset

Check:

```python
df[target].value_counts(normalize=True)
```

| Ratio | Action |
|---|---|
| 60:40 | Usually okay |
| 80:20 | Monitor F1 |
| 90:10+ | Use balancing techniques |

Solutions:

| Problem | Use |
|---|---|
| Minority too small | SMOTE |
| Majority too dominant | Undersampling |
| Want weighted loss | class_weight="balanced" |
| Deep learning | Focal Loss |

---

# 📉 6. Outliers

Detection:

- Boxplot
- IQR
- Z-score

| Scenario | Use |
|---|---|
| Legitimate rare values | Keep them |
| Data entry errors | Remove |
| Affects scaling badly | RobustScaler |
| Extreme skew | Winsorization |

IQR rule:

```text
Q1 - 1.5*IQR
Q3 + 1.5*IQR
```

---

# 🤖 7. Model Selection Guide

# Classification

| Scenario | Best Option |
|---|---|
| Small structured data | Logistic Regression |
| Non-linear structured data | Random Forest |
| High-dimensional | XGBoost / LightGBM |
| Large tabular dataset | CatBoost |
| Images | CNN |
| Text | Transformer / LSTM |
| Sequential data | LSTM / GRU |
| Graph data | GNN |

---

# Regression

| Scenario | Best Option |
|---|---|
| Linear relationship | Linear Regression |
| Non-linear relationship | Random Forest Regressor |
| Complex tabular | XGBoost Regressor |
| High outlier sensitivity | Huber Regression |

---

# Clustering

| Scenario | Use |
|---|---|
| Circular clusters | KMeans |
| Irregular clusters | DBSCAN |
| Hierarchical relations | Agglomerative |
| Probabilistic clusters | Gaussian Mixture |

---

# 🎯 8. Loss Function Selection

# Classification

| Scenario | Use |
|---|---|
| Binary classification | BCEWithLogitsLoss |
| Multi-class | CrossEntropyLoss |
| Imbalanced binary | Focal Loss |
| Multi-label | BCELoss |

---

# Regression

| Scenario | Use |
|---|---|
| Standard regression | MSELoss |
| Outlier resistant | HuberLoss |
| Absolute error focus | MAELoss |

---

# 🔥 9. Activation Functions

| Scenario | Use |
|---|---|
| Hidden layers default | ReLU |
| Dead neurons problem | LeakyReLU |
| Smooth gradients needed | GELU |
| Binary output | Sigmoid |
| Multi-class output | Softmax |
| Regression output | Linear |

---

# 🧱 10. Weight Initialization

| Activation | Initialization |
|---|---|
| ReLU / LeakyReLU | He Initialization |
| Sigmoid / Tanh | Xavier Initialization |

---

# 🛡️ 11. Regularization Decision Guide

| Problem | Solution |
|---|---|
| Overfitting | Dropout |
| Weights too large | L2 Regularization |
| Feature selection needed | L1 Regularization |
| Training unstable | BatchNorm |
| Validation loss fluctuating | ReduceLROnPlateau |

---

# 📈 12. Metric Selection

# Classification

| Goal | Metric |
|---|---|
| Balanced classes | Accuracy |
| Reduce false positives | Precision |
| Reduce false negatives | Recall |
| Imbalanced dataset | F1 Score |
| Minority class focus | PR-AUC |
| Threshold independent | ROC-AUC |

---

# Regression

| Goal | Metric |
|---|---|
| Average error understanding | MAE |
| Penalize big mistakes | RMSE |
| Overall fit quality | R² |

---

# 📉 13. Training Curve Diagnosis

| Pattern | Meaning | Solution |
|---|---|---|
| Train ↓ Val ↑ | Overfitting | More regularization |
| Train ↑ Val ↑ | Underfitting | Bigger model |
| Both flat | Learning stalled | Increase LR |
| Violent oscillation | LR too high | Lower LR |
| Sudden spike | Gradient explosion | Clip gradients |

---

# 🚀 14. Learning Rate Decision

| Scenario | Use |
|---|---|
| Default starting point | 1e-3 |
| Transformer | 3e-4 |
| Fine-tuning | 1e-5 |
| Unstable training | Lower LR |
| Slow learning | Increase LR |

---

# ⏹️ 15. Early Stopping

| Scenario | Patience |
|---|---|
| Small dataset | 5–10 |
| Large dataset | 10–20 |
| Noisy validation | 15–25 |

Monitor:

```python
val_loss
```

---

# 🏭 16. Deployment Optimization

| Scenario | Use |
|---|---|
| Faster inference | Quantization |
| Smaller model | Pruning |
| Cross-platform | ONNX |
| Mobile deployment | TensorFlow Lite |
| Edge devices | TensorRT |

---

# 🧪 17. Data Leakage Prevention

Always:

✅ Split first  
✅ Fit transformers on train only  
✅ Transform val/test later  

Wrong:

```python
scaler.fit(df)
```

Correct:

```python
scaler.fit(X_train)
```

---

# 🧠 Quick Rule Summary

```text
Skewed Data → Median + Log Transform
Outliers → RobustScaler
Imbalanced → F1 + SMOTE
Overfitting → Dropout + L2
Underfitting → Bigger Model
High Cardinality → Target Encoding
Images → CNN
Text → Transformer
Sequential → LSTM
Tabular → XGBoost / CatBoost
Binary Classification → BCEWithLogitsLoss
Regression + Outliers → HuberLoss
```