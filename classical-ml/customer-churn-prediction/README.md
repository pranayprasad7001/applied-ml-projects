# Customer Churn Prediction using Artificial Neural Networks (ANN)

## Overview

This project predicts whether a bank customer is likely to churn based on demographic and financial attributes using an Artificial Neural Network (ANN). The model is built using **TensorFlow/Keras** and deployed with **Streamlit** for interactive inference.

It demonstrates a complete end-to-end machine learning workflow, including:

* Data preprocessing
* Feature engineering
* Model training
* Model evaluation
* Threshold optimization
* TensorBoard visualization
* Model artifact saving
* Streamlit deployment

---

## Live Demo

**Streamlit App:**
https://bank-customer-churn-predictor-ann.streamlit.app

---

## Project Source

**GitHub Repository:**
https://github.com/pranayprasad7001/applied-ml-projects/tree/main/classical-ml/customer-churn-prediction

---

## Problem Statement

Customer churn is a critical challenge in the banking industry. Losing customers directly impacts revenue and long-term profitability.

Predicting churn helps businesses:

* Improve customer retention
* Reduce revenue loss
* Create targeted retention strategies
* Increase customer lifetime value

This model predicts the probability of a customer leaving the bank.

---

## Dataset Features

The model uses the following input features:

| Feature         | Description              |
| --------------- | ------------------------ |
| CreditScore     | Customer credit score    |
| Geography       | Customer location        |
| Gender          | Customer gender          |
| Age             | Customer age             |
| Tenure          | Years with the bank      |
| Balance         | Account balance          |
| NumOfProducts   | Number of products used  |
| HasCrCard       | Credit card ownership    |
| IsActiveMember  | Active membership status |
| EstimatedSalary | Estimated yearly salary  |

**Target Variable:**

* **Exited** → Customer churn status

  * `0` = No churn
  * `1` = Churn

---

## Project Structure

```text
customer-churn-prediction/
│── app.py
│── requirements.txt
│── README.md
│
├── artifacts/
│   ├── label_encoder.pkl
│   ├── onehot_encoder.pkl
│   ├── standard_scaler.pkl
│
├── data/
│   ├── X_train_processed.csv
│   ├── X_val_processed.csv
│   ├── X_test_processed.csv
│   ├── y_train.csv
│   ├── y_val.csv
│   └── y_test.csv
│
├── models/
│   └── ann_model.keras
│
├── logs/
│   └── fit/
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_train.ipynb
│   ├── 04_evaluate.ipynb
│   └── 05_predict.ipynb
│
├── scripts/
│   ├── 02_feature_engineering.py
│   ├── 03_training.py
│   ├── 04_evaluate.py
│   └── 05_predict.py
```

---

## Tech Stack

* Python
* Pandas
* TensorFlow / Keras
* Scikit-learn
* Streamlit
* TensorBoard
* Pickle

---

## Feature Engineering

Custom engineered features were added to improve model learning:

* **BalanceSalaryRatio** → Balance / Estimated Salary
* **TenureByAge** → Tenure / Age
* **AgeGroup** → Age bucket categorization
* **CreditScoreGivenAge** → Credit Score / Age
* **Products_Active_Interaction** → Product count × Active member

These features improve business context representation and model convergence.

---

## Model Architecture

The ANN architecture:

* Input Layer
* Dense Layer (**64 neurons**, ReLU)
* Dense Layer (**32 neurons**, ReLU)
* Output Layer (**1 neuron**, Sigmoid)

### Loss Function

* Binary Crossentropy

### Optimizer

* Adam

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* ROC-AUC
* F1-score

---

## Preprocessing Pipeline

The preprocessing workflow:

1. Drop irrelevant columns (`RowNumber`, `CustomerId`, `Surname`)
2. Apply custom feature engineering
3. Label Encoding (`Gender`)
4. One-Hot Encoding (`Geography`, `AgeGroup`)
5. Train-Test Split (**80-20**)
6. Train-Validation Split (**80-20 of training set**)
7. Feature Scaling using `StandardScaler`

Saved preprocessing artifacts:

* `label_encoder.pkl`
* `onehot_encoder.pkl`
* `standard_scaler.pkl`

---

## Threshold Optimization

Instead of using the default threshold (**0.50**), the decision threshold was optimized to **0.45**.

This improves the precision-recall tradeoff and helps better identify churn-prone customers.

Why?

In churn prediction:

* Missing an actual churn customer (**False Negative**) is expensive.
* Lowering the threshold improves detection sensitivity.

---

## Model Performance

Final model performance on test data:

### Evaluation at Threshold = **0.45**

| Metric            |   Score |
| ----------------- | ------: |
| Accuracy          | **83%** |
| Precision (Churn) | **57%** |
| Recall (Churn)    | **67%** |
| F1-score (Churn)  | **62%** |

### Confusion Matrix

|                 | Predicted No Churn | Predicted Churn |
| --------------- | -----------------: | --------------: |
| Actual No Churn |               1390 |             203 |
| Actual Churn    |                135 |             272 |

### Observations

* Strong performance in identifying non-churn customers.
* Balanced churn detection with optimized thresholding.
* Reduced false positives significantly.
* Improved F1-score for churn class.

---

## TensorBoard Integration

TensorBoard was used for:

* Monitoring training loss
* Monitoring validation loss
* Accuracy tracking
* Precision & Recall tracking
* Learning rate scheduling

Run TensorBoard:

```bash
tensorboard --logdir logs/fit
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/pranayprasad7001/applied-ml-projects.git
cd applied-ml-projects/classical-ml/customer-churn-prediction
```

Create virtual environment:

```bash
conda create -n churn_modelling_venv python=3.11
conda activate churn_modelling_venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## Streamlit Features

The deployed app allows users to:

* Input customer details
* Select geography and gender
* Enter financial details
* Predict churn probability
* Get retention recommendations instantly

---

## Prediction Workflow

```text
Raw Input
   ↓
Feature Engineering
   ↓
Encoding
   ↓
Feature Scaling
   ↓
ANN Model
   ↓
Probability Prediction
   ↓
Threshold Decision (0.45)
```

---

## Future Improvements

* Hyperparameter tuning
* Dropout regularization
* Batch normalization
* Advanced feature selection
* Model explainability using SHAP
* Docker deployment
* FastAPI integration

---

## Author

**Pranay Prasad**
MTech in AI & ML
Aspiring Machine Learning Engineer