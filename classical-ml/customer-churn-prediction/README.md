# Customer Churn Prediction using Artificial Neural Networks (ANN)

## Overview

This project predicts whether a bank customer is likely to churn based on demographic and financial attributes using an Artificial Neural Network (ANN). The model is trained using TensorFlow/Keras and deployed with Streamlit for interactive inference.

The project demonstrates an end-to-end machine learning workflow including:

* Data preprocessing
* Feature engineering
* Model training
* Model evaluation
* TensorBoard visualization
* Model artifact saving
* Streamlit deployment

## Live Demo

Streamlit App: https://bank-customer-churn-predictor-ann.streamlit.app

## Project Source

GitHub Repository:
https://github.com/pranayprasad7001/applied-ml-projects/tree/main/classical-ml/customer-churn-prediction

---

## Problem Statement

Customer churn is one of the major challenges in the banking sector. Predicting churn helps businesses:

* Improve customer retention
* Reduce revenue loss
* Build targeted retention strategies

This model predicts the probability of a customer leaving the bank.

---

## Dataset Features

The model uses the following input features:

| Feature         | Description                  |
| --------------- | ---------------------------- |
| Geography       | Customer country             |
| CreditScore     | Credit score of customer     |
| Gender          | Customer gender              |
| Age             | Customer age                 |
| Tenure          | Number of years with bank    |
| Balance         | Account balance              |
| NumOfProducts   | Number of bank products used |
| HasCrCard       | Credit card ownership        |
| IsActiveMember  | Customer activity status     |
| EstimatedSalary | Customer salary              |

Target variable:

* **Exited** → Customer churn status (0 = No, 1 = Yes)

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
├── models/
│   └── ann_model.keras
│
├── notebooks/
│   ├── training.ipynb
│   └── prediction.ipynb
│
├── logs/
│   └── TensorBoard logs
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

## Model Architecture

The ANN architecture:

* Input Layer
* Dense Layer (64 neurons, ReLU)
* Dense Layer (32 neurons, ReLU)
* Output Layer (1 neuron, Sigmoid)

Loss Function:

* Binary Crossentropy

Optimizer:

* Adam

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1-score

---

## Preprocessing Pipeline

The preprocessing steps:

1. Label Encoding (`Gender`)
2. One-Hot Encoding (`Geography`)
3. Train-Test Split
4. Feature Scaling (excluding one-hot encoded columns)

Saved preprocessing artifacts:

* `label_encoder.pkl`
* `onehot_encoder.pkl`
* `standard_scaler.pkl`

---

## Threshold Optimization

Threshold tuning was performed to optimize the precision-recall tradeoff for churn prediction.

Instead of using the default classification threshold of **0.5**, the threshold was lowered to **0.4** to improve the model's ability to detect churned customers.

Results:

* **Recall improved from 49% → 53%**
* **F1-score improved**
* Better sensitivity for identifying potential churn customers

This adjustment helps reduce false negatives, which is important in churn prediction since missing a potential churn customer can result in revenue loss.

---

## Model Performance

Final model performance on test data:

- Accuracy: **85%**
- Precision (Churn): **67%**
- Recall (Churn): **53%**
- F1-score (Churn): **59%**

Confusion Matrix:

|               | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 1486 | 107 |
| Actual Churn | 191 | 216 |

Observations:

- The model performs strongly in identifying non-churn customers.
- Threshold tuning improved churn recall from **49% → 53%**, increasing the model’s sensitivity toward churned customers.
- This helps reduce false negatives, which is critical for customer retention strategies.

---

## TensorBoard Integration

TensorBoard was used for:

* Monitoring training loss
* Monitoring validation loss
* Accuracy tracking
* Learning rate visualization

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

Start the Streamlit app:

```bash
streamlit run app.py
```

---

## Streamlit Features

The deployed application allows users to:

* Select customer geography
* Select gender
* Enter financial details
* Predict churn probability
* View churn decision instantly

---

## Sample Prediction Workflow

User Input → Preprocessing → Feature Scaling → ANN Model → Churn Probability

---

## Future Improvements

* Hyperparameter tuning
* Dropout regularization
* Batch normalization
* Learning rate scheduling
* Docker deployment

---

## Author

**Pranay Prasad**  
MTech in AI & ML  
Aspiring Machine Learning Engineer