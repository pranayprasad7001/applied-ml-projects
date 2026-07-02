from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# --- Path Configuration ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def run_evaluation():
    print("Loading model and test data...")

    # Load Model (wrapping in str() for broader Keras version compatibility)
    model = load_model(str(MODELS_DIR / "ann_model.keras"))

    # Load Processed Test Data
    X_test = pd.read_csv(DATA_DIR / "X_test_processed.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["Exited"]

    # --- Base Keras Evaluation ---
    print("\n--- Base Keras Metrics ---")
    metrics = model.evaluate(X_test, y_test, verbose=0)
    for name, val in zip(model.metrics_names, metrics):
        print(f"{name.capitalize()}: {val:.4f}")

    # --- Business-Optimized Evaluation ---
    threshold = 0.45
    print(f"\n--- Evaluation at Optimized Threshold: {threshold} ---")

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > threshold).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    run_evaluation()
