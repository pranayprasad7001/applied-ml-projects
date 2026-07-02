import datetime

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.models import Sequential


def run_training():
    print("Loading processed data...")
    X_train = pd.read_csv("../data/X_train_processed.csv")
    X_val = pd.read_csv("../data/X_val_processed.csv")
    X_test = pd.read_csv("../data/X_test_processed.csv")

    # Flatten y dataframes to Series
    y_train = pd.read_csv("../data/y_train.csv")["Exited"]
    y_val = pd.read_csv("../data/y_val.csv")["Exited"]
    y_test = pd.read_csv("../data/y_test.csv")["Exited"]

    print("Building ANN model...")
    model = Sequential(
        [
            Input(shape=(X_train.shape[1],)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="roc_auc"),
        ],
    )

    log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    # Implement class weights to penalize missing actual churners
    class_weights = {0: 1.0, 1: 2.5}

    print("Training model...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    print("Saving trained model...")
    model.save("../models/ann_model.keras")

    print("\n--- Final Evaluation (Threshold = 0.45) ---")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.45).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    run_training()
