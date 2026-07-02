import pickle
from pathlib import Path

import pandas as pd
from tensorflow.keras.models import load_model

# --- Path Configuration ---
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"


class ChurnPredictor:
    def __init__(self):
        """Loads all artifacts and the model into memory upon initialization."""
        print("Initializing Predictor and loading artifacts...")

        with open(ARTIFACTS_DIR / "label_encoder.pkl", "rb") as f:
            self.le_gender = pickle.load(f)
        with open(ARTIFACTS_DIR / "onehot_encoder.pkl", "rb") as f:
            self.ct_geo = pickle.load(f)
        with open(ARTIFACTS_DIR / "standard_scaler.pkl", "rb") as f:
            self.sc = pickle.load(f)

        self.model = load_model(str(MODELS_DIR / "ann_model.keras"))
        self.threshold = 0.45

    def _engineer_features(self, df):
        """Applies the exact same feature engineering logic used during training."""
        df_eng = df.copy()

        # Avoid division by zero in edge cases for production data
        estimated_salary = df_eng["EstimatedSalary"].replace(0, 0.01)
        age = df_eng["Age"].replace(0, 1)

        df_eng["BalanceSalaryRatio"] = df_eng["Balance"] / estimated_salary
        df_eng["TenureByAge"] = df_eng["Tenure"] / age

        bins = [18, 30, 40, 50, 60, 100]
        labels = ["Young_Adult", "Adult", "Middle_Age", "Senior", "Elderly"]
        df_eng["AgeGroup"] = pd.cut(
            df_eng["Age"], bins=bins, labels=labels, right=False
        )

        df_eng["CreditScoreGivenAge"] = df_eng["CreditScore"] / age
        df_eng["Products_Active_Interaction"] = (
            df_eng["NumOfProducts"] * df_eng["IsActiveMember"]
        )

        return df_eng

    def predict(self, input_data: dict) -> dict:
        """Processes raw input data and returns a churn prediction."""
        df = pd.DataFrame([input_data])

        # 1. Feature Engineering
        df = self._engineer_features(df)

        # 2. Encode Gender
        df["Gender"] = self.le_gender.transform(df["Gender"])

        # 3. Encode Geography and AgeGroup
        cols = self.ct_geo.get_feature_names_out()
        df_encoded = pd.DataFrame(
            self.ct_geo.transform(df), columns=[col.split("__")[-1] for col in cols]
        )

        # 4. Scale Features
        df_encoded = pd.DataFrame(
            self.sc.transform(df_encoded), columns=df_encoded.columns
        )

        # 5. Predict
        probability = float(self.model.predict(df_encoded, verbose=0)[0][0])

        return {
            "churn_probability": round(probability, 4),
            "will_churn": bool(probability > self.threshold),
            "applied_threshold": self.threshold,
        }


# --- Example Usage for Testing ---
if __name__ == "__main__":
    predictor = ChurnPredictor()

    # Simulate a raw JSON payload from a frontend application
    raw_customer_data = {
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0.00,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 101348.88,
    }

    result = predictor.predict(raw_customer_data)
    print(f"\nInference Result:\n{result}")
