import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def engineer_features(df):
    """Generates new business-logic features to aid model convergence."""
    df_engineered = df.copy()
    df_engineered["BalanceSalaryRatio"] = (
        df_engineered["Balance"] / df_engineered["EstimatedSalary"]
    )
    df_engineered["TenureByAge"] = df_engineered["Tenure"] / df_engineered["Age"]
    bins = [18, 30, 40, 50, 60, 100]
    labels = ["Young_Adult", "Adult", "Middle_Age", "Senior", "Elderly"]

    df_engineered["AgeGroup"] = pd.cut(
        df_engineered["Age"], bins=bins, labels=labels, right=False
    )
    df_engineered["CreditScoreGivenAge"] = (
        df_engineered["CreditScore"] / df_engineered["Age"]
    )
    df_engineered["Products_Active_Interaction"] = (
        df_engineered["NumOfProducts"] * df_engineered["IsActiveMember"]
    )

    return df_engineered


def run_feature_engineering(data_path):
    print("Loading raw data...")
    dataset = pd.read_csv(data_path)
    dataset = dataset.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

    print("Engineering new features...")
    dataset = engineer_features(dataset)

    X = dataset.drop("Exited", axis=1)
    y = dataset["Exited"]

    print("Encoding categorical variables...")
    le_gender = LabelEncoder()
    X["Gender"] = le_gender.fit_transform(X["Gender"])

    ct = ColumnTransformer(
        transformers=[
            ("encode", OneHotEncoder(sparse_output=False), ["Geography", "AgeGroup"])
        ],
        remainder="passthrough",
    )

    cols = ct.fit(X).get_feature_names_out()
    X_encoded = pd.DataFrame(
        ct.transform(X), columns=[col.split("__")[-1] for col in cols]
    )

    print("Splitting data (64% Train, 16% Val, 20% Test)...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full,
    )

    print("Scaling numerical features...")
    sc = StandardScaler()
    X_train_scaled = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(sc.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(sc.transform(X_test), columns=X_test.columns)

    print("Saving processed datasets...")
    X_train_scaled.to_csv("../data/X_train_processed.csv", index=False)
    X_val_scaled.to_csv("../data/X_val_processed.csv", index=False)
    X_test_scaled.to_csv("../data/X_test_processed.csv", index=False)

    y_train.to_csv("../data/y_train.csv", index=False)
    y_val.to_csv("../data/y_val.csv", index=False)
    y_test.to_csv("../data/y_test.csv", index=False)

    print("Saving artifacts...")
    with open("../artifacts/label_encoder.pkl", "wb") as f:
        pickle.dump(le_gender, f)
    with open("../artifacts/onehot_encoder.pkl", "wb") as f:
        pickle.dump(ct, f)
    with open("../artifacts/standard_scaler.pkl", "wb") as f:
        pickle.dump(sc, f)

    print("Feature engineering complete!")


if __name__ == "__main__":
    run_feature_engineering("../data/churn-modelling.csv")
