import numpy as np
import pandas as pd

# =====================
# CONFIG
# =====================
KEEP_COLS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud"
]


# =====================
# TRAINING PREPROCESS
# =====================
def preprocess(df, return_feature_names=False):
    # Select required columns
    df = df.loc[:, KEEP_COLS]

    # Target
    y = df["isFraud"].values

    # Features
    X = df.drop(columns=["isFraud"])

    # One-hot encode transaction type
    X = pd.get_dummies(X, columns=["type"], drop_first=True)

    # Feature engineering
    X["amount_log"] = np.log1p(X["amount"])
    X["balance_delta_org"] = X["oldbalanceOrg"] - X["newbalanceOrig"]
    X["balance_delta_dest"] = X["newbalanceDest"] - X["oldbalanceDest"]

    feature_names = X.columns.tolist()

    X = X.values.astype("float32")

    if return_feature_names:
        return X, y, feature_names

    return X, y


# =====================
# INFERENCE PREPROCESS
# =====================
def preprocess_single(payload: dict, feature_names: list):
    df = pd.DataFrame([payload])

    # One-hot encoding
    df = pd.get_dummies(df, columns=["type"], drop_first=True)

    # Feature engineering
    df["amount_log"] = np.log1p(df["amount"])
    df["balance_delta_org"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_delta_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # Ensure column alignment
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    return df.values.astype("float32")