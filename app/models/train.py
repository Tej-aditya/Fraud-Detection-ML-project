import os
import joblib
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

from data.dataset_loader import load_data
from app.features.build_features import preprocess

# =====================
# CONFIG
# =====================
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.pkl")


def train():
    print("🚀 Loading data...")
    df = load_data()

    print("⚙️ Building features...")
    X, y, feature_names = preprocess(df, return_feature_names=True)

    print("🧾 Features used for training:")
    for i, f in enumerate(feature_names):
        print(f"{i}: {f}")

    print("🔀 Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("🤖 Training XGBoost Classifier...")
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="aucpr"
    )

    model.fit(X_train_scaled, y_train)

    print("📊 Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n=== Classification Report (threshold = 0.5) ===")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_proba)
    print("ROC AUC:", round(roc_auc, 4))

    # =====================
    # SAVE ARTIFACTS
    # =====================
    print("💾 Saving model artifacts...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_names, FEATURES_PATH)

    print("✅ Training complete & artifacts saved")
    print(f"📦 Model   : {MODEL_PATH}")
    print(f"📦 Scaler  : {SCALER_PATH}")
    print(f"📦 Features: {FEATURES_PATH}")


if __name__ == "__main__":
    train()