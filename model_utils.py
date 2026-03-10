from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "app" / "models" / "model"

def load_model():
    model_path = MODEL_DIR / "fraud_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Model or scaler file not found. Train the model first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler
