from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import numpy as np
import secrets
from fastapi import FastAPI
from app.api.admin import router as admin_router

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.models import Organization, APIKey, PredictionLog
from app.seed import seed_initial_data


# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "model"

MODEL_PATH = MODEL_DIR / "fraud_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURES_PATH = MODEL_DIR / "features.pkl"


# -------------------------------------------------
# Globals
# -------------------------------------------------
model = None
scaler = None
features = None


# -------------------------------------------------
# Lifespan (ONLY ONE)
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, features

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)

    seed_initial_data()

    print("[OK] App fully initialized")
    yield


# -------------------------------------------------
# App
# -------------------------------------------------
app = FastAPI(
    title="Fraud Detection SaaS API",
    version="1.0",
    lifespan=lifespan
)
app.include_router(admin_router)

# -------------------------------------------------
# DB Dependency
# -------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------------------------------------
# API Key Security
# -------------------------------------------------
def verify_api_key(
    x_api_key: str = Header(...),
    db: Session = Depends(get_db)
):
    key = db.query(APIKey).filter(
        APIKey.key == x_api_key,
        APIKey.active == True
    ).first()

    if not key:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    return key


# -------------------------------------------------
# Schemas
# -------------------------------------------------
class TransactionInput(BaseModel):
    step: int = 0
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    transaction_type: str
    transaction_hour: int
    transaction_day: int
    merchant_category: int
    user_transaction_count_24h: int
    user_avg_transaction_amount: float
    is_international: int
    distance_from_home: float
    device_change: int
    failed_login_attempts: int


class OrgCreate(BaseModel):
    name: str


class APIKeyCreate(BaseModel):
    org_id: int
    threshold: float = 0.5


# -------------------------------------------------
# Health
# -------------------------------------------------
@app.get("/")
def root():
    return {"status": "Fraud Detection SaaS running 🚀"}


@app.get("/admin/health/db")
def db_health(db: Session = Depends(get_db)):
    return {
        "organizations": db.query(Organization).count(),
        "api_keys": db.query(APIKey).count(),
        "prediction_logs": db.query(PredictionLog).count()
    }


# -------------------------------------------------
# Admin
# -------------------------------------------------
@app.post("/admin/org")
def create_org(payload: OrgCreate, db: Session = Depends(get_db)):
    org = Organization(name=payload.name)
    db.add(org)
    db.commit()
    db.refresh(org)
    return org


@app.post("/admin/apikey")
def create_api_key(payload: APIKeyCreate, db: Session = Depends(get_db)):
    org = db.query(Organization).filter(
        Organization.id == payload.org_id
    ).first()

    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    key_value = "sk_" + secrets.token_hex(16)

    api_key = APIKey(
        key=key_value,
        org_id=payload.org_id,
        threshold=payload.threshold
    )

    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    return {
        "api_key": api_key.key,
        "threshold": api_key.threshold
    }


# -------------------------------------------------
# Prediction
# -------------------------------------------------
@app.post("/predict")
def predict(
    data: TransactionInput,
    api_key: APIKey = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    input_dict = data.dict()
    tx_type = input_dict.pop("transaction_type").upper()

    for f in features:
        if f.startswith("type_"):
            input_dict[f] = 1 if f == f"type_{tx_type}" else 0

    for f in features:
        input_dict.setdefault(f, 0)

    X = np.array([[input_dict[f] for f in features]])
    X_scaled = scaler.transform(X)

    prob = float(model.predict_proba(X_scaled)[0][1])
    pred = int(prob >= api_key.threshold)

    log = PredictionLog(
        api_key_id=api_key.id,
        fraud_probability=prob,
        fraud_prediction=pred,
        amount=data.amount
    )

    db.add(log)
    db.commit()

    return {
        "fraud_prediction": pred,
        "fraud_probability": round(prob, 4)
    }