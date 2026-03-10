from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import secrets

from app.db import SessionLocal
from app.models import Organization, APIKey, PredictionLog
from pydantic import BaseModel

router = APIRouter(prefix="/admin", tags=["admin"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class OrgCreate(BaseModel):
    name: str


class APIKeyCreate(BaseModel):
    org_id: int
    threshold: float = 0.5


@router.get("/health/db")
def db_health(db: Session = Depends(get_db)):
    return {
        "organizations": db.query(Organization).count(),
        "api_keys": db.query(APIKey).count(),
        "prediction_logs": db.query(PredictionLog).count()
    }


@router.post("/org")
def create_org(payload: OrgCreate, db: Session = Depends(get_db)):
    org = Organization(name=payload.name)
    db.add(org)
    db.commit()
    db.refresh(org)
    return org


@router.post("/apikey")
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