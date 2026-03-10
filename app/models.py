from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db import Base

class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    api_keys = relationship("APIKey", back_populates="organization")


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True)
    key = Column(String, unique=True, nullable=False)
    org_id = Column(Integer, ForeignKey("organizations.id"))
    threshold = Column(Float, default=0.5)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    organization = relationship("Organization", back_populates="api_keys")
    predictions = relationship("PredictionLog", back_populates="api_key")


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True)
    api_key_id = Column(Integer, ForeignKey("api_keys.id"))
    fraud_probability = Column(Float, nullable=False)
    fraud_prediction = Column(Integer, nullable=False)
    amount = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    api_key = relationship("APIKey", back_populates="predictions")