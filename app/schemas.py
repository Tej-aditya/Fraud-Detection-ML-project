"""
Enhanced request validation schemas with strict validation
"""
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """Validated prediction request with strict field constraints"""
    
    # Transaction details
    step: int = Field(ge=1, le=743, description="Hour of transaction (1-743)")
    amount: float = Field(gt=0, le=10000000, description="Transaction amount")
    
    # Balance information
    oldbalanceOrg: float = Field(ge=0, description="Origin account balance before transaction")
    newbalanceOrig: float = Field(ge=0, description="Origin account balance after transaction")
    oldbalanceDest: float = Field(ge=0, description="Destination account balance before transaction")
    newbalanceDest: float = Field(ge=0, description="Destination account balance after transaction")
    
    # Transaction type
    transaction_type: Literal["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"] = Field(
        description="Type of transaction"
    )
    
    # Additional features
    transaction_hour: int = Field(ge=0, le=23, description="Hour of day (0-23)")
    transaction_day: int = Field(ge=1, le=31, description="Day of month (1-31)")
    merchant_category: int = Field(ge=0, le=9999, description="Merchant category code")
    user_transaction_count_24h: int = Field(ge=0, le=1000, description="User's transaction count in last 24h")
    user_avg_transaction_amount: float = Field(ge=0, description="User's average transaction amount")
    is_international: int = Field(ge=0, le=1, description="1 if international, 0 if domestic")
    distance_from_home: float = Field(ge=0, le=20000, description="Distance from home location (km)")
    device_change: int = Field(ge=0, le=1, description="1 if device changed, 0 otherwise")
    failed_login_attempts: int = Field(ge=0, le=10, description="Failed login attempts before transaction")
    
    @validator('amount')
    def validate_amount(cls, v):
        """Ensure amount is reasonable"""
        if v <= 0:
            raise ValueError("Amount must be positive")
        if v > 10000000:
            raise ValueError("Amount exceeds maximum allowed")
        return v
    
    @validator('newbalanceOrig')
    def validate_balance_consistency(cls, v, values):
        """Check balance consistency"""
        if 'oldbalanceOrg' in values and 'amount' in values:
            expected = values['oldbalanceOrg'] - values['amount']
            # Allow some tolerance for floating point
            if abs(v - expected) > 0.01 and expected >= 0:
                # This is suspicious but not necessarily invalid
                pass
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "step": 1,
                "amount": 9500.0,
                "oldbalanceOrg": 20000.0,
                "newbalanceOrig": 10500.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "transaction_type": "TRANSFER",
                "transaction_hour": 14,
                "transaction_day": 15,
                "merchant_category": 5411,
                "user_transaction_count_24h": 3,
                "user_avg_transaction_amount": 500.0,
                "is_international": 0,
                "distance_from_home": 0.0,
                "device_change": 0,
                "failed_login_attempts": 0
            }
        }


class PredictionResponse(BaseModel):
    """Standardized prediction response"""
    fraud_prediction: int = Field(description="0 for legitimate, 1 for fraud")
    fraud_probability: float = Field(ge=0, le=1, description="Probability of fraud (0-1)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "fraud_prediction": 0,
                "fraud_probability": 0.0403,
                "timestamp": "2026-02-15T07:43:00Z"
            }
        }


class OrganizationCreate(BaseModel):
    """Organization creation request"""
    name: str = Field(min_length=1, max_length=255, description="Organization name")
    
    @validator('name')
    def validate_name(cls, v):
        """Sanitize organization name"""
        # Remove leading/trailing whitespace
        v = v.strip()
        if not v:
            raise ValueError("Organization name cannot be empty")
        # Basic sanitization - remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", ';', '&', '|']
        for char in dangerous_chars:
            if char in v:
                raise ValueError(f"Organization name contains invalid character: {char}")
        return v


class APIKeyCreate(BaseModel):
    """API key creation request"""
    org_id: int = Field(gt=0, description="Organization ID")
    threshold: float = Field(ge=0, le=1, default=0.5, description="Fraud probability threshold")
    rate_limit: Optional[int] = Field(default=100, ge=1, le=10000, description="Requests per minute")


class APIKeyRotateRequest(BaseModel):
    """API key rotation request"""
    old_key: str = Field(min_length=10, description="Current API key to rotate")
    grace_period_days: int = Field(default=7, ge=1, le=30, description="Days before old key expires")


class ErrorResponse(BaseModel):
    """Standardized error response"""
    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    detail: Optional[str] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid request parameters",
                "detail": "Amount must be positive",
                "timestamp": "2026-02-15T07:43:00Z"
            }
        }
