"""
Enhanced Fraud Detection API with Security & Production Features
"""
from fastapi import FastAPI, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import secrets
import logging
from sqlalchemy.orm import Session
from slowapi.errors import RateLimitExceeded
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Import app modules
from app.db import SessionLocal, Base, engine
from app.models import Organization, APIKey, PredictionLog
from app.seed import seed_initial_data
from app.schemas import (
    PredictionRequest, PredictionResponse, OrganizationCreate,
    APIKeyCreate, APIKeyRotateRequest, ErrorResponse
)
from app.exceptions import (
    InvalidAPIKeyError, RateLimitExceededError, ModelNotLoadedError,
    ValidationError as CustomValidationError, FraudDetectionException
)
from app.logging_config import setup_logging, get_logger
from app.middleware.rate_limit import limiter, get_api_key_identifier
from app.api.admin import router as admin_router

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)

# Prometheus metrics
prediction_counter = Counter('fraud_predictions_total', 'Total number of predictions', ['result'])
prediction_latency = Histogram('fraud_prediction_latency_seconds', 'Prediction latency')
api_requests = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method', 'status'])

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
# Lifespan
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, features
    
    logger.info("Starting application initialization")
    
    try:
        # Load model artifacts
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)
        logger.info(f"Model loaded successfully with {len(features)} features")
        
        # Seed initial data
        seed_initial_data()
        
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
        raise
    
    yield
    
    logger.info("Shutting down application")

# -------------------------------------------------
# App
# -------------------------------------------------
app = FastAPI(
    title="Fraud Detection SaaS API",
    description="Enterprise-grade fraud detection API with ML-powered predictions",
    version="2.0.0",
    lifespan=lifespan
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
    content=ErrorResponse(
        error="RateLimitExceeded",
        message="Too many requests. Please try again later.",
        timestamp=datetime.utcnow()
    ).dict()
))

# Include admin router
app.include_router(admin_router)

# -------------------------------------------------
# Exception Handlers
# -------------------------------------------------
@app.exception_handler(FraudDetectionException)
async def fraud_detection_exception_handler(request: Request, exc: FraudDetectionException):
    """Handle custom fraud detection exceptions"""
    logger.warning(f"FraudDetectionException: {exc.message}", extra={
        "status_code": exc.status_code,
        "path": request.url.path
    })
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.message,
            timestamp=datetime.utcnow()
        ).dict()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors"""
    logger.warning(f"Validation error: {exc.errors()}", extra={
        "path": request.url.path
    })
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="ValidationError",
            message="Invalid request parameters",
            detail=str(exc.errors()),
            timestamp=datetime.utcnow()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {exc}", exc_info=True, extra={
        "path": request.url.path
    })
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            timestamp=datetime.utcnow()
        ).dict()
    )

# -------------------------------------------------
# Middleware
# -------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with correlation ID"""
    correlation_id = request.headers.get("X-Correlation-ID", secrets.token_hex(8))
    
    start_time = datetime.utcnow()
    logger.info(f"Request started", extra={
        "correlation_id": correlation_id,
        "method": request.method,
        "path": request.url.path,
        "client": request.client.host if request.client else None
    })
    
    response = await call_next(request)
    
    duration = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"Request completed", extra={
        "correlation_id": correlation_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_seconds": duration
    })
    
    # Track metrics
    api_requests.labels(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code
    ).inc()
    
    response.headers["X-Correlation-ID"] = correlation_id
    return response

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
    x_api_key: str = Header(..., description="API key for authentication"),
    db: Session = Depends(get_db)
) -> APIKey:
    """Verify API key and return key object"""
    key_obj = db.query(APIKey).filter(APIKey.key == x_api_key).first()
    
    if not key_obj:
        logger.warning(f"Invalid API key attempted: {x_api_key[:10]}...")
        raise InvalidAPIKeyError("Invalid API key")
    
    if not key_obj.active:
        logger.warning(f"Inactive API key attempted: {x_api_key[:10]}...")
        raise InvalidAPIKeyError("API key is inactive")
    
    # Check expiration if field exists
    if hasattr(key_obj, 'expires_at') and key_obj.expires_at:
        if datetime.utcnow() > key_obj.expires_at:
            logger.warning(f"Expired API key attempted: {x_api_key[:10]}...")
            raise InvalidAPIKeyError("API key has expired")
    
    return key_obj

# -------------------------------------------------
# Health & Metrics Endpoints
# -------------------------------------------------
@app.get("/", tags=["Health"])
async def root():
    """Basic health check"""
    return {
        "service": "Fraud Detection API",
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness check"""
    return {"status": "healthy"}

@app.get("/health/ready", tags=["Health"])
async def readiness_check(db: Session = Depends(get_db)):
    """Readiness check - verify DB and model loaded"""
    if model is None or scaler is None or features is None:
        raise ModelNotLoadedError("Model not loaded")
    
    # Check DB connectivity
    try:
        db.execute("SELECT 1")
        return {
            "status": "ready",
            "model_loaded": True,
            "database_connected": True
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )

@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# -------------------------------------------------
# Prediction Endpoint
# -------------------------------------------------
@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
@limiter.limit("100/minute")
async def predict(
    request: Request,
    prediction_request: PredictionRequest,
    api_key: APIKey = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Make fraud prediction for a transaction
    
    Rate limit: 100 requests per minute per API key
    """
    with prediction_latency.time():
        try:
            logger.info(f"Prediction request received", extra={
                "org_id": api_key.org_id,
                "amount": prediction_request.amount
            })
            
            # Check model loaded
            if model is None or scaler is None or features is None:
                raise ModelNotLoadedError()
            
            # Build feature vector
            feature_dict = {
                "step": prediction_request.step,
                "amount": prediction_request.amount,
                "oldbalanceOrg": prediction_request.oldbalanceOrg,
                "newbalanceOrig": prediction_request.newbalanceOrig,
                "oldbalanceDest": prediction_request.oldbalanceDest,
                "newbalanceDest": prediction_request.newbalanceDest,
            }
            
            # One-hot encode transaction type
            for t_type in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
                feature_dict[f"type_{t_type}"] = 1 if prediction_request.transaction_type == t_type else 0
            
            # Feature engineering
            feature_dict["amount_log"] = np.log1p(prediction_request.amount)
            feature_dict["balance_delta_org"] = prediction_request.oldbalanceOrg - prediction_request.newbalanceOrig
            feature_dict["balance_delta_dest"] = prediction_request.newbalanceDest - prediction_request.oldbalanceDest
            
            # Create DataFrame and ensure feature order
            df = pd.DataFrame([feature_dict])
            df = df[features]  # Reorder to match training
            
            # Scale and predict
            X_scaled = scaler.transform(df.values)
            fraud_prob = float(model.predict_proba(X_scaled)[0, 1])
            fraud_pred = int(fraud_prob >= api_key.threshold)
            
            # Log prediction to database
            log_entry = PredictionLog(
                api_key_id=api_key.id,
                fraud_probability=fraud_prob,
                fraud_prediction=fraud_pred,
                amount=prediction_request.amount
            )
            db.add(log_entry)
            db.commit()
            
            # Track metrics
            prediction_counter.labels(result="fraud" if fraud_pred == 1 else "legitimate").inc()
            
            logger.info(f"Prediction completed", extra={
                "org_id": api_key.org_id,
                "fraud_prediction": fraud_pred,
                "fraud_probability": fraud_prob
            })
            
            return PredictionResponse(
                fraud_prediction=fraud_pred,
                fraud_probability=round(fraud_prob, 4),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise

# -------------------------------------------------
# API Key Rotation Endpoint
# -------------------------------------------------
@app.post("/admin/apikey/rotate", tags=["Admin"])
async def rotate_api_key(
    rotate_request: APIKeyRotateRequest,
    db: Session = Depends(get_db)
):
    """
    Rotate an API key with grace period
    
    The old key will remain active for the specified grace period
    """
    # Find old key
    old_key = db.query(APIKey).filter(APIKey.key == rotate_request.old_key).first()
    if not old_key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    # Generate new key
    new_key_value = "sk_" + secrets.token_hex(16)
    new_key = APIKey(
        key=new_key_value,
        org_id=old_key.org_id,
        threshold=old_key.threshold,
        active=True
    )
    
    # Set expiration on old key
    if not hasattr(old_key, 'expires_at'):
        # If expires_at field doesn't exist, just deactivate after grace period
        # For now, keep it active
        pass
    else:
        old_key.expires_at = datetime.utcnow() + timedelta(days=rotate_request.grace_period_days)
    
    db.add(new_key)
    db.commit()
    db.refresh(new_key)
    
    logger.info(f"API key rotated", extra={
        "org_id": old_key.org_id,
        "grace_period_days": rotate_request.grace_period_days
    })
    
    return {
        "new_api_key": new_key_value,
        "old_key_expires_at": old_key.expires_at.isoformat() if hasattr(old_key, 'expires_at') and old_key.expires_at else None,
        "grace_period_days": rotate_request.grace_period_days
    }
