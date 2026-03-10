"""
Rate limiting middleware for the fraud detection API
"""
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from app.db import SessionLocal
from app.models import APIKey


def get_api_key_identifier(request: Request) -> str:
    """
    Extract API key from request for rate limiting
    Falls back to IP address if no API key provided
    """
    api_key = request.headers.get("x-api-key")
    
    if api_key:
        # Use API key as identifier
        return f"key:{api_key}"
    
    # Fall back to IP address
    return f"ip:{get_remote_address(request)}"


def get_rate_limit_for_key(api_key: str) -> str:
    """
    Get rate limit for a specific API key
    Returns limit string like "100/minute"
    """
    if not api_key:
        return "10/minute"  # Default for unauthenticated requests
    
    # Query database for API key rate limit
    db = SessionLocal()
    try:
        key_obj = db.query(APIKey).filter(APIKey.key == api_key).first()
        if key_obj and hasattr(key_obj, 'rate_limit'):
            return f"{key_obj.rate_limit}/minute"
        return "100/minute"  # Default for authenticated requests
    finally:
        db.close()


# Create limiter instance
limiter = Limiter(
    key_func=get_api_key_identifier,
    default_limits=["1000/hour"],
    storage_uri="memory://",  # Use in-memory storage (for production, use Redis)
    strategy="fixed-window"
)
