"""
Custom exceptions for the fraud detection API
"""
from fastapi import HTTPException, status


class FraudDetectionException(Exception):
    """Base exception for fraud detection API"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class InvalidAPIKeyError(FraudDetectionException):
    """Raised when API key is invalid or expired"""
    def __init__(self, message: str = "Invalid or expired API key"):
        super().__init__(message, status_code=status.HTTP_401_UNAUTHORIZED)


class RateLimitExceededError(FraudDetectionException):
    """Raised when rate limit is exceeded"""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=status.HTTP_429_TOO_MANY_REQUESTS)


class ModelNotLoadedError(FraudDetectionException):
    """Raised when ML model is not loaded"""
    def __init__(self, message: str = "Model not loaded"):
        super().__init__(message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


class ValidationError(FraudDetectionException):
    """Raised when request validation fails"""
    def __init__(self, message: str = "Validation error"):
        super().__init__(message, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


class DatabaseError(FraudDetectionException):
    """Raised when database operation fails"""
    def __init__(self, message: str = "Database error"):
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


class OrganizationNotFoundError(FraudDetectionException):
    """Raised when organization is not found"""
    def __init__(self, message: str = "Organization not found"):
        super().__init__(message, status_code=status.HTTP_404_NOT_FOUND)
