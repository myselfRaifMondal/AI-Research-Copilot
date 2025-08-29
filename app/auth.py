import os
import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .config import settings

# Simple API key authentication for development
security = HTTPBearer()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Simple API key verification for development.
    
    In production, replace with JWT or integrate with enterprise SSO:
    - JWT: Decode and verify token signature
    - SSO: Integrate with Auth0, Keycloak, or Azure AD
    - Store secrets in AWS Secrets Manager or GCP Secret Manager
    """
    if credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def create_jwt_token(user_id: str, expires_hours: int = 24) -> str:
    """
    Create JWT token (template for production).
    
    Usage in production:
    1. Replace simple API key with this JWT system
    2. Add user management and role-based permissions
    3. Store JWT_SECRET in secure key management system
    """
    expires = datetime.now(timezone.utc) + timedelta(hours=expires_hours)
    payload = {
        "user_id": user_id,
        "exp": expires,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def verify_jwt_token(token: str) -> Optional[dict]:
    """Verify JWT token (template for production)."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# Rate limiting placeholder (implement with Redis in production)
class RateLimiter:
    """
    Simple in-memory rate limiter.
    
    Production recommendations:
    - Use Redis with sliding window algorithm
    - Implement per-user/per-API-key limits
    - Add exponential backoff for repeated violations
    """
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, identifier: str, limit: int = 100, window: int = 3600) -> bool:
        """Check if request is within rate limit."""
        now = datetime.now(timezone.utc).timestamp()
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < window
        ]
        
        # Check limit
        if len(self.requests[identifier]) >= limit:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter()
