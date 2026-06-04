"""
OmixTwin Authentication — JWT + SHA256 (no bcrypt dependency issues)
"""
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = "omixtwin-secret-2025-bi-int-oncology"
ALGORITHM  = "HS256"
TOKEN_EXPIRE_HOURS = 8

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def _hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _verify(plain: str, hashed: str) -> bool:
    return hmac.compare_digest(_hash_pw(plain), hashed)


USERS = {
    "dorsane": {
        "username":  "dorsane",
        "full_name": "Zertit Dorsane",
        "role":      "CEO",
        "hashed_pw": _hash_pw("OmixAdmin2025!"),
    },
    "mahinar": {
        "username":  "mahinar",
        "full_name": "Zertit Mahinar",
        "role":      "CTO",
        "hashed_pw": _hash_pw("OmixCTO2025!"),
    },
    "admin": {
        "username":  "admin",
        "full_name": "Admin",
        "role":      "Admin",
        "hashed_pw": _hash_pw("omixtwin2025"),
    },
}


def authenticate(username: str, password: str) -> Optional[dict]:
    user = USERS.get(username)
    if not user:
        return None
    if not _verify(password, user["hashed_pw"]):
        return None
    return user


def create_token(data: dict, expires_hours: int = TOKEN_EXPIRE_HOURS) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=expires_hours)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return payload
