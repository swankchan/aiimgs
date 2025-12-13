"""
Authentication router
"""
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
import jwt
from typing import Optional
import hashlib
import sys
from pathlib import Path

# Add parent directory to path to import db modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.models.schemas import UserLogin, UserCreate, UserResponse, Token
from db_config import get_connection

router = APIRouter()
security = HTTPBearer()

# JWT settings
SECRET_KEY = "your-secret-key-change-in-production"  # TODO: Move to environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username: str, password: str) -> dict | None:
    """Verify user credentials and return user info"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        
        cursor.execute("""
            SELECT id, username, full_name, email, is_admin, created_at, last_login
            FROM users
            WHERE username = %s AND password_hash = %s
        """, (username, password_hash))
        
        result = cursor.fetchone()
        
        if result:
            # Update last login
            cursor.execute("""
                UPDATE users
                SET last_login = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (result[0],))
            conn.commit()
            
            user_info = {
                'id': result[0],
                'username': result[1],
                'full_name': result[2],
                'email': result[3],
                'is_admin': result[4],
                'created_at': result[5],
                'last_login': result[6]
            }
            
            cursor.close()
            conn.close()
            return user_info
        
        cursor.close()
        conn.close()
        return None
        
    except Exception as e:
        print(f"Error verifying user: {e}")
        return None

def create_user(username: str, password: str, full_name: str = "", email: str = "", is_admin: bool = False) -> bool:
    """Create a new user"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        
        cursor.execute("""
            INSERT INTO users (username, password_hash, full_name, email, is_admin)
            VALUES (%s, %s, %s, %s, %s)
        """, (username, password_hash, full_name, email, is_admin))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating user: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str) -> dict:
    """Decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Get current user from token"""
    token = credentials.credentials
    payload = decode_token(token)
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    
    # Get user from database
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, username, full_name, email, is_admin, created_at, last_login
            FROM users WHERE id = %s
        """, (user_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return {
            "id": result[0],
            "username": result[1],
            "full_name": result[2] or "",
            "email": result[3] or "",
            "is_admin": result[4],
            "created_at": result[5],
            "last_login": result[6]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

async def require_admin(current_user: dict = Depends(get_current_user)):
    """Require admin user"""
    if not current_user.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

@router.post("/login", response_model=Token)
async def login(user_login: UserLogin):
    """Login endpoint"""
    user_info = verify_user(user_login.username, user_login.password)
    
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user_info["id"])},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_info
    }

@router.post("/register", response_model=UserResponse)
async def register(user_create: UserCreate, current_user: dict = Depends(require_admin)):
    """Register new user (admin only)"""
    success = create_user(
        user_create.username,
        user_create.password,
        user_create.full_name,
        user_create.email,
        user_create.is_admin
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    # Return created user
    user_info = verify_user(user_create.username, user_create.password)
    return user_info

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user info"""
    return current_user

@router.get("/users", response_model=list[UserResponse])
async def list_users(current_user: dict = Depends(require_admin)):
    """List all users (admin only)"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, username, full_name, email, is_admin, created_at, last_login
            FROM users ORDER BY username
        """)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        users = []
        for row in results:
            users.append({
                "id": row[0],
                "username": row[1],
                "full_name": row[2],
                "email": row[3],
                "is_admin": row[4],
                "created_at": row[5],
                "last_login": row[6]
            })
        
        return users
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
