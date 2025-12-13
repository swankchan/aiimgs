"""
Admin router for user management
"""
from fastapi import APIRouter, HTTPException, Depends, status
from typing import List
import sys
from pathlib import Path

# Add parent directory to path to import db modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.models.schemas import UserCreate, UserResponse, UserUpdate
from api.routers.auth import require_admin, hash_password, get_current_user
from db_config import get_connection

router = APIRouter()

@router.get("/users", response_model=List[UserResponse])
async def get_all_users(current_user: dict = Depends(require_admin)):
    """Get all users (admin only)"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, username, full_name, email, is_admin, created_at, last_login
            FROM users ORDER BY created_at DESC
        """)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        users = []
        for row in results:
            users.append({
                "id": row[0],
                "username": row[1],
                "full_name": row[2] or "",
                "email": row[3] or "",
                "is_admin": row[4],
                "created_at": row[5],
                "last_login": row[6]
            })
        
        return users
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_new_user(user_create: UserCreate, current_user: dict = Depends(require_admin)):
    """Create new user (admin only)"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check if username already exists
        cursor.execute("SELECT id FROM users WHERE username = %s", (user_create.username,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        # Hash password and create user
        password_hash = hash_password(user_create.password)
        
        cursor.execute("""
            INSERT INTO users (username, password_hash, full_name, email, is_admin)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, username, full_name, email, is_admin, created_at, last_login
        """, (
            user_create.username,
            password_hash,
            user_create.full_name or "",
            user_create.email or "",
            user_create.is_admin
        ))
        
        result = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            "id": result[0],
            "username": result[1],
            "full_name": result[2] or "",
            "email": result[3] or "",
            "is_admin": result[4],
            "created_at": result[5],
            "last_login": result[6]
        }
    except HTTPException:
        raise
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(user_id: int, user_update: UserUpdate, current_user: dict = Depends(require_admin)):
    """Update user (admin only)"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
        if not cursor.fetchone():
            cursor.close()
            conn.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Build update query dynamically based on provided fields
        update_fields = []
        update_values = []
        
        if user_update.full_name is not None:
            update_fields.append("full_name = %s")
            update_values.append(user_update.full_name)
        
        if user_update.email is not None:
            update_fields.append("email = %s")
            update_values.append(user_update.email)
        
        if user_update.is_admin is not None:
            update_fields.append("is_admin = %s")
            update_values.append(user_update.is_admin)
        
        if user_update.password is not None:
            update_fields.append("password_hash = %s")
            update_values.append(hash_password(user_update.password))
        
        if not update_fields:
            cursor.close()
            conn.close()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )
        
        # Add user_id to values
        update_values.append(user_id)
        
        # Execute update
        query = f"""
            UPDATE users
            SET {', '.join(update_fields)}
            WHERE id = %s
            RETURNING id, username, full_name, email, is_admin, created_at, last_login
        """
        
        cursor.execute(query, update_values)
        result = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            "id": result[0],
            "username": result[1],
            "full_name": result[2] or "",
            "email": result[3] or "",
            "is_admin": result[4],
            "created_at": result[5],
            "last_login": result[6]
        }
    except HTTPException:
        raise
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int, current_user: dict = Depends(require_admin)):
    """Delete user (admin only)"""
    # Prevent self-deletion
    if current_user["id"] == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
        if not cursor.fetchone():
            cursor.close()
            conn.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Delete user
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        cursor.close()
        conn.close()
        
        return None
    except HTTPException:
        raise
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
