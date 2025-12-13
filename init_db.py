"""
Initialize database and create default admin user for FastAPI version
"""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from db_config import get_connection
import hashlib

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_users_table():
    """Create users table if not exists"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(64) NOT NULL,
                full_name VARCHAR(100),
                email VARCHAR(100),
                is_admin BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating users table: {e}")
        return False

def create_metadata_table():
    """Create metadata table if not exists"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_metadata (
                id SERIAL PRIMARY KEY,
                image_path TEXT UNIQUE NOT NULL,
                caption TEXT,
                keywords TEXT[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_image_path 
            ON image_metadata(image_path)
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error creating metadata table: {e}")
        return False

def create_user(username: str, password: str, full_name: str = "", email: str = "", is_admin: bool = False) -> bool:
    """Create a new user"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        
        cursor.execute("""
            INSERT INTO users (username, password_hash, full_name, email, is_admin)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (username) DO NOTHING
        """, (username, password_hash, full_name, email, is_admin))
        
        rows_affected = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        return rows_affected > 0
    except Exception as e:
        print(f"Error creating user: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False

if __name__ == "__main__":
    print("========================================")
    print("  AI Image Search - Database Setup")
    print("========================================")
    print()
    
    # Create users table
    print("Creating users table...")
    if create_users_table():
        print("✓ Users table created")
    else:
        print("✗ Failed to create users table")
        sys.exit(1)
    
    # Create metadata table
    print("Creating metadata table...")
    if create_metadata_table():
        print("✓ Metadata table created")
    else:
        print("✗ Failed to create metadata table")
        sys.exit(1)
    
    print()
    
    # Create default admin user
    print("Creating default admin user...")
    if create_user("admin", "admin123", "Administrator", "", is_admin=True):
        print("✓ Default admin user created")
        print()
        print("  Username: admin")
        print("  Password: admin123")
        print()
        print("⚠ IMPORTANT: Please change this password after first login!")
    else:
        print("ℹ Admin user already exists (or creation failed)")
    
    print()
    print("========================================")
    print("  Database setup complete!")
    print("========================================")
    print()
    print("Next steps:")
    print("1. Install frontend dependencies: cd frontend && npm install")
    print("2. Start backend: uvicorn api.main:app --reload --port 8000")
    print("3. Start frontend: cd frontend && npm run dev")
    print("4. Visit: http://localhost:3000")
