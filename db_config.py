"""
Database configuration and connection management
"""
import os
from typing import Optional
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database connection settings
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'aiimgs'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres'),
}

# Optional: Connection pool settings
POOL_CONFIG = {
    'minconn': 1,
    'maxconn': 10,
}

def get_connection():
    """
    Get database connection
    
    Usage:
        import psycopg2
        conn = get_connection()
        cursor = conn.cursor()
    """
    try:
        import psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except ImportError:
        raise ImportError("Please install psycopg2: pip install psycopg2-binary")
    except Exception as e:
        raise Exception(f"Failed to connect to database: {e}")

def init_db():
    """
    Initialize database schema with tables and pgvector extension
    Run this once to set up the database
    """
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    
    # Connect to PostgreSQL server (not specific database)
    conn = psycopg2.connect(
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Create database if not exists
    cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_CONFIG['database']}'")
    if not cursor.fetchone():
        cursor.execute(f"CREATE DATABASE {DB_CONFIG['database']}")
        print(f"✓ Database '{DB_CONFIG['database']}' created")
    else:
        print(f"✓ Database '{DB_CONFIG['database']}' already exists")
    
    cursor.close()
    conn.close()
    
    # Connect to the specific database
    conn = get_connection()
    cursor = conn.cursor()
    
    # Try to enable pgvector extension (skip if not available)
    try:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        print("✓ pgvector extension enabled")
        use_vector = True
    except Exception as e:
        conn.rollback()  # Rollback failed transaction
        print(f"⚠ pgvector extension not available (will use BYTEA for embeddings)")
        print(f"  To install pgvector: https://github.com/pgvector/pgvector")
        use_vector = False
    
    # Create images table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id SERIAL PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("✓ Table 'images' created")
    
    # Create image_metadata table with foreign key to images
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS image_metadata (
            id SERIAL PRIMARY KEY,
            image_id INTEGER UNIQUE REFERENCES images(id) ON DELETE CASCADE,
            caption TEXT,
            keywords TEXT[],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("✓ Table 'image_metadata' created")
    
    # Create index on image_metadata for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_image_metadata_image_id ON image_metadata(image_id)
    """)
    
    # Create PDFs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pdfs (
            id SERIAL PRIMARY KEY,
            path TEXT UNIQUE NOT NULL,
            extracted_text TEXT,
            page_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("✓ Table 'pdfs' created")
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print("\n✅ Database initialization complete!")

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
