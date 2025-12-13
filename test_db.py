"""Test PostgreSQL connection"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': 'postgres',  # Connect to default postgres DB first
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'qazXCV25!@'),
}

print("Testing PostgreSQL connection...")
print(f"Host: {DB_CONFIG['host']}")
print(f"Port: {DB_CONFIG['port']}")
print(f"User: {DB_CONFIG['user']}")
print(f"Password: {'*' * len(DB_CONFIG['password'])}")

try:
    import psycopg2
    print("\n✓ psycopg2 module found")
except ImportError:
    print("\n✗ psycopg2 not installed!")
    print("Install it with: pip install psycopg2-binary")
    exit(1)

try:
    conn = psycopg2.connect(**DB_CONFIG)
    print("✓ Connection successful!")
    
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print(f"✓ PostgreSQL version: {version[0][:50]}...")
    
    cursor.close()
    conn.close()
    print("\n✅ Database connection works!")
    
except Exception as e:
    print(f"\n✗ Connection failed!")
    print(f"Error: {e}")
    print("\nPossible solutions:")
    print("1. Check password in .env file")
    print("2. Make sure PostgreSQL service is running")
    print("3. Check if port 5432 is correct")
