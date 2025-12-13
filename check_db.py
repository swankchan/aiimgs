"""Check database contents"""
import psycopg2
from db_config import get_connection

try:
    conn = get_connection()
    cur = conn.cursor()
    
    # Check users table
    cur.execute('SELECT COUNT(*) FROM users')
    user_count = cur.fetchone()[0]
    print(f'Users: {user_count}')
    
    # Check image_metadata table
    cur.execute('SELECT COUNT(*) FROM image_metadata')
    metadata_count = cur.fetchone()[0]
    print(f'Image metadata records: {metadata_count}')
    
    # Show sample metadata
    if metadata_count > 0:
        print('\nSample metadata rows:')
        cur.execute('SELECT image_path, caption, keywords FROM image_metadata LIMIT 5')
        rows = cur.fetchall()
        for row in rows:
            path, caption, keywords = row
            print(f'  Path: {path}')
            print(f'  Caption: {caption or "(empty)"}')
            print(f'  Keywords: {keywords or []}')
            print()
    else:
        print('\nNo metadata records found - database is empty')
    
    cur.close()
    conn.close()
    
except Exception as e:
    print(f'Error: {e}')
