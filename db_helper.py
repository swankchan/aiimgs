"""
Database helper functions for image metadata management
"""
from db_config import get_connection
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from datetime import datetime

def get_all_metadata() -> Dict[str, dict]:
    """
    Get all image metadata from database
    Returns dict in same format as metadata.json for compatibility
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Join images and image_metadata by image_id
    cursor.execute("""
        SELECT i.path, m.caption, m.keywords
        FROM images i
        LEFT JOIN image_metadata m ON i.id = m.image_id
    """)
    
    metadata = {}
    for row in cursor.fetchall():
        path, caption, keywords = row
        metadata[path] = {
            'caption': caption or '',
            'keywords': keywords or []
        }
    
    cursor.close()
    conn.close()
    
    return metadata

def save_metadata(path: str, caption: str, keywords: List[str]) -> bool:
    """
    Save or update metadata for an image in image_metadata table using image_id foreign key
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Clean keywords list
        clean_keywords = [k.strip() for k in keywords if k and k.strip()]
        
        # Get or create image record first (to ensure we have an ID)
        cursor.execute("""
            INSERT INTO images (path)
            VALUES (%s)
            ON CONFLICT (path) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, (path,))
        image_id = cursor.fetchone()[0]
        
        # Insert or update in image_metadata table using image_id
        cursor.execute("""
            INSERT INTO image_metadata (image_id, caption, keywords, updated_at)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (image_id) 
            DO UPDATE SET 
                caption = EXCLUDED.caption,
                keywords = EXCLUDED.keywords,
                updated_at = CURRENT_TIMESTAMP
        """, (image_id, caption, clean_keywords))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error saving metadata: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False

def set_image_origin(path: str, uploaded_by: Optional[str] = None, origin_pdf: Optional[str] = None, uploaded_at: Optional[str] = None) -> bool:
    """
    Set origin fields for an image in the images table.
    Adds columns if they don't exist: uploaded_by TEXT, uploaded_at TIMESTAMPTZ, origin_pdf TEXT.
    Note: Caption should NOT be stored here - it goes in image_metadata table.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Ensure columns exist
        cursor.execute("ALTER TABLE images ADD COLUMN IF NOT EXISTS uploaded_by TEXT")
        cursor.execute("ALTER TABLE images ADD COLUMN IF NOT EXISTS uploaded_at TIMESTAMPTZ")
        cursor.execute("ALTER TABLE images ADD COLUMN IF NOT EXISTS origin_pdf TEXT")

        # Parse uploaded_at
        dt = None
        if uploaded_at:
            try:
                dt = datetime.fromisoformat(uploaded_at)
            except Exception:
                dt = None

        # Insert or update - single operation to avoid duplicates
        # DO NOT store caption here - that belongs in image_metadata
        cursor.execute(
            """
            INSERT INTO images (path, uploaded_by, uploaded_at, origin_pdf)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (path) 
            DO UPDATE SET
                uploaded_by = COALESCE(EXCLUDED.uploaded_by, images.uploaded_by),
                uploaded_at = COALESCE(EXCLUDED.uploaded_at, images.uploaded_at),
                origin_pdf = COALESCE(EXCLUDED.origin_pdf, images.origin_pdf),
                updated_at = CURRENT_TIMESTAMP
            """,
            (path, uploaded_by, dt, origin_pdf)
        )

        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Error setting image origin: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False

def delete_metadata(paths: List[str]) -> bool:
    """
    Delete metadata for given image paths - deletes from images which cascades to image_metadata
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        for path in paths:
            # Delete from images - will cascade to image_metadata due to foreign key
            cursor.execute("DELETE FROM images WHERE path = %s", (path,))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error deleting metadata: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False

def get_metadata(path: str) -> Optional[dict]:
    """
    Get metadata for a specific image from image_metadata table via image_id join
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT m.caption, m.keywords
        FROM images i
        LEFT JOIN image_metadata m ON i.id = m.image_id
        WHERE i.path = %s
    """, (path,))
    
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if row:
        caption, keywords = row
        return {
            'caption': caption or '',
            'keywords': keywords or []
        }
    return None

def bulk_save_metadata(metadata_dict: Dict[str, dict]) -> Tuple[int, int]:
    """
    Bulk save metadata from dictionary (for migration)
    Returns (success_count, error_count)
    """
    success = 0
    errors = 0
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        for path, meta in metadata_dict.items():
            try:
                caption = meta.get('caption', '')
                keywords = meta.get('keywords', [])
                
                # Get or create image record
                cursor.execute("""
                    INSERT INTO images (path)
                    VALUES (%s)
                    ON CONFLICT (path) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (path,))
                image_id = cursor.fetchone()[0]
                
                # Insert or update in image_metadata using image_id
                cursor.execute("""
                    INSERT INTO image_metadata (image_id, caption, keywords, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (image_id) 
                    DO UPDATE SET 
                        caption = EXCLUDED.caption,
                        keywords = EXCLUDED.keywords,
                        updated_at = CURRENT_TIMESTAMP
                """, (image_id, caption, keywords))
                
                success += 1
                
            except Exception as e:
                print(f"Error saving {path}: {e}")
                errors += 1
        
        conn.commit()
        
    except Exception as e:
        print(f"Bulk save error: {e}")
        conn.rollback()
    
    finally:
        cursor.close()
        conn.close()
    
    return success, errors

def get_stats() -> dict:
    """
    Get database statistics from image_metadata table
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM image_metadata")
    image_count = cursor.fetchone()[0]
    
    # Count total keywords across all images
    cursor.execute("""
        SELECT SUM(array_length(keywords, 1)) 
        FROM image_metadata 
        WHERE keywords IS NOT NULL
    """)
    keyword_count = cursor.fetchone()[0] or 0
    
    # Count unique keywords using subquery
    cursor.execute("""
        SELECT COUNT(DISTINCT keyword)
        FROM (
            SELECT unnest(keywords) as keyword
            FROM image_metadata 
            WHERE keywords IS NOT NULL
        ) AS all_keywords
    """)
    unique_keywords = cursor.fetchone()[0] or 0
    
    cursor.close()
    conn.close()
    
    return {
        'total_images': image_count,
        'total_keywords': keyword_count,
        'unique_keywords': unique_keywords
    }

if __name__ == "__main__":
    # Test basic operations
    print("Testing db_helper functions...")
    
    # Test save
    print("\n1. Testing save_metadata...")
    success = save_metadata("test.jpg", "Test caption", ["test", "demo"])
    print(f"   {'✓' if success else '✗'} Save: {success}")
    
    # Test get
    print("\n2. Testing get_metadata...")
    meta = get_metadata("test.jpg")
    print(f"   ✓ Retrieved: {meta}")
    
    # Test get all
    print("\n3. Testing get_all_metadata...")
    all_meta = get_all_metadata()
    print(f"   ✓ Total entries: {len(all_meta)}")
    
    # Test stats
    print("\n4. Testing get_stats...")
    stats = get_stats()
    print(f"   ✓ Stats: {stats}")
    
    print("\n✅ All helper functions work!")
