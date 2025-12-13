"""
Fix metadata schema issues:
1. Remove caption column from images table (should only be in image_metadata)
2. Clean origin/timestamp tags from keywords arrays in image_metadata
3. Remove empty/useless rows from image_metadata
"""
import sys
from db_config import get_connection

def fix_schema():
    conn = get_connection()
    cur = conn.cursor()
    
    print("=== Fixing Metadata Schema ===\n")
    
    # 1. Remove caption column from images table if it exists
    print("1. Checking images.caption column...")
    try:
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'images' AND column_name = 'caption'
        """)
        if cur.fetchone():
            print("   Found images.caption - this column should not exist (caption belongs in image_metadata)")
            print("   Keeping it for now but it should be empty after this script runs")
            # Clear any captions from images table
            cur.execute("UPDATE images SET caption = NULL WHERE caption IS NOT NULL")
            conn.commit()
            print("   ✓ Cleared caption values from images table")
        else:
            print("   ✓ images.caption column doesn't exist (good)")
    except Exception as e:
        print(f"   ⚠ Error checking caption column: {e}")
        conn.rollback()
    
    # 2. Clean keywords arrays - remove origin_user, origin_pdf, uploaded_at tags
    print("\n2. Cleaning system tags from keywords arrays...")
    try:
        cur.execute("""
            UPDATE image_metadata
            SET keywords = ARRAY(
                SELECT elem
                FROM unnest(keywords) AS elem
                WHERE elem NOT LIKE 'origin_user:%'
                  AND elem NOT LIKE 'origin_pdf:%'
                  AND elem NOT LIKE 'uploaded_at:%'
            )
            WHERE keywords IS NOT NULL
              AND (
                  EXISTS (SELECT 1 FROM unnest(keywords) AS k WHERE k LIKE 'origin_user:%')
                  OR EXISTS (SELECT 1 FROM unnest(keywords) AS k WHERE k LIKE 'origin_pdf:%')
                  OR EXISTS (SELECT 1 FROM unnest(keywords) AS k WHERE k LIKE 'uploaded_at:%')
              )
        """)
        cleaned = cur.rowcount
        conn.commit()
        print(f"   ✓ Cleaned {cleaned} rows")
    except Exception as e:
        print(f"   ⚠ Error cleaning keywords: {e}")
        conn.rollback()
    
    # 3. Remove rows with no useful data (empty caption and empty/null keywords)
    print("\n3. Removing empty metadata rows...")
    try:
        cur.execute("""
            DELETE FROM image_metadata
            WHERE (caption IS NULL OR caption = '')
              AND (keywords IS NULL OR keywords = '{}' OR array_length(keywords, 1) IS NULL)
        """)
        removed = cur.rowcount
        conn.commit()
        print(f"   ✓ Removed {removed} empty rows")
    except Exception as e:
        print(f"   ⚠ Error removing empty rows: {e}")
        conn.rollback()
    
    # 4. Show final state
    print("\n=== Final State ===")
    cur.execute("SELECT COUNT(*) FROM images")
    images_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM image_metadata")
    metadata_count = cur.fetchone()[0]
    
    print(f"Images table: {images_count} rows")
    print(f"Image_metadata table: {metadata_count} rows")
    
    # Show sample from each table
    print("\n--- Sample from images ---")
    cur.execute("""
        SELECT path, uploaded_by, origin_pdf, uploaded_at 
        FROM images 
        ORDER BY created_at DESC 
        LIMIT 3
    """)
    for row in cur.fetchall():
        print(f"  {row[0]} | by:{row[1]} | pdf:{row[2]} | at:{row[3]}")
    
    print("\n--- Sample from image_metadata ---")
    cur.execute("""
        SELECT image_path, caption, keywords 
        FROM image_metadata 
        ORDER BY created_at DESC 
        LIMIT 3
    """)
    for row in cur.fetchall():
        print(f"  {row[0]} | caption:'{row[1]}' | keywords:{row[2]}")
    
    cur.close()
    conn.close()
    print("\n✅ Schema fix completed!")

if __name__ == "__main__":
    fix_schema()
