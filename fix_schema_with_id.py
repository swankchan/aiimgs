"""
Restructure tables to use proper id-based relationship:
1. Drop and recreate image_metadata with image_id foreign key
2. Ensure single row per image
3. Normalize path format
"""
import sys
from pathlib import Path
from db_config import get_connection

def normalize_path(path_str):
    """Normalize path to relative format from workspace root"""
    p = Path(path_str)
    # If absolute and contains AIIMGS, make it relative
    if p.is_absolute():
        try:
            # Try to make relative to workspace
            workspace = Path(r'C:\AIIMGS')
            if workspace in p.parents or p == workspace:
                return str(p.relative_to(workspace)).replace('\\', '/')
        except:
            pass
    # Already relative or couldn't convert
    return str(p).replace('\\', '/')

def fix_schema():
    conn = get_connection()
    cur = conn.cursor()
    
    print("=== Restructuring Schema with ID-based Relationship ===\n")
    
    # 1. Get all data from current image_metadata before dropping
    print("1. Backing up image_metadata data...")
    cur.execute("SELECT image_path, caption, keywords FROM image_metadata")
    backup_data = {}
    for row in cur.fetchall():
        path = normalize_path(row[0])
        # Keep the first occurrence with actual data
        if path not in backup_data or (row[1] or row[2]):
            backup_data[path] = {
                'caption': row[1] or '',
                'keywords': row[2] or []
            }
    print(f"   Backed up {len(backup_data)} unique paths")
    
    # 2. Normalize paths in images table
    print("\n2. Normalizing paths in images table...")
    cur.execute("SELECT id, path FROM images")
    images = cur.fetchall()
    for img_id, old_path in images:
        new_path = normalize_path(old_path)
        if new_path != old_path:
            cur.execute("UPDATE images SET path = %s WHERE id = %s", (new_path, img_id))
    conn.commit()
    print(f"   Normalized {len(images)} paths")
    
    # 3. Drop old image_metadata table and recreate with proper schema
    print("\n3. Recreating image_metadata table with image_id foreign key...")
    cur.execute("DROP TABLE IF EXISTS image_metadata CASCADE")
    cur.execute("""
        CREATE TABLE image_metadata (
            id SERIAL PRIMARY KEY,
            image_id INTEGER UNIQUE NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            caption TEXT,
            keywords TEXT[],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    print("   ✓ Table recreated with image_id foreign key")
    
    # 4. Restore data using image_id from images table
    print("\n4. Restoring metadata linked by image_id...")
    cur.execute("SELECT id, path FROM images")
    restored = 0
    for img_id, path in cur.fetchall():
        norm_path = normalize_path(path)
        if norm_path in backup_data:
            meta = backup_data[norm_path]
            cur.execute("""
                INSERT INTO image_metadata (image_id, caption, keywords)
                VALUES (%s, %s, %s)
            """, (img_id, meta['caption'], meta['keywords']))
            restored += 1
    conn.commit()
    print(f"   ✓ Restored {restored} metadata records")
    
    # 5. Verify final state
    print("\n=== Final State ===")
    cur.execute("SELECT COUNT(*) FROM images")
    images_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM image_metadata")
    metadata_count = cur.fetchone()[0]
    
    print(f"Images table: {images_count} rows")
    print(f"Image_metadata table: {metadata_count} rows")
    
    if images_count == metadata_count:
        print("✓ Perfect 1:1 relationship")
    else:
        print(f"⚠ Mismatch: {images_count - metadata_count} images without metadata")
    
    # Show sample
    print("\n--- Sample joined data ---")
    cur.execute("""
        SELECT i.id, i.path, i.uploaded_by, m.caption, m.keywords
        FROM images i
        LEFT JOIN image_metadata m ON i.id = m.image_id
        ORDER BY i.id
        LIMIT 5
    """)
    for row in cur.fetchall():
        print(f"  ID {row[0]}: {row[1]}")
        print(f"    uploaded_by: {row[2]}")
        print(f"    caption: '{row[3]}'")
        print(f"    keywords: {row[4]}")
        print()
    
    cur.close()
    conn.close()
    print("✅ Schema restructure completed!")

if __name__ == "__main__":
    fix_schema()
