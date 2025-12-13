"""
Test database connection and basic operations
"""
from db_config import get_connection
import json
from pathlib import Path

def test_connection():
    """Test basic database connection"""
    print("=" * 50)
    print("Testing Database Connection")
    print("=" * 50)
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Test query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✓ Connected to PostgreSQL")
        print(f"  Version: {version[0][:80]}...")
        
        # Check tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = cursor.fetchall()
        print(f"\n✓ Tables found: {len(tables)}")
        for table in tables:
            print(f"  - {table[0]}")
        
        cursor.close()
        conn.close()
        print("\n✅ Connection test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Connection test failed!")
        print(f"Error: {e}")
        return False

def test_insert_and_query():
    """Test inserting and querying data"""
    print("\n" + "=" * 50)
    print("Testing Insert & Query Operations")
    print("=" * 50)
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Insert test image
        test_path = "test_image.jpg"
        test_caption = "Test image for database"
        
        cursor.execute("""
            INSERT INTO images (path, caption)
            VALUES (%s, %s)
            ON CONFLICT (path) DO UPDATE SET caption = EXCLUDED.caption
            RETURNING id
        """, (test_path, test_caption))
        
        image_id = cursor.fetchone()[0]
        print(f"✓ Inserted test image with ID: {image_id}")
        
        # Insert test keywords
        test_keywords = ["test", "database", "demo"]
        for keyword in test_keywords:
            cursor.execute("""
                INSERT INTO keywords (image_id, keyword)
                VALUES (%s, %s)
                ON CONFLICT (image_id, keyword) DO NOTHING
            """, (image_id, keyword))
        
        conn.commit()
        print(f"✓ Inserted {len(test_keywords)} keywords")
        
        # Query back the data
        cursor.execute("""
            SELECT i.id, i.path, i.caption, i.created_at
            FROM images i
            WHERE i.path = %s
        """, (test_path,))
        
        result = cursor.fetchone()
        if result:
            print(f"\n✓ Query result:")
            print(f"  ID: {result[0]}")
            print(f"  Path: {result[1]}")
            print(f"  Caption: {result[2]}")
            print(f"  Created: {result[3]}")
        
        # Query keywords
        cursor.execute("""
            SELECT keyword
            FROM keywords
            WHERE image_id = %s
            ORDER BY keyword
        """, (image_id,))
        
        keywords = [row[0] for row in cursor.fetchall()]
        print(f"  Keywords: {', '.join(keywords)}")
        
        # Count total records
        cursor.execute("SELECT COUNT(*) FROM images")
        image_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM keywords")
        keyword_count = cursor.fetchone()[0]
        
        print(f"\n✓ Database summary:")
        print(f"  Total images: {image_count}")
        print(f"  Total keywords: {keyword_count}")
        
        cursor.close()
        conn.close()
        
        print("\n✅ Insert & query test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Insert & query test failed!")
        print(f"Error: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False

def test_metadata_compatibility():
    """Test loading existing metadata.json"""
    print("\n" + "=" * 50)
    print("Testing Metadata Compatibility")
    print("=" * 50)
    
    metadata_path = Path("metadata-files/clip-vit-b-32/metadata.json")
    
    if not metadata_path.exists():
        print("⚠ metadata.json not found, skipping compatibility test")
        return True
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"✓ Loaded metadata.json: {len(metadata)} entries")
        
        # Show first entry structure
        if metadata:
            first_key = list(metadata.keys())[0]
            first_entry = metadata[first_key]
            print(f"\n✓ Sample entry structure:")
            print(f"  Path: {first_key}")
            print(f"  Caption: {first_entry.get('caption', 'N/A')[:50]}")
            print(f"  Keywords: {len(first_entry.get('keywords', []))} keywords")
        
        print(f"\n✅ Metadata structure is compatible!")
        return True
        
    except Exception as e:
        print(f"\n❌ Metadata compatibility test failed!")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Starting Database Tests\n")
    
    # Run tests
    test1 = test_connection()
    test2 = test_insert_and_query()
    test3 = test_metadata_compatibility()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Connection test: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Insert & Query test: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Metadata compatibility: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if test1 and test2 and test3:
        print("\n🎉 All tests passed! Ready to migrate to database.")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
