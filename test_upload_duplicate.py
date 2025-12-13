"""
Test to verify upload doesn't create duplicates
"""
import sys
sys.path.insert(0, r'c:\AIIMGS')

from pathlib import Path
from db_helper import save_metadata, set_image_origin
from db_config import get_connection
from datetime import datetime

# Clean test data first
conn = get_connection()
cur = conn.cursor()
cur.execute("DELETE FROM images WHERE path LIKE 'images/test_%'")
conn.commit()
cur.close()
conn.close()

print("=== Testing Upload Flow (simulated) ===\n")

# Simulate Windows path from upload
dest = Path("images") / "test_upload.jpg"
print(f"1. File destination: {dest}")
print(f"   str(dest) = '{str(dest)}'")

# OLD WAY (causes duplicates):
print("\n2. OLD WAY - Two different paths:")
path1 = str(dest)  # Windows backslash
path2 = str(dest)  # Same, but might get converted
print(f"   Path 1: '{path1}'")
print(f"   Path 2: '{path2}'")
print(f"   Are they equal? {path1 == path2}")

# NEW WAY (prevents duplicates):
print("\n3. NEW WAY - Normalized path:")
normalized_path = str(dest).replace('\\', '/')
print(f"   Normalized: '{normalized_path}'")

# Test actual DB operations
print("\n4. Testing DB operations with normalized path...")
uploaded_at = datetime.utcnow().isoformat()

# Call both functions with same normalized path
save_metadata(normalized_path, "Test caption", ["test", "upload"])
set_image_origin(normalized_path, uploaded_by="admin", origin_pdf=None, uploaded_at=uploaded_at)

# Check database
conn = get_connection()
cur = conn.cursor()

cur.execute("SELECT id, path FROM images WHERE path LIKE 'images/test_%'")
rows = cur.fetchall()

print(f"\n5. Database check:")
print(f"   Total rows in images table: {len(rows)}")
for row in rows:
    print(f"     ID {row[0]}: '{row[1]}'")

if len(rows) == 1:
    print("\n✅ SUCCESS: Only 1 row created (no duplicate!)")
else:
    print(f"\n❌ FAILED: {len(rows)} rows created (duplicate exists!)")

# Cleanup
cur.execute("DELETE FROM images WHERE path LIKE 'images/test_%'")
conn.commit()
cur.close()
conn.close()
