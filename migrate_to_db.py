"""
Migrate existing metadata.json to PostgreSQL database
"""
import json
from pathlib import Path
from db_helper import bulk_save_metadata, get_stats

def migrate_metadata_to_db():
    """Migrate metadata from JSON file to database"""
    print("=" * 60)
    print("Migrating Metadata to Database")
    print("=" * 60)
    
    # Find metadata file
    metadata_path = Path("metadata-files/clip-vit-b-32/metadata.json")
    
    if not metadata_path.exists():
        print(f"✗ Metadata file not found: {metadata_path}")
        return False
    
    try:
        # Load existing metadata
        print(f"\n📁 Loading {metadata_path}...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"✓ Loaded {len(metadata)} entries from JSON")
        
        # Convert relative paths to absolute
        print("\n🔄 Converting paths...")
        normalized_metadata = {}
        for rel_path, data in metadata.items():
            try:
                abs_path = str(Path(rel_path).resolve())
                normalized_metadata[abs_path] = data
            except Exception as e:
                print(f"⚠ Warning: Could not normalize {rel_path}: {e}")
                normalized_metadata[rel_path] = data
        
        print(f"✓ Normalized {len(normalized_metadata)} paths")
        
        # Save to database
        print("\n💾 Saving to database...")
        success, errors = bulk_save_metadata(normalized_metadata)
        
        print(f"\n✓ Successfully saved: {success}")
        if errors > 0:
            print(f"✗ Errors: {errors}")
        
        # Show database stats
        print("\n📊 Database Statistics:")
        stats = get_stats()
        print(f"   Total images: {stats['total_images']}")
        print(f"   Total keywords: {stats['total_keywords']}")
        print(f"   Unique keywords: {stats['unique_keywords']}")
        
        # Create backup
        backup_path = metadata_path.parent / f"metadata_backup_{Path(metadata_path).stem}.json"
        print(f"\n💾 Creating backup: {backup_path}")
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"✓ Backup created")
        
        print("\n" + "=" * 60)
        print("✅ Migration Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. The original metadata.json is backed up")
        print("2. Data is now in PostgreSQL database")
        print("3. You can now run the app with database backend")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Migration failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🚀 Starting metadata migration...\n")
    success = migrate_metadata_to_db()
    
    if success:
        print("\n✅ You can now use the database-backed application!")
    else:
        print("\n❌ Migration failed. Please fix errors and try again.")
