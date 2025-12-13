"""
Wipe database metadata and purge filesystem contents for images and catalogs.

Usage:
  python wipe_all.py --force        # no prompt
  python wipe_all.py                # interactive confirmation
"""
import sys
from pathlib import Path
from shutil import rmtree

from db_cleanup import clear_all_metadata

TARGET_DIRS = [
    Path("images"),
    Path("catalog"),
]

INDEX_DIRS = [
    Path("metadata-files/clip-vit-b-16"),
    Path("metadata-files/clip-vit-b-32"),
    Path("metadata-files/clip-vit-l-14"),
    Path("metadata-files/clip-vit-g-14"),
]

def purge_dirs():
    removed = []
    for d in TARGET_DIRS:
        if d.exists():
            # Remove contents but keep the folder
            for p in d.iterdir():
                try:
                    if p.is_dir():
                        rmtree(p)
                    else:
                        p.unlink()
                    removed.append(str(p))
                except Exception as e:
                    print(f"✗ Failed to remove {p}: {e}")
        else:
            # Create empty dir for consistency
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"✗ Failed to create directory {d}: {e}")
    print(f"✓ Purged contents of: {', '.join(str(d) for d in TARGET_DIRS)}")
    return removed

def clear_indexes():
    """Clear CLIP index files from metadata directories"""
    removed = []
    index_files = ["paths.npz", "features.npy", "image_features.index", "metadata.json"]
    
    for idx_dir in INDEX_DIRS:
        if not idx_dir.exists():
            continue
        for fname in index_files:
            fpath = idx_dir / fname
            if fpath.exists():
                try:
                    fpath.unlink()
                    removed.append(str(fpath))
                except Exception as e:
                    print(f"✗ Failed to remove {fpath}: {e}")
    
    if removed:
        print(f"✓ Cleared {len(removed)} index files")
    else:
        print("✓ No index files to clear")
    return removed

def main():
    force = "--force" in sys.argv
    if not force:
        print("⚠ WARNING: This will DELETE all DB metadata, CLIP indexes, and files in 'images/' and 'catalog/'.")
        confirm = input("Type 'yes' to proceed: ")
        if confirm.strip().lower() != "yes":
            print("Cancelled.")
            return

    ok = clear_all_metadata()
    if not ok:
        print("✗ Database wipe failed. Aborting filesystem purge.")
        return
    purge_dirs()
    clear_indexes()
    print("✓ Wipe completed.")

if __name__ == "__main__":
    main()
