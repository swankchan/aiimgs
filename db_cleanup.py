"""
Database cleanup and maintenance functions

Added features:
- dry-run support (preview deletions without committing)
- `clear-pdfs` to remove only PDF entries
- `clear-metadata` to remove only `image_metadata` rows
- `counts` to show current row counts for key tables
"""
from db_config import get_connection
from pathlib import Path
from typing import List, Dict
import os
import json
import shutil


def load_config_folders() -> Dict[str, str]:
    """Load folder settings from config.json with sensible defaults."""
    cfg_path = Path("config.json")
    defaults = {"images": "images", "pdf_catalog": "catalog", "metadata": "metadata-files"}
    if not cfg_path.exists():
        return defaults
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            folders = cfg.get("folders", {})
            return {
                "images": folders.get("images", defaults["images"]),
                "pdf_catalog": folders.get("pdf_catalog", defaults["pdf_catalog"]),
                "metadata": folders.get("metadata", defaults["metadata"]),
            }
    except Exception:
        return defaults


def _delete_path(path: str, dry_run: bool = False) -> bool:
    """Delete a file or directory. Returns True if deleted (or would be deleted in dry-run)."""
    p = Path(path)
    if not p.exists():
        return False
    if dry_run:
        print(f"[dry] Would remove: {path}")
        return True
    try:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"Removed: {path}")
        return True
    except Exception as e:
        print(f"Failed to remove {path}: {e}")
        return False

def clear_all_metadata(dry_run: bool = False, remove_files: bool = False):
    """
    Clear all metadata from database (images, image_metadata and pdfs tables)
    Use this when you want to start fresh or when changing image folders
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Get counts before deletion for reporting
        cursor.execute("SELECT COUNT(*) FROM image_metadata")
        metadata_count_before = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM pdfs")
        pdf_count_before = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM images")
        image_count_before = cursor.fetchone()[0]

        print("Current counts:")
        print(f"  images: {image_count_before}")
        print(f"  image_metadata: {metadata_count_before}")
        print(f"  pdfs: {pdf_count_before}")

        if dry_run:
            print("\nDry run mode: no changes will be committed.")
            cursor.close()
            conn.close()
            return True

        # If requested, collect file paths before deleting DB rows
        folders = load_config_folders()
        images_paths: List[str] = []
        pdf_paths: List[str] = []
        metadata_dir = folders.get("metadata")

        if remove_files:
            # Collect image paths
            cursor.execute("SELECT path FROM images")
            images_paths = [r[0] for r in cursor.fetchall() if r and r[0]]
            # Collect pdf paths
            cursor.execute("SELECT path FROM pdfs")
            pdf_paths = [r[0] for r in cursor.fetchall() if r and r[0]]

        # Delete all image metadata
        cursor.execute("DELETE FROM image_metadata")
        metadata_deleted = cursor.rowcount

        # Delete all pdf entries
        cursor.execute("DELETE FROM pdfs")
        pdfs_deleted = cursor.rowcount

        # Delete all images (will cascade to image_metadata if FK is present)
        cursor.execute("DELETE FROM images")
        images_deleted = cursor.rowcount

        # Remove files and metadata index folders if requested
        if remove_files:
            for p in images_paths:
                _delete_path(p, dry_run=dry_run)
            for p in pdf_paths:
                _delete_path(p, dry_run=dry_run)
            # Remove metadata index folder
            md_path = Path(metadata_dir)
            if md_path.exists():
                _delete_path(str(md_path), dry_run=dry_run)

        # Reset sequences (auto-increment IDs) -- ignore errors if sequences missing
        try:
            cursor.execute("ALTER SEQUENCE images_id_seq RESTART WITH 1")
        except Exception:
            pass
        try:
            cursor.execute("ALTER SEQUENCE image_metadata_id_seq RESTART WITH 1")
        except Exception:
            pass
        try:
            cursor.execute("ALTER SEQUENCE pdfs_id_seq RESTART WITH 1")
        except Exception:
            pass

        conn.commit()
        cursor.close()
        conn.close()

        print(f"✓ Cleared {images_deleted} images, {metadata_deleted} metadata rows and {pdfs_deleted} pdf entries from database")
        return True

    except Exception as e:
        print(f"✗ Error clearing database: {e}")
        if conn is not None:
            conn.rollback()
        return False

def remove_missing_images(dry_run: bool = False):
    """
    Remove metadata for images that no longer exist on disk.
    Image metadata (in `image_metadata`) will be removed via cascade when images are deleted.
    Useful for cleaning up after moving/deleting image files
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Get all image paths from database
        cursor.execute("SELECT id, path FROM images")
        all_images = cursor.fetchall()

        missing_ids: List[int] = []
        for img_id, path in all_images:
            if not Path(path).exists():
                missing_ids.append(img_id)

        if missing_ids:
            print(f"Found {len(missing_ids)} missing images. IDs: {missing_ids}")
            if dry_run:
                print("Dry run: no deletions performed.")
            else:
                # Before deleting DB rows, attempt to remove any lingering files for these IDs
                try:
                    cursor.execute(f"SELECT path FROM images WHERE id = ANY(%s)", (missing_ids,))
                    paths_to_remove = [r[0] for r in cursor.fetchall() if r and r[0]]
                    for p in paths_to_remove:
                        if Path(p).exists():
                            _delete_path(p, dry_run=False)
                except Exception:
                    pass

                # Delete missing images (image_metadata will cascade via FK)
                cursor.execute(f"DELETE FROM images WHERE id = ANY(%s)", (missing_ids,))
                deleted_count = cursor.rowcount
                conn.commit()
                print(f"✓ Removed {deleted_count} missing images from database")
        else:
            print("✓ No missing images found")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"✗ Error removing missing images: {e}")
        if conn is not None:
            conn.rollback()
        return False

def sync_db_with_index(remove_files: bool = False, dry_run: bool = False):
    """
    Remove database entries that are not in FAISS index
    Call this after rebuilding index to keep DB in sync
    """
    try:
        from pathlib import Path
        import numpy as np
        
        # Load FAISS index paths
        paths_file = Path("metadata-files/clip-vit-l-14/paths.npz")
        if not paths_file.exists():
            # Try old model
            paths_file = Path("metadata-files/clip-vit-b-32/paths.npz")
        
        if not paths_file.exists():
            print("⚠ No index file found. Build index first.")
            return False
        
        indexed_paths = set(np.load(paths_file, allow_pickle=True)["paths"].tolist())
        
        # Get all DB paths
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, path FROM images")
        all_db_images = cursor.fetchall()
        
        # Find paths in DB but not in index
        to_remove = []
        for img_id, path in all_db_images:
            if path not in indexed_paths:
                to_remove.append(img_id)
        
        if to_remove:
            # If requested, collect file paths for these ids
            files_to_remove: List[str] = []
            if remove_files:
                try:
                    cursor.execute(f"SELECT path FROM images WHERE id = ANY(%s)", (to_remove,))
                    files_to_remove = [r[0] for r in cursor.fetchall() if r and r[0]]
                except Exception:
                    files_to_remove = []

            # Delete DB entries
            cursor.execute(f"DELETE FROM images WHERE id = ANY(%s)", (to_remove,))
            deleted_count = cursor.rowcount
            conn.commit()

            # Remove files if requested
            if remove_files and files_to_remove:
                for fp in files_to_remove:
                    _delete_path(fp, dry_run=dry_run)

            print(f"✓ Removed {deleted_count} orphaned entries from database")
        else:
            print("✓ Database is in sync with index")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Error syncing database: {e}")
        if 'conn' in locals():
            conn.rollback()
        return False


def get_table_counts() -> Dict[str, int]:
    """Return counts for key tables: images, image_metadata, pdfs"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM images")
        images_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM image_metadata")
        metadata_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM pdfs")
        pdfs_count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return {"images": images_count, "image_metadata": metadata_count, "pdfs": pdfs_count}
    except Exception:
        return {"images": 0, "image_metadata": 0, "pdfs": 0}

if __name__ == "__main__":
    import sys

    def usage():
        print("Database Cleanup Utilities")
        print("\nUsage:")
        print("  python db_cleanup.py clear [dry]         - Clear all metadata (images, metadata, pdfs). Add 'dry' to preview")
        print("  python db_cleanup.py clear-pdfs [dry]    - Clear only pdf entries")
        print("  python db_cleanup.py clear-metadata [dry]- Clear only image_metadata table")
        print("  python db_cleanup.py missing [dry]       - Remove entries for missing image files (dry to preview)")
        print("  python db_cleanup.py sync                - Sync DB with FAISS index")
        print("  python db_cleanup.py counts              - Show current counts for tables")

    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    command = sys.argv[1].lower()
    args = [a.lower() for a in sys.argv[2:]]
    dry = any(a in ("dry", "--dry", "-n") for a in args)
    delete_files = any(a in ("files", "--files", "--delete-files", "delete-files", "--delete") for a in args)

    if command == "clear":
        print("⚠ WARNING: This will delete ALL metadata from the database!")
        if dry or delete_files:
            print("Running in dry-run mode (no changes will be made)") if dry else None
            print("Will remove files as requested.") if delete_files else None
            clear_all_metadata(dry_run=dry, remove_files=delete_files)
        else:
            confirm = input("Type 'yes' to confirm: ")
            if confirm.lower() == 'yes':
                clear_all_metadata(dry_run=False, remove_files=False)
            else:
                print("Cancelled.")

    elif command == "clear-pdfs":
        print("Clearing only PDF entries from database.")
        if dry:
            print("Dry run: showing counts only")
            counts = get_table_counts()
            print(counts)
            if delete_files:
                folders = load_config_folders()
                pdf_dir = folders.get("pdf_catalog")
                print(f"[dry] Would remove metadata folder: {folders.get('metadata')}")
                print(f"[dry] Would remove pdf folder: {pdf_dir}")
        else:
            try:
                conn = get_connection()
                cur = conn.cursor()
                # Collect paths if deleting files
                pdf_paths = []
                if delete_files:
                    cur.execute("SELECT path FROM pdfs")
                    pdf_paths = [r[0] for r in cur.fetchall() if r and r[0]]

                cur.execute("DELETE FROM pdfs")
                deleted = cur.rowcount
                try:
                    cur.execute("ALTER SEQUENCE pdfs_id_seq RESTART WITH 1")
                except Exception:
                    pass
                conn.commit()
                cur.close()
                conn.close()

                # Remove physical pdfs and metadata index if requested
                if delete_files and pdf_paths:
                    for p in pdf_paths:
                        _delete_path(p, dry_run=False)
                    # also remove metadata folder
                    md = load_config_folders().get("metadata")
                    _delete_path(md, dry_run=False)

                print(f"✓ Removed {deleted} pdf entries from database")
            except Exception as e:
                print(f"Error clearing pdfs: {e}")

    elif command == "clear-metadata":
        print("Clearing only image metadata table (image_metadata).")
        if dry:
            counts = get_table_counts()
            print(counts)
        else:
            try:
                conn = get_connection()
                cur = conn.cursor()
                cur.execute("DELETE FROM image_metadata")
                deleted = cur.rowcount
                try:
                    cur.execute("ALTER SEQUENCE image_metadata_id_seq RESTART WITH 1")
                except Exception:
                    pass
                conn.commit()
                cur.close()
                conn.close()
                print(f"✓ Removed {deleted} image_metadata rows from database")
            except Exception as e:
                print(f"Error clearing image_metadata: {e}")

    elif command == "missing":
        remove_missing_images(dry_run=dry)

    elif command == "sync":
        sync_db_with_index(remove_files=delete_files, dry_run=dry)

    elif command == "counts":
        counts = get_table_counts()
        print("Current counts:")
        for k, v in counts.items():
            print(f"  {k}: {v}")

    else:
        print(f"Unknown command: {command}")
        usage()
        sys.exit(1)
