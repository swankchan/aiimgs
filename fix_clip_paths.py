"""
Fix CLIP index paths from absolute to relative format
"""
import numpy as np
from pathlib import Path

def normalize_path(path):
    """Convert to relative format with forward slashes"""
    path_str = str(path).replace('\\', '/')
    if path_str.startswith('C:/AIIMGS/') or path_str.startswith('c:/AIIMGS/'):
        path_str = path_str.split('AIIMGS/', 1)[1]
    return path_str

# Fix ViT-B-16 index
index_dir = Path("metadata-files/clip-vit-b-16")
if index_dir.exists():
    paths_file = index_dir / "paths.npz"
    if paths_file.exists():
        print("Loading existing index...")
        data = np.load(str(paths_file), allow_pickle=True)
        old_paths = data['paths']
        
        print(f"\nFound {len(old_paths)} paths")
        print("\nBefore normalization (first 3):")
        for p in old_paths[:3]:
            print(f"  {p}")
        
        # Normalize all paths
        new_paths = np.array([normalize_path(p) for p in old_paths])
        
        print("\nAfter normalization (first 3):")
        for p in new_paths[:3]:
            print(f"  {p}")
        
        # Save updated paths
        np.savez_compressed(str(paths_file), paths=new_paths)
        print(f"\n✓ Updated {len(new_paths)} paths in index")
    else:
        print("No paths.npz found")
else:
    print("Index directory doesn't exist")
