"""Check what's in the CLIP index"""
import numpy as np

try:
    # Load paths from old index
    paths_data = np.load('metadata-files/clip-vit-b-16/paths.npz', allow_pickle=True)
    paths = paths_data['paths']
    
    print(f"Total images in CLIP index: {len(paths)}")
    print("\nFirst 5 image paths:")
    for i, path in enumerate(paths[:5]):
        print(f"  {i+1}. {path}")
    
    # Check if paths exist
    from pathlib import Path
    existing = sum(1 for p in paths if Path(p).exists())
    print(f"\nImages that still exist: {existing}/{len(paths)}")
    
except Exception as e:
    print(f"Error: {e}")
