#!/usr/bin/env python3
"""Convert absolute paths in metadata.json to relative paths."""

import json
from pathlib import Path
import os

METADATA_FILE = Path("metadata-files") / "clip-vit-b-32" / "metadata.json"
IMAGES_DIR = Path("images")

# Read current metadata with absolute paths
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Convert absolute paths to relative paths
new_metadata = {}
for abs_path, data in metadata.items():
    p = Path(abs_path)
    # Try to find relative path from images folder
    try:
        # Get just the filename
        rel_path = str(IMAGES_DIR / p.name)
    except:
        rel_path = abs_path
    
    new_metadata[rel_path] = data

# Write back with relative paths
with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(new_metadata, f, ensure_ascii=False, indent=2)

print(f"Converted {len(new_metadata)} paths to relative paths in {METADATA_FILE}")
for k in list(new_metadata.keys())[:3]:
    print(f"  Example: {k}")
