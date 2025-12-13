"""
Metadata router
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict
from pathlib import Path

from ..models.schemas import ImageMetadata, MetadataUpdate, MessageResponse
from ..routers.auth import get_current_user
from db_helper import get_all_metadata, get_metadata, save_metadata, delete_metadata

router = APIRouter()

def normalize_path(path_str: str) -> str:
    """
    Normalize path to relative format with forward slashes (matching database format)
    Converts: C:\\AIIMGS\\images\\0.jpg -> images/0.jpg
    """
    p = Path(path_str)
    if p.is_absolute():
        try:
            # Try to make relative to workspace
            workspace = Path(r'C:\AIIMGS')
            if workspace in p.parents or workspace == p.parent.parent:
                relative = p.relative_to(workspace)
                return str(relative).replace('\\', '/')
        except:
            pass
    # Already relative or couldn't convert - just normalize slashes
    return str(p).replace('\\', '/')

@router.get("/all", response_model=Dict[str, Dict])
async def get_all(
    current_user: dict = Depends(get_current_user)
):
    """Get all metadata"""
    try:
        metadata = get_all_metadata()
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")

@router.get("/{path:path}", response_model=ImageMetadata)
async def get_single_metadata(
    path: str,
    current_user: dict = Depends(get_current_user)
):
    """Get metadata for single image"""
    try:
        # Normalize path to match database format
        normalized_path = normalize_path(path)
        metadata = get_metadata(normalized_path)
        if metadata is None:
            return ImageMetadata(path=path, caption="", keywords=[])
        
        return ImageMetadata(
            path=path,
            caption=metadata.get("caption", ""),
            keywords=metadata.get("keywords", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")

@router.post("/", response_model=MessageResponse)
async def save_single_metadata(
    metadata: ImageMetadata,
    current_user: dict = Depends(get_current_user)
):
    """Save metadata for single image"""
    try:
        # Normalize path to match database format
        normalized_path = normalize_path(metadata.path)
        success = save_metadata(normalized_path, metadata.caption, metadata.keywords)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save metadata")
        
        return MessageResponse(message="Metadata saved successfully", success=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save metadata: {str(e)}")

@router.patch("/{path:path}", response_model=MessageResponse)
async def update_metadata(
    path: str,
    update: MetadataUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update metadata for single image"""
    try:
        # Normalize path to match database format
        normalized_path = normalize_path(path)
        
        # Get existing metadata
        existing = get_metadata(normalized_path)
        if existing is None:
            existing = {"caption": "", "keywords": []}
        
        # Update fields - allow empty values to clear the fields
        if update.caption is not None:
            caption = update.caption
        else:
            caption = existing.get("caption", "")

        # Preserve system tags (read-only): origin_user:, origin_pdf:, uploaded_at:
        existing_keywords = existing.get("keywords", []) or []
        system_prefixes = ("origin_user:", "origin_pdf:", "uploaded_at:")
        system_tags = [k for k in existing_keywords if any(k.startswith(pref) for pref in system_prefixes)]

        if update.keywords is not None:
            # Remove any system-like tags from user input and re-append preserved system tags
            user_keywords = [k for k in (update.keywords or []) if not any(str(k).startswith(pref) for pref in system_prefixes)]
            keywords = user_keywords + system_tags
        else:
            keywords = existing_keywords
        
        success = save_metadata(normalized_path, caption, keywords)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update metadata")
        
        return MessageResponse(message="Metadata updated successfully", success=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update metadata: {str(e)}")

@router.delete("/{path:path}", response_model=MessageResponse)
async def delete_single_metadata(
    path: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete metadata for single image"""
    try:
        # Normalize path to match database format
        normalized_path = normalize_path(path)
        delete_metadata([normalized_path])
        return MessageResponse(message="Metadata deleted successfully", success=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete metadata: {str(e)}")
