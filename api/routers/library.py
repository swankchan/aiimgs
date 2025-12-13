"""
Library router
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from pathlib import Path
from typing import Optional

from ..models.schemas import LibraryResponse, LibraryItem
from ..services.clip_service import ClipService
from ..main import get_clip_service
from ..routers.auth import get_current_user
from db_helper import get_all_metadata

router = APIRouter()

@router.get("/", response_model=LibraryResponse)
async def get_library(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    filter_type: Optional[str] = Query(None, regex="^(all|images|pdfs)$"),
    current_user: dict = Depends(get_current_user),
    clip_service: ClipService = Depends(get_clip_service)
):
    """Get library items with pagination"""
    try:
        # Get all indexed images
        indexed_paths, _ = clip_service.load_index()
        
        # Get all PDFs
        catalog_folder = Path("catalog")
        pdf_files = []
        if catalog_folder.exists():
            pdf_files = list(catalog_folder.glob("*.pdf"))
        
        # Build items list
        all_items = []
        
        # Add images
        if filter_type is None or filter_type == "all" or filter_type == "images":
            metadata = get_all_metadata()
            for path in indexed_paths:
                # Normalize path for metadata lookup (convert to relative with forward slashes)
                normalized_path = str(path).replace('\\', '/')
                if normalized_path.startswith('C:/AIIMGS/') or normalized_path.startswith('c:/AIIMGS/'):
                    normalized_path = normalized_path.split('AIIMGS/', 1)[1]
                
                meta = metadata.get(normalized_path, {})
                all_items.append(LibraryItem(
                    path=normalized_path,
                    type="image",
                    name=Path(path).name,
                    caption=meta.get("caption", ""),
                    keywords=meta.get("keywords", []),
                    thumbnail_url=f"/images/{Path(path).name}" if "images" in normalized_path else None
                ))
        
        # Add PDFs
        if filter_type is None or filter_type == "all" or filter_type == "pdfs":
            for pdf_path in pdf_files:
                # Look for thumbnail with _preview suffix
                pdf_stem = pdf_path.stem
                thumbnail_path = catalog_folder / f"{pdf_stem}_preview.jpg"
                thumbnail_url = f"/catalog/{pdf_stem}_preview.jpg" if thumbnail_path.exists() else f"/catalog/{pdf_path.name}"
                
                all_items.append(LibraryItem(
                    path=str(pdf_path),
                    type="pdf",
                    name=pdf_path.name,
                    caption="",
                    keywords=[],
                    thumbnail_url=thumbnail_url
                ))
        
        # Pagination
        total_items = len(all_items)
        total_pages = (total_items + per_page - 1) // per_page
        
        # Ensure page is valid
        if page > total_pages and total_pages > 0:
            page = total_pages
        
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total_items)
        page_items = all_items[start_idx:end_idx]
        
        return LibraryResponse(
            items=page_items,
            total=total_items,
            page=page,
            per_page=per_page,
            total_pages=total_pages
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get library: {str(e)}")

@router.get("/folders")
async def list_folders(
    current_user: dict = Depends(get_current_user)
):
    """List available folders for indexing"""
    try:
        base_path = Path(".")
        folders = [str(p) for p in base_path.iterdir() if p.is_dir() and not p.name.startswith(".")]
        return {"folders": sorted(folders)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list folders: {str(e)}")
