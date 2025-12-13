"""
Search router
"""
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from time import perf_counter
from typing import Optional
from pathlib import Path as PathLib

from ..models.schemas import SearchQuery, SearchResponse, SearchResult
from ..services.clip_service import ClipService
from ..main import get_clip_service
from ..routers.auth import get_current_user
from db_helper import get_all_metadata

router = APIRouter()

@router.post("/text", response_model=SearchResponse)
async def search_by_text(
    query: SearchQuery,
    current_user: dict = Depends(get_current_user),
    clip_service: ClipService = Depends(get_clip_service)
):
    """Search images by text query"""
    start_time = perf_counter()
    
    try:
        # Encode text query
        query_vector = clip_service.encode_text(query.query)
        
        # Search similar images
        matches = clip_service.search_similar(query_vector, top_k=query.top_k)
        
        # Get metadata
        metadata = get_all_metadata()
        
        # Build results
        results = []
        for path, score in matches:
            meta = metadata.get(path, {})
            # Extract just the filename from the full path
            filename = PathLib(path).name
            results.append(SearchResult(
                path=path,
                score=score,
                caption=meta.get("caption", ""),
                keywords=meta.get("keywords", []),
                thumbnail_url=f"/images/{filename}" if 'images' in path.lower() else None
            ))
        
        duration = perf_counter() - start_time
        
        return SearchResponse(
            results=results,
            total=len(results),
            duration=duration
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = 32,
    current_user: dict = Depends(get_current_user),
    clip_service: ClipService = Depends(get_clip_service)
):
    """Search images by uploaded image"""
    start_time = perf_counter()
    
    try:
        # Read uploaded image
        image_bytes = await file.read()
        
        # Encode image
        query_vector = clip_service.encode_image_from_bytes(image_bytes)
        
        # Search similar images
        matches = clip_service.search_similar(query_vector, top_k=top_k)
        
        # Get metadata
        metadata = get_all_metadata()
        
        # Build results
        results = []
        from pathlib import Path as PathLib
        for path, score in matches:
            meta = metadata.get(path, {})
            # Extract filename from path (works with both / and \\ separators)
            filename = PathLib(path).name
            results.append(SearchResult(
                path=path,
                score=score,
                caption=meta.get("caption", ""),
                keywords=meta.get("keywords", []),
                thumbnail_url=f"/images/{filename}" if 'images' in path.lower() else None
            ))
        
        duration = perf_counter() - start_time
        
        return SearchResponse(
            results=results,
            total=len(results),
            duration=duration
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/stats")
async def get_search_stats(
    current_user: dict = Depends(get_current_user),
    clip_service: ClipService = Depends(get_clip_service)
):
    """Get search statistics"""
    try:
        paths, features = clip_service.load_index()
        metadata = get_all_metadata()
        
        return {
            "total_indexed": len(paths),
            "total_with_metadata": len(metadata),
            "embedding_dim": clip_service.embed_dim,
            "model": clip_service.model_name,
            "device": clip_service.device
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
