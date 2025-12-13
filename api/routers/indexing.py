"""
Indexing router
"""
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pathlib import Path
from typing import List
import shutil

from ..models.schemas import (
    SyncRequest, SyncResponse, IndexStats, MessageResponse, 
    PDFUploadResponse, IndexRequest, IndexResponse, SavePDFMetadataRequest
)
from ..services.clip_service import ClipService
from ..main import get_clip_service
from ..routers.auth import get_current_user
from db_helper import get_all_metadata, save_metadata, delete_metadata, get_stats

try:
    from pdf_utils import (
        extract_images_from_pdf,
        extract_text_from_pdf,
        extract_keywords_from_text,
        analyze_pdf_with_ai,
        generate_smart_caption,
        save_pdf_to_catalog,
        PDF_SUPPORT
    )
except ImportError:
    PDF_SUPPORT = False

router = APIRouter()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def normalize_path_for_db(path):
    """Convert path to relative format with forward slashes for consistent DB storage"""
    path_obj = Path(path)
    try:
        # Try to make relative to C:\AIIMGS
        rel = path_obj.relative_to(Path.cwd())
        return str(rel).replace('\\', '/')
    except ValueError:
        # If not under cwd, just use as-is with forward slashes
        return str(path).replace('\\', '/')

def list_image_paths(folders: List[str]) -> List[Path]:
    """List all image files in folders"""
    paths = []
    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            continue
        for path in folder_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                paths.append(path.resolve())
    return sorted(set(paths))

@router.post("/index", response_model=IndexResponse)
async def index_folder(
    request: IndexRequest,
    current_user: dict = Depends(get_current_user),
    clip_service: ClipService = Depends(get_clip_service)
):
    """Index images in a folder (incremental - only new/changed files)"""
    try:
        folder = Path(request.folder_path)
        if not folder.exists():
            raise HTTPException(status_code=404, detail=f"Folder not found: {request.folder_path}")
        
        # Get all images in folder
        image_paths = list_image_paths([str(folder)])
        
        if not image_paths:
            raise HTTPException(status_code=404, detail="No images found in folder")
        
        # Load existing index
        existing_paths, existing_features = clip_service.load_index()
        existing_set = set(existing_paths)
        
        # Convert to strings
        image_paths_str = [str(p) for p in image_paths]
        current_set = set(image_paths_str)
        
        # Find new and removed images
        new_images = [p for p in image_paths_str if p not in existing_set]
        removed_images = [p for p in existing_paths if p not in current_set]
        kept_images = [p for p in existing_paths if p in current_set]
        
        start_time = __import__('time').time()
        
        # Build new index
        final_paths = []
        final_features = []
        
        # Keep existing features for unchanged images
        existing_map = {path: existing_features[idx] for idx, path in enumerate(existing_paths)}
        for path in kept_images:
            final_paths.append(path)
            final_features.append(existing_map[path])
        
        # Encode only new images
        new_count = 0
        if new_images:
            new_features = clip_service.encode_image_batch(new_images)
            final_paths.extend(new_images)
            final_features.extend(list(new_features))
            new_count = len(new_images)
            
            # Initialize metadata for new images
            from db_helper import get_all_metadata, save_metadata
            existing_meta = get_all_metadata()
            for path in new_images:
                normalized = normalize_path_for_db(path)
                if normalized not in existing_meta:
                    save_metadata(normalized, "", [])
        
        # Remove metadata for deleted images
        if removed_images:
            from db_helper import delete_metadata
            delete_metadata(removed_images)
        
        # Save updated index
        if final_paths:
            import numpy as np
            final_features_array = np.vstack(final_features) if final_features else np.array([])
            clip_service.create_index(final_paths, final_features_array)
        
        duration = __import__('time').time() - start_time
        
        message_parts = []
        if new_count > 0:
            message_parts.append(f"{new_count} new images added")
        if removed_images:
            message_parts.append(f"{len(removed_images)} images removed")
        if not new_count and not removed_images:
            message_parts.append("No changes detected")
        
        return IndexResponse(
            success=True,
            message=f"Indexing complete: {', '.join(message_parts)}. Total: {len(final_paths)} images",
            images_indexed=new_count,
            duration=duration,
            model_name=request.model_name,
            folder_path=str(folder)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

@router.post("/sync", response_model=SyncResponse)
async def sync_folders(
    sync_request: SyncRequest,
    current_user: dict = Depends(get_current_user),
    clip_service: ClipService = Depends(get_clip_service)
):
    """Sync image folders to index"""
    try:
        # Get all images from folders
        folder_paths = [Path(f) for f in sync_request.folders]
        image_paths = list_image_paths(sync_request.folders)
        
        if not image_paths:
            return SyncResponse(
                success=False,
                message="No images found in selected folders",
                added=0,
                removed=0,
                total=0
            )
        
        # Load existing index
        existing_paths, existing_features = clip_service.load_index()
        existing_map = {str(path): existing_features[idx] for idx, path in enumerate(existing_paths)}
        existing_meta = get_all_metadata()
        
        # Separate retained and new images
        retained_paths = []
        retained_features = []
        new_candidates = []
        
        for path in image_paths:
            path_str = str(path)
            if path_str in existing_map:
                retained_paths.append(path_str)
                retained_features.append(existing_map[path_str])
            else:
                new_candidates.append(path_str)
        
        # Encode new images
        new_count = 0
        if new_candidates:
            new_features = clip_service.encode_image_batch(new_candidates)
            retained_paths.extend(new_candidates[:len(new_features)])
            retained_features.extend(list(new_features))
            new_count = len(new_features)
            
            # Add default metadata for new images
            for path in new_candidates[:len(new_features)]:
                normalized = normalize_path_for_db(path)
                if normalized not in existing_meta:
                    save_metadata(normalized, "", [])
        
        # Create new index
        import numpy as np
        if retained_features:
            combined_features = np.vstack(retained_features)
        else:
            combined_features = np.empty((0, clip_service.embed_dim), dtype=np.float32)
        
        clip_service.create_index(retained_paths, combined_features)
        
        removed_count = max(len(existing_paths) - len(retained_paths), 0)
        
        return SyncResponse(
            success=True,
            message=f"Sync complete: {len(retained_paths)} images indexed",
            added=new_count,
            removed=removed_count,
            total=len(retained_paths)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

@router.post("/upload", response_model=MessageResponse)
async def upload_images(
    files: List[UploadFile] = File(...),
    caption: str = Form(default=""),
    keywords: str = Form(default=""),
    current_user: dict = Depends(get_current_user)
):
    """Upload new images"""
    try:
        images_folder = Path("images")
        images_folder.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        from datetime import datetime
        uploaded_at = datetime.utcnow().isoformat()
        # Parse keywords (comma-separated)
        user_keywords = [k.strip() for k in (keywords or "").split(",") if k.strip()]
        for file in files:
            suffix = Path(file.filename).suffix.lower()
            if suffix not in IMAGE_EXTS:
                continue
            
            stem = Path(file.filename).stem
            dest = images_folder / f"{stem}{suffix}"
            counter = 1
            while dest.exists():
                dest = images_folder / f"{stem}_{counter}{suffix}"
                counter += 1
            
            with open(dest, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            saved_count += 1
            # Normalize path to relative with forward slashes BEFORE any DB operations
            normalized_path = str(dest).replace('\\', '/')
            
            # Save user metadata (caption + keywords) to image_metadata table
            save_metadata(normalized_path, caption or "", user_keywords)
            # Persist origin fields in images table columns
            try:
                from db_helper import set_image_origin
                set_image_origin(normalized_path, uploaded_by=current_user.get('username','unknown'), origin_pdf=None, uploaded_at=uploaded_at)
            except Exception as e:
                print(f"Failed to set image origin fields: {e}")
        
        return MessageResponse(
            message=f"Uploaded {saved_count} images. Indexing will start automatically",
            success=True
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    use_ai: bool = False,
    caption: str = Form(default=""),
    keywords: str = Form(default=""),
    current_user: dict = Depends(get_current_user)
):
    """Upload and process PDF file - Extract images, analyze with 4 methods, generate metadata"""
    if not PDF_SUPPORT:
        raise HTTPException(status_code=501, detail="PDF support not available")
    
    try:
        from io import BytesIO
        import json
        
        # Read PDF into BytesIO
        pdf_bytes = await file.read()
        pdf_filename = file.filename or "uploaded.pdf"
        
        # Save PDF to catalog
        catalog_folder = Path("catalog")
        pdf_buffer = BytesIO(pdf_bytes)
        pdf_path = save_pdf_to_catalog(pdf_buffer, catalog_folder, pdf_filename)
        
        # Extract images
        pdf_buffer = BytesIO(pdf_bytes)
        images_folder = Path("images")
        image_paths, pdf_name = extract_images_from_pdf(
            pdf_buffer, 
            images_folder,
            pdf_filename=pdf_filename
        )
        
        # Extract text for analysis
        pdf_buffer = BytesIO(pdf_bytes)
        extracted_text = extract_text_from_pdf(pdf_buffer)
        
        # Method 1: AI analysis (if enabled and text available)
        ai_info = None
        smart_caption = None
        print(f"=== AI Analysis Debug ===")
        print(f"use_ai: {use_ai}")
        print(f"extracted_text length: {len(extracted_text.strip())}")
        
        if use_ai and extracted_text.strip():
            try:
                config = json.load(open("config.json"))
                ai_config = config.get("pdf", {}).get("ai_analysis", {})
                
                print(f"AI Config: {ai_config}")
                print(f"Calling Ollama with model: {ai_config.get('model')}")
                
                ai_info = analyze_pdf_with_ai(
                    extracted_text,
                    custom_fields=ai_config.get("fields", []),
                    llm_url=ai_config.get("ollama_url", "http://localhost:11434"),
                    model=ai_config.get("model", "llama3.2:3b")
                )
                
                print(f"AI Info result: {ai_info}")
                
                if "error" not in ai_info:
                    smart_caption = generate_smart_caption(
                        ai_info,
                        template=ai_config.get("caption_template", "{project_name}")
                    )
                    print(f"Smart caption: {smart_caption}")
                else:
                    print(f"AI returned error: {ai_info.get('error')}")
            except Exception as e:
                print(f"AI analysis exception: {e}")
                import traceback
                traceback.print_exc()
                ai_info = {"error": str(e)}
        else:
            print(f"Skipping AI: use_ai={use_ai}, text_length={len(extracted_text.strip())}")
        
        # Methods 2-4: Pattern-based, Capitalized, Frequency analysis
        config = json.load(open("config.json"))
        max_keywords = config.get("pdf", {}).get("max_keywords", 5)
        suggested_keywords = extract_keywords_from_text(
            extracted_text,
            max_keywords=max_keywords,
            ai_info=ai_info if ai_info and "error" not in ai_info else None
        )
        
        # NEW: Get keywords grouped by method for frontend display
        from pdf_utils import extract_keywords_by_method
        keywords_by_method = extract_keywords_by_method(
            extracted_text,
            max_keywords=max_keywords,
            ai_info=ai_info if ai_info and "error" not in ai_info else None
        )
        
        # DO NOT auto-save metadata - let user review and edit first
        # The frontend will handle metadata saving after user confirmation

        return PDFUploadResponse(
            success=True,
            message=f"Extracted {len(image_paths)} images from PDF",
            pdf_path=str(pdf_path),
            images_extracted=len(image_paths),
            image_paths=[str(p) for p in image_paths],
            suggested_keywords=suggested_keywords,
            keywords_by_method=keywords_by_method,
            ai_info=ai_info if ai_info and "error" not in ai_info else None,
            smart_caption=smart_caption
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@router.post("/save-pdf-metadata")
async def save_pdf_metadata(
    request: SavePDFMetadataRequest,
    current_user: dict = Depends(get_current_user)
):
    """Save metadata for PDF-extracted images after user review"""
    image_paths = request.image_paths
    caption = request.caption
    keywords = request.keywords
    pdf_filename = request.pdf_filename
    try:
        from datetime import datetime
        uploaded_at = datetime.utcnow().isoformat()
        
        for path in image_paths:
            # Normalize path
            normalized_path = str(path).replace('\\', '/')
            if normalized_path.startswith('C:/AIIMGS/') or normalized_path.startswith('c:/AIIMGS/'):
                normalized_path = normalized_path.split('AIIMGS/', 1)[1]
            
            # Save metadata
            save_metadata(normalized_path, caption, keywords)
            
            # Set origin
            try:
                from db_helper import set_image_origin
                set_image_origin(
                    normalized_path,
                    uploaded_by=current_user.get('username', 'unknown'),
                    origin_pdf=pdf_filename,
                    uploaded_at=uploaded_at
                )
            except Exception as e:
                print(f"Failed to set image origin: {e}")
        
        return MessageResponse(
            success=True,
            message=f"Saved metadata for {len(image_paths)} images"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save metadata: {str(e)}")

@router.delete("/remove")
async def remove_images(
    paths: List[str],
    delete_files: bool = False,
    current_user: dict = Depends(get_current_user),
    clip_service: ClipService = Depends(get_clip_service)
):
    """Remove images from index"""
    try:
        if not paths:
            return MessageResponse(message="No images to remove", success=True)
        
        # Load existing index
        existing_paths, existing_features = clip_service.load_index()
        
        # Filter out removed paths
        remove_set = set(paths)
        keep_indices = [idx for idx, path in enumerate(existing_paths) if path not in remove_set]
        new_paths = [existing_paths[idx] for idx in keep_indices]
        
        import numpy as np
        if keep_indices:
            new_features = existing_features[keep_indices]
        else:
            new_features = np.empty((0, clip_service.embed_dim), dtype=np.float32)
        
        # Delete metadata from database
        delete_metadata(paths)
        
        # Create new index
        clip_service.create_index(new_paths, new_features)
        
        # Delete files if requested
        if delete_files:
            for path in paths:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception as e:
                    print(f"Failed to delete {path}: {e}")
        
        return MessageResponse(
            message=f"Removed {len(paths)} images from index",
            success=True
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Remove failed: {str(e)}")

@router.get("/stats", response_model=IndexStats)
async def get_index_stats(
    current_user: dict = Depends(get_current_user),
    clip_service: ClipService = Depends(get_clip_service)
):
    """Get index statistics"""
    try:
        paths, features = clip_service.load_index()
        
        # Count PDFs
        catalog_folder = Path("catalog")
        pdf_count = len(list(catalog_folder.glob("*.pdf"))) if catalog_folder.exists() else 0
        
        # Get database stats
        db_stats = get_stats()
        
        return IndexStats(
            total_images=len(paths),
            total_pdfs=pdf_count,
            index_size=len(paths),
            last_updated=db_stats.get("last_updated")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
