"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Authentication
class UserLogin(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = ""
    email: Optional[str] = ""
    is_admin: bool = False

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    is_admin: Optional[bool] = None
    password: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    username: str
    full_name: Optional[str] = ""
    email: Optional[str] = ""
    is_admin: bool
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

# Search
class SearchQuery(BaseModel):
    query: str
    top_k: int = Field(default=32, ge=1, le=100)

class SearchResult(BaseModel):
    path: str
    score: float
    caption: Optional[str] = ""
    keywords: List[str] = []
    thumbnail_url: Optional[str] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int
    duration: float

# Metadata
class ImageMetadata(BaseModel):
    path: str
    caption: str = ""
    keywords: List[str] = []

class MetadataUpdate(BaseModel):
    caption: Optional[str] = None
    keywords: Optional[List[str]] = None

# Indexing
class IndexRequest(BaseModel):
    folder_path: str
    model_name: str = "clip-vit-b-32"

class IndexResponse(BaseModel):
    success: bool
    message: str
    images_indexed: int
    duration: float
    model_name: str
    folder_path: str

class IndexStats(BaseModel):
    total_images: int
    total_pdfs: int
    index_size: int
    last_updated: Optional[datetime] = None

class SyncRequest(BaseModel):
    folders: List[str]

class SyncResponse(BaseModel):
    success: bool
    message: str
    added: int
    removed: int
    total: int

# PDF Processing
class PDFUploadResponse(BaseModel):
    success: bool
    message: str
    pdf_path: str
    images_extracted: int
    image_paths: List[str]
    suggested_keywords: List[str]
    keywords_by_method: Optional[Dict[str, List[str]]] = None  # NEW: keywords grouped by extraction method
    ai_info: Optional[Dict[str, Any]] = None
    smart_caption: Optional[str] = None

class SavePDFMetadataRequest(BaseModel):
    image_paths: List[str]
    caption: str
    keywords: List[str]
    pdf_filename: str

# Library
class LibraryItem(BaseModel):
    path: str
    type: str  # "image" or "pdf"
    name: str
    caption: Optional[str] = ""
    keywords: List[str] = []
    thumbnail_url: Optional[str] = None

class LibraryResponse(BaseModel):
    items: List[LibraryItem]
    total: int
    page: int
    per_page: int
    total_pages: int

# General responses
class MessageResponse(BaseModel):
    message: str
    success: bool = True

class ErrorResponse(BaseModel):
    detail: str
    error: bool = True
