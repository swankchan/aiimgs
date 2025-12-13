"""
FastAPI backend for AI Image Search
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
from pathlib import Path

from .services.clip_service import ClipService

# Initialize CLIP service on startup
clip_service = None

def get_clip_service() -> ClipService:
    """Dependency to get CLIP service"""
    if clip_service is None:
        raise HTTPException(status_code=503, detail="CLIP service not initialized")
    return clip_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    global clip_service
    print("🚀 Starting AI Image Search API...")
    
    # Load CLIP model on startup
    clip_service = ClipService()
    await clip_service.initialize()
    
    yield
    
    print("👋 Shutting down AI Image Search API...")

# Create FastAPI app
app = FastAPI(
    title="AI Image Search API",
    description="CLIP-based image search with metadata management",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://localhost:8000",  # FastAPI dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
images_path = Path("images")
if images_path.exists():
    app.mount("/images", StaticFiles(directory="images"), name="images")

catalog_path = Path("catalog")
if catalog_path.exists():
    app.mount("/catalog", StaticFiles(directory="catalog"), name="catalog")

# Import routers after defining get_clip_service
from .routers import search, indexing, metadata, library, auth, admin, help

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(search.router, prefix="/api/search", tags=["Search"])
app.include_router(indexing.router, prefix="/api/indexing", tags=["Indexing"])
app.include_router(metadata.router, prefix="/api/metadata", tags=["Metadata"])
app.include_router(library.router, prefix="/api/library", tags=["Library"])
app.include_router(help.router, prefix="/api/help", tags=["Help"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Image Search API",
        "version": "2.0.0",
        "docs": "/docs"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "clip_loaded": clip_service is not None and clip_service.is_ready()
    }
