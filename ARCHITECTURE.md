# AI Imagery Search - Architecture Documentation

## Overview
AI Imagery Search is a web application for managing, searching, and analyzing images using AI models. It supports both direct image uploads and PDF extraction, with semantic search powered by CLIP embeddings and AI-generated captions.

## Technology Stack

### Frontend
- **Framework**: Next.js 14.2.33 (App Router)
- **Language**: TypeScript
- **UI Library**: React 18
- **Data Fetching**: React Query (TanStack Query)
- **State Management**: React hooks + localStorage for persistence
- **Styling**: Tailwind CSS

### Backend
- **Framework**: FastAPI
- **Language**: Python 3.11
- **Server**: Uvicorn
- **Environment**: Conda (aiimgs)
- **PDF Processing**: PyMuPDF (fitz)
- **AI Models**: 
  - CLIP (via sentence-transformers) for image embeddings
  - Ollama (llama3.2-vision) for image analysis

### Database
- **System**: PostgreSQL
- **ORM**: psycopg2 (raw SQL)
- **Vector Search**: FAISS (not pgvector)
- **Timezone**: UTC stored, convertible to HK time on display

## Database Schema

### Core Tables

#### `images`
```sql
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    uploaded_by TEXT NOT NULL,
    uploaded_at TIMESTAMP DEFAULT NOW(),
    origin_pdf TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```
- **Purpose**: File metadata and ownership
- **Key Field**: `origin_pdf` links images extracted from PDFs

#### `image_metadata`
```sql
CREATE TABLE image_metadata (
    id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES images(id) ON DELETE CASCADE,
    caption TEXT,
    keywords TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```
- **Purpose**: AI-generated content metadata
- **Separation Rationale**: Content can be regenerated, files cannot

#### `pdfs`
```sql
CREATE TABLE pdfs (
    id SERIAL PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    extracted_text TEXT,
    page_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```
- **Purpose**: PDF metadata and full-text content

### Removed Components
- ❌ `keywords` table (redundant with image_metadata.keywords[])
- ❌ `images.caption` column (moved to image_metadata)
- ❌ `images.embedding` column (stored in FAISS only)
- ❌ pgvector extension (using FAISS for vector similarity)

## Key Architecture Decisions

### 1. FAISS vs pgvector for Embeddings
**Decision**: Use FAISS for vector similarity search

**Rationale**:
- Faster for read-heavy workloads
- Separate index files per CLIP model variant
- No need for pgvector extension
- Simpler backup/restore (copy .npy and .index files)

**Location**: `metadata-files/clip-vit-{model}/`
- `features.npy`: Image embeddings
- `image_features.index`: FAISS index
- `paths.npz`: Image path mapping

### 2. PDF Thumbnail Generation
**Implementation**: First page rendered as JPEG

**Code**: `pdf_utils.py::generate_pdf_thumbnail()`
```python
# Renders first page at 400px width
page = doc[0]
pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
pix.save(thumbnail_path)
```

**Naming**: `{pdf_stem}_preview.jpg` (e.g., `document_preview.jpg`)

**Trigger**: Automatic on PDF upload via `save_pdf_to_catalog()`

### 3. Library State Persistence
**Decision**: Use localStorage instead of URL query parameters

**Attempted Approach**: URL params with `useSearchParams()` and `useRouter()`
- **Problem**: Next.js client-side navigation doesn't trigger full component remount
- **Issue**: `useState` initial value only calculated once, doesn't read updated URL params

**Working Approach**: localStorage with useEffect
```typescript
// Restore on mount
useEffect(() => {
  const saved = localStorage.getItem('libraryState');
  if (saved) {
    const state = JSON.parse(saved);
    setSelectedType(state.selectedType);
    setSearchTerm(state.searchTerm);
    // ...
  }
}, []);

// Auto-save on changes
useEffect(() => {
  localStorage.setItem('libraryState', JSON.stringify({
    selectedType, searchTerm, currentPage, itemsPerPage
  }));
}, [selectedType, searchTerm, currentPage, itemsPerPage]);
```

**State Tracked**:
- `selectedType`: "images" | "pdfs"
- `searchTerm`: Search query
- `currentPage`: Pagination state
- `itemsPerPage`: Results per page

### 4. Timestamp Management
**Storage**: UTC in database
**Display**: Can convert to HK time (UTC+8) in frontend
**Consistency**: All `created_at` and `updated_at` use `NOW()` default

### 5. AI JSON Parsing Robustness
**Problem**: Ollama sometimes returns malformed JSON or text explanations

**Solution**: Multi-layer parsing in `pdf_utils.py`
1. Brace counting to extract valid JSON from text
2. Control character stripping
3. Fallback to null values if parsing fails

```python
# Extract JSON from mixed text/JSON response
brace_count = 0
for i, char in enumerate(text):
    if char == '{':
        if brace_count == 0:
            start = i
        brace_count += 1
    elif char == '}':
        brace_count -= 1
        if brace_count == 0:
            json_str = text[start:i+1]
            break
```

## File Organization

### Backend Structure
```
c:\AIIMGS\
├── app.py                  # Main FastAPI application
├── auth.py                 # JWT authentication
├── db_config.py           # Database initialization
├── db_helper.py           # Database utilities
├── pdf_utils.py           # PDF processing & AI analysis
├── api/
│   └── routers/
│       ├── library.py     # Image/PDF listing
│       ├── search.py      # Semantic search
│       └── upload.py      # File uploads
├── catalog/               # Uploaded files (PDFs + images)
└── metadata-files/        # FAISS indexes per CLIP model
    ├── clip-vit-b-16/
    ├── clip-vit-b-32/
    ├── clip-vit-g-14/
    └── clip-vit-l-14/
```

### Frontend Structure
```
frontend/
├── src/
│   ├── app/
│   │   ├── library/       # Library view with state persistence
│   │   ├── indexing/      # Upload interface
│   │   ├── search/        # Semantic search
│   │   └── login/         # Authentication
│   └── components/
│       ├── Navbar.tsx     # Navigation bar
│       └── LoginPage.tsx  # Login form
└── public/                # Static assets
```

## Common Operations

### Reset All Data
```bash
python wipe_all.py
```
- Deletes all files in `catalog/`
- Clears all database tables
- Removes FAISS indexes

### Generate Missing PDF Thumbnails
```python
from pdf_utils import generate_pdf_thumbnail
import glob

for pdf_path in glob.glob("catalog/*.pdf"):
    generate_pdf_thumbnail(pdf_path)
```

### Database Reinitialization
```bash
run_db_init.bat
```
- Drops and recreates all tables
- Does not delete files

## Known Issues & Solutions

### Issue: Library State Lost on Navigation
**Symptom**: Returning from image detail page resets Library to initial state

**Cause**: Next.js component reuse without full remount

**Solution**: localStorage persistence (see Architecture Decision #3)

### Issue: Duplicate PDF Processing
**Symptom**: Same PDF processed multiple times on upload

**Cause**: File state not cleared after upload

**Solution**: Set file state to null after successful upload
```typescript
const handlePdfUpload = async () => {
  // ... upload logic ...
  setPdfFile(null); // Clear file input
};
```

### Issue: AI Returns Malformed JSON
**Symptom**: JSON parsing errors from Ollama responses

**Cause**: Model returns explanatory text with JSON

**Solution**: Brace counting extraction (see Architecture Decision #5)

## Development Workflow

### Start Backend
```bash
conda activate aiimgs
cd c:\AIIMGS
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend
```bash
cd c:\AIIMGS\frontend
npm run dev
```

### Access Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Future Considerations

### Potential Enhancements
1. **Shareable Library States**: Add URL query params for sharing specific views
2. **Batch PDF Processing**: Queue system for large PDF uploads
3. **Thumbnail Regeneration**: Admin interface to rebuild all thumbnails
4. **HK Timezone Display**: Convert UTC timestamps to HK time in UI
5. **Caption Regeneration**: Bulk AI analysis for existing images

### Scaling Considerations
- FAISS indexes loaded in memory (consider lazy loading for large datasets)
- PDF thumbnail generation blocks upload (consider async processing)
- No caching layer (consider Redis for frequently accessed data)

## Maintenance Notes

### Backup Checklist
- [ ] PostgreSQL database dump
- [ ] `catalog/` folder (images + PDFs)
- [ ] `metadata-files/` folder (FAISS indexes)
- [ ] `config.json` (application settings)

### After Schema Changes
1. Update `db_config.py` table definitions
2. Test with `run_db_init.bat`
3. Update this documentation

### After CLIP Model Changes
1. Rebuild FAISS indexes for all images
2. Update model references in search endpoints
3. Clear old model folders in `metadata-files/`

---

**Last Updated**: December 4, 2025
**Application Version**: 1.0
**Database Schema Version**: 2.0 (cleaned)
