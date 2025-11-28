# Run command: streamlit run app.py
import os
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, List, Sequence, Tuple, Dict, Optional
import json
from io import BytesIO

import faiss  # FAISS vector search library
import numpy as np
import open_clip  # type: ignore  # CLIP model
import streamlit as st  # Web application framework
import torch  # PyTorch deep learning framework
from PIL import Image  # Image processing
from streamlit.runtime.uploaded_file_manager import UploadedFile

# PDF processing related
try:
    import PyPDF2
    import fitz  # PyMuPDF
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ===== Load Configuration =====
# Default configuration values
DEFAULT_CONFIG = {
    "folders": {
        "images": "images",
        "pdf_catalog": "catalog",
        "metadata": "metadata-files"
    },
    "image_formats": [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"],
    "model": {
        "name": "clip-vit-b-32",
        "architecture": "ViT-B-32",
        "pretrained": "openai",
        "embedding_dim": 512
    },
    "search": {
        "top_k": 8,
        "batch_size": 8
    },
    "pdf": {
        "max_keywords": 5,
        "jpeg_quality": 85
    }
}

def load_config() -> dict:
    """Load configuration from config.json file"""
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Return default config if file doesn't exist
    return DEFAULT_CONFIG

CONFIG = load_config()

# ===== Configuration Constants =====
IMAGE_FOLDERS = [Path(CONFIG["folders"]["images"])]  # Image storage folder
IMAGE_EXTS = set(CONFIG["image_formats"])  # Supported image formats
PDF_CATALOG_FOLDER = Path(CONFIG["folders"]["pdf_catalog"])  # PDF file storage folder
MODEL_NAME = CONFIG["model"]["name"]  # Model name
INDEX_DIR = Path(CONFIG["folders"]["metadata"]) / MODEL_NAME  # Index file directory
PATHS_FILE = INDEX_DIR / "paths.npz"  # Image paths file
FEATURES_FILE = INDEX_DIR / "features.npy"  # Feature vectors file
FAISS_INDEX_FILE = INDEX_DIR / "image_features.index"  # FAISS index file
METADATA_FILE = INDEX_DIR / "metadata.json"  # Metadata file (captions, keywords)
TOP_K = CONFIG["search"]["top_k"]  # Number of search results
BATCH_SIZE = CONFIG["search"]["batch_size"]  # Batch processing size
CLIP_MODEL = CONFIG["model"]["architecture"]  # CLIP model architecture
CLIP_PRETRAINED = CONFIG["model"]["pretrained"]  # CLIP pretrained version
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU or CPU
EMBED_DIM = CONFIG["model"]["embedding_dim"]  # Embedding dimension
PDF_MAX_KEYWORDS = CONFIG["pdf"]["max_keywords"]  # Max keywords to extract from PDF
PDF_JPEG_QUALITY = CONFIG["pdf"]["jpeg_quality"]  # JPEG quality for PDF image extraction

# ===== Model Loading Functions =====
@st.cache_resource(show_spinner=False)
def load_clip_components():
    """Load CLIP model, preprocessor and tokenizer (cached for speed)"""
    model, preprocess, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model = model.to(DEVICE)
    model.eval()
    return model, preprocess, tokenizer


def ensure_index_dir():
    """Ensure index directory exists"""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def list_available_directories(base: Path = Path(".")) -> List[Path]:
    """List available directories"""
    return sorted([p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")])


def list_image_paths(folders: Iterable[Path] | None = None) -> List[Path]:
    """Recursively search for all image files in specified folders"""
    paths = []
    search_folders = list(folders) if folders is not None else IMAGE_FOLDERS
    for folder in search_folders:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                paths.append(path.resolve())
    return sorted(set(paths))


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit vectors (for similarity search)"""
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


# ===== Metadata Related Functions =====
def load_metadata_arrays() -> Tuple[List[str], np.ndarray]:
    """Load image paths and feature vectors from index"""
    if not PATHS_FILE.exists() or not FEATURES_FILE.exists():
        return [], np.empty((0, EMBED_DIM), dtype=np.float32)
    paths = np.load(PATHS_FILE, allow_pickle=True)["paths"].tolist()
    features = np.load(FEATURES_FILE)
    return paths, features


def normalize_path_key(path_str: str) -> str:
    """Normalize path to support interoperability between absolute and relative paths"""
    p = Path(path_str).resolve()
    return str(p)


def load_all_metadata() -> Dict[str, Dict]:
    """Load all image caption and keyword metadata"""
    if not METADATA_FILE.exists():
        return {}
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                # Convert all relative paths to absolute paths for consistency
                normalized = {}
                for key, val in data.items():
                    norm_key = normalize_path_key(key)
                    normalized[norm_key] = val
                return normalized
    except Exception:
        pass
    return {}


def save_metadata_file(metadata: Dict[str, Dict]):
    """Save metadata to JSON file (using relative paths for portability)"""
    ensure_index_dir()
    try:
        # Convert absolute paths to relative paths for portability
        relative_meta: Dict[str, Dict] = {}
        for abs_path, data in metadata.items():
            try:
                rel_path = str(Path(abs_path).relative_to(Path.cwd()))
            except ValueError:
                # If path cannot be made relative, use original path
                rel_path = abs_path
            relative_meta[rel_path] = data
        
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(relative_meta, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        st.warning(f"Failed to save metadata: {exc}")


def persist_index(paths: Sequence[str], features: np.ndarray, metadata: Optional[Dict[str, Dict]] = None):
    """Save index files, feature vectors and metadata"""
    ensure_index_dir()
    if not paths or features.size == 0:
        # Clear all index files
        for file in [PATHS_FILE, FEATURES_FILE, FAISS_INDEX_FILE]:
            if file.exists():
                file.unlink()
        if METADATA_FILE.exists():
            METADATA_FILE.unlink()
        st.session_state["indexed_paths"] = []
        st.session_state["faiss_index"] = None
        return
    normalized = normalize_vectors(features).astype(np.float32)
    index = faiss.IndexFlatIP(normalized.shape[1])
    index.add(normalized)
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    np.savez_compressed(PATHS_FILE, paths=np.array(paths))
    np.save(FEATURES_FILE, normalized)
    st.session_state["indexed_paths"] = list(paths)
    st.session_state["faiss_index"] = index

    # Save metadata (use provided if available, otherwise load existing and filter)
    if metadata is None:
        existing_meta = load_all_metadata()
    else:
        existing_meta = metadata
    filtered_meta: Dict[str, Dict] = {}
    for p in paths:
        if p in existing_meta and isinstance(existing_meta[p], dict):
            filtered_meta[p] = existing_meta[p]
        else:
            # Create default metadata entry for new images
            filtered_meta[p] = {"caption": "", "keywords": []}
    save_metadata_file(filtered_meta)
    st.session_state["metadata"] = filtered_meta


def load_index_into_session():
    """Load index from files into session state"""
    if "faiss_index" in st.session_state and "indexed_paths" in st.session_state:
        return
    if not FAISS_INDEX_FILE.exists() or not PATHS_FILE.exists():
        st.session_state["indexed_paths"] = []
        st.session_state["faiss_index"] = None
        st.session_state["metadata"] = {}
        return
    data = np.load(PATHS_FILE, allow_pickle=True)
    paths = data["paths"].tolist()
    index = faiss.read_index(str(FAISS_INDEX_FILE))
    st.session_state["indexed_paths"] = paths
    st.session_state["faiss_index"] = index
    st.session_state["metadata"] = load_all_metadata()

# ===== Embedding and Indexing Functions =====
def encode_image_batch(tensors: List[torch.Tensor], model: torch.nn.Module) -> np.ndarray:
    """Encode a batch of images into feature vectors"""
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        features = model.encode_image(batch)
    return features.cpu().numpy().astype(np.float32)


def extract_image_index(
    paths: Sequence[Path],
    progress_callback: Callable[[int, int], None] | None = None,
) -> Tuple[List[Path], np.ndarray]:
    """Extract feature vectors from images and build index"""
    model, preprocess, _ = load_clip_components()
    valid_paths: List[Path] = []
    tensors: List[torch.Tensor] = []
    embeddings: List[np.ndarray] = []
    total = len(paths)
    processed = 0

    def flush_batch():
        nonlocal tensors
        if not tensors:
            return
        embeddings.append(encode_image_batch(tensors, model))
        tensors = []

    for path in paths:
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            st.warning(f"Failed to process image {path.name}: {exc}")
            continue
        tensors.append(preprocess(image))
        valid_paths.append(path)
        if len(tensors) == BATCH_SIZE:
            flush_batch()
        processed += 1
        if progress_callback:
            progress_callback(processed, total)

    flush_batch()

    if not embeddings:
        return [], np.empty((0, EMBED_DIM), dtype=np.float32)

    features = np.vstack(embeddings).astype(np.float32)
    return valid_paths, features


def sync_directories(
    dir_paths: Sequence[Path],
    progress_reporter: Callable[[int, int], None] | None = None,
):
    """Sync images from selected folders to index"""
    image_paths = list_image_paths(dir_paths)
    if not image_paths:
        st.warning("No images found in selected folders; index unchanged.")
        return
    existing_paths, existing_features = load_metadata_arrays()
    existing_map = {path: existing_features[idx] for idx, path in enumerate(existing_paths)}
    existing_meta = load_all_metadata()
    retained_paths: List[str] = []
    retained_features: List[np.ndarray] = []
    new_candidates: List[Path] = []
    for path in image_paths:
        path_str = str(path)
        if path_str in existing_map:
            retained_paths.append(path_str)
            retained_features.append(existing_map[path_str])
        else:
            new_candidates.append(path)
    new_count = 0
    if new_candidates:
        new_paths, new_features = extract_image_index(
            new_candidates,
            progress_callback=progress_reporter,
        )
        retained_paths.extend(str(p) for p in new_paths)
        retained_features.extend(list(new_features))
        new_count = len(new_paths)
        # Add default metadata entries for newly indexed images
        for p in new_paths:
            pstr = str(p)
            if pstr not in existing_meta:
                existing_meta[pstr] = {"caption": "", "keywords": []}
    if retained_paths:
        combined_features = np.vstack(retained_features)
    else:
        combined_features = np.empty((0, EMBED_DIM), dtype=np.float32)
    persist_index(retained_paths, combined_features, metadata=existing_meta)
    removed_count = max(len(existing_paths) - len(retained_paths), 0)
    st.success(
        f"Sync complete: kept {len(retained_paths)} files, "
        f"added {new_count}, removed {removed_count}."
    )


def remove_images_from_index(paths_to_remove: Sequence[str], delete_files: bool = False):
    """Remove images from index"""
    if not paths_to_remove:
        return
    existing_paths, existing_features = load_metadata_arrays()
    if not existing_paths:
        st.info("No images in index to remove.")
        return
    remove_set = set(paths_to_remove)
    keep_indices = [idx for idx, path in enumerate(existing_paths) if path not in remove_set]
    new_paths = [existing_paths[idx] for idx in keep_indices]
    if keep_indices:
        new_features = existing_features[keep_indices]
    else:
        new_features = np.empty((0, EMBED_DIM), dtype=np.float32)
    # Filter metadata for retained paths
    existing_meta = load_all_metadata()
    new_meta: Dict[str, Dict] = {}
    for p in new_paths:
        if p in existing_meta:
            new_meta[p] = existing_meta[p]
    persist_index(new_paths, new_features, metadata=new_meta)
    if delete_files:
        for path in paths_to_remove:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception as exc:
                st.warning(f"Failed to delete file {path}: {exc}")
    st.success(f"Removed {len(paths_to_remove)} images from index.")

# ===== Search Related Functions =====
def embed_uploaded_image(uploaded_file) -> np.ndarray:
    """Encode uploaded image into vector"""
    model, preprocess, _ = load_clip_components()
    image = Image.open(uploaded_file).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model.encode_image(tensor)
    vector = features[0].cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / max(norm, 1e-12)


def embed_text(query: str) -> np.ndarray:
    """Encode text search query into vector"""
    model, _, tokenizer = load_clip_components()
    tokens = tokenizer([query])
    with torch.no_grad():
        features = model.encode_text(tokens.to(DEVICE))
    vector = features[0].cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / max(norm, 1e-12)


def search_similar(vector: np.ndarray, top_k: int = TOP_K) -> List[Tuple[str, float]]:
    """Search for similar images using FAISS"""
    index = st.session_state.get("faiss_index")
    paths = st.session_state.get("indexed_paths", [])
    if index is None or not paths:
        return []
    query = vector.astype(np.float32)[None, :]
    scores, indices = index.search(query, min(top_k, len(paths)))
    results: List[Tuple[str, float]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(paths):
            continue
        results.append((paths[idx], float(score)))
    return results


def record_query_metrics(mode: str, duration: float):
    """Record search performance metrics"""
    comparisons = len(st.session_state.get("indexed_paths", []))
    throughput = comparisons / duration if duration > 0 else float("inf")
    st.session_state["last_metrics"] = {
        "mode": mode,
        "duration": duration,
        "comparisons": comparisons,
        "throughput": throughput,
    }


def render_results(results: List[Tuple[str, float]]):
    """Display search results"""
    if not results:
        st.info("No results yet. Build index or adjust query.")
        return
    cols = st.columns(4)
    for idx, (img_path, score) in enumerate(results):
        col = cols[idx % len(cols)]
        with col:
            meta = st.session_state.get("metadata", {}).get(img_path, {})
            caption_parts = [f"{Path(img_path).name}", f"Similarity {score:.2f}"]
            if isinstance(meta, dict):
                caption = meta.get("caption", "")
                keywords = meta.get("keywords", [])
                if caption:
                    caption_parts.insert(1, f"{caption}")
                if keywords:
                    kw = ", ".join(keywords)
                    caption_parts.append(f"Keywords: {kw}")
            full_caption = " · ".join(caption_parts)
            st.image(
                img_path,
                caption=full_caption,
                width="stretch",
            )


def save_library_uploads(files: Sequence[UploadedFile]) -> List[Path]:
    """Save uploaded images to image folder"""
    saved: List[Path] = []
    if not files:
        return saved
    target_dir = IMAGE_FOLDERS[0]
    target_dir.mkdir(parents=True, exist_ok=True)
    for uploaded in files:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in IMAGE_EXTS:
            st.warning(f"{uploaded.name} is not a supported image format; skipped.")
            continue
        stem = Path(uploaded.name).stem
        dest = target_dir / f"{stem}{suffix}"
        counter = 1
        while dest.exists():
            dest = target_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())
        saved.append(dest)
    return saved


# ===== PDF Processing Functions =====
def save_pdf_to_catalog(pdf_file: UploadedFile, catalog_folder: Path) -> Path:
    """
    Save uploaded PDF to catalog folder
    
    Args:
        pdf_file: Uploaded PDF file
        catalog_folder: Target folder for PDF storage
    
    Returns:
        Path to saved PDF file
    """
    catalog_folder.mkdir(parents=True, exist_ok=True)
    
    pdf_filename = Path(pdf_file.name).name
    dest_path = catalog_folder / pdf_filename
    
    # Avoid duplicate filenames
    counter = 1
    stem = Path(pdf_file.name).stem
    while dest_path.exists():
        dest_path = catalog_folder / f"{stem}_{counter}.pdf"
        counter += 1
    
    # Save PDF
    with open(dest_path, "wb") as f:
        f.write(pdf_file.read())
    
    return dest_path


def extract_images_from_pdf(pdf_file: UploadedFile, output_folder: Path) -> Tuple[List[Path], str]:
    """
    Extract all embedded images from PDF and save them
    
    Args:
        pdf_file: Uploaded PDF file
        output_folder: Image output folder
    
    Returns:
        (List of image paths, PDF filename)
    """
    if not PDF_SUPPORT:
        raise ImportError("PDF support not available. Install PyMuPDF (fitz).")
    
    pdf_filename = Path(pdf_file.name).stem  # PDF filename (without extension)
    pdf_bytes = pdf_file.read()
    
    # Open PDF using PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    image_paths: List[Path] = []
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    img_counter = 1
    
    # Iterate through each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Get all images on the page
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]  # Image xref number
            
            try:
                # Extract image
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]  # Image format (png, jpeg, etc.)
                
                # Convert to PIL Image
                img = Image.open(BytesIO(image_bytes))
                
                # Image filename format: pdfname_img1.jpg, pdfname_img2.jpg ...
                img_filename = f"{pdf_filename}_img{img_counter}.jpg"
                img_path = output_folder / img_filename
                
                # Avoid duplicate filenames
                counter = 1
                while img_path.exists():
                    img_filename = f"{pdf_filename}_img{img_counter}_{counter}.jpg"
                    img_path = output_folder / img_filename
                    counter += 1
                
                # Save as JPEG
                if img.mode in ("RGBA", "LA", "P"):
                    # Convert transparent background to white
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                    img = background
                elif img.mode != "RGB":
                    img = img.convert("RGB")
                
                img.save(img_path, "JPEG", quality=PDF_JPEG_QUALITY)
                image_paths.append(img_path)
                img_counter += 1
                
            except Exception as e:
                st.warning(f"Unable to extract image (page {page_num + 1}, img {img_index + 1}): {str(e)}")
                continue
    
    doc.close()
    
    return image_paths, pdf_filename


def extract_text_from_pdf(pdf_file: UploadedFile) -> str:
    """
    Extract all text from PDF
    
    Args:
        pdf_file: Uploaded PDF file
    
    Returns:
        Extracted text content
    """
    if not PDF_SUPPORT:
        raise ImportError("PDF support not available. Install PyPDF2.")
    
    pdf_bytes = pdf_file.read()
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    
    all_text = []
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            all_text.append(text)
    
    return "\n\n".join(all_text)


def extract_keywords_from_text(text: str, max_keywords: int = PDF_MAX_KEYWORDS) -> List[str]:
    """
    Extract keywords from text
    
    Args:
        text: Text content
        max_keywords: Maximum number of keywords to return
    
    Returns:
        List of keywords
    """
    import re
    from collections import Counter
    
    # Common stop words (English and Chinese)
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those",
        "i", "you", "he", "she", "it", "we", "they", "them", "their", "its", "our", "your",
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "個", "上", "也", "說",
        "出", "到", "時", "要", "以", "用", "著", "能", "之", "會", "後", "然", "沒", "很", "好", "來",
        "page", "pages", "document", "file", "pdf", "image", "fig", "figure"
    }
    
    # Remove special characters, keep letters, numbers, Chinese
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    
    # Filter stop words and words that are too short
    filtered_words = [
        word for word in words 
        if len(word) >= 3 and word not in stop_words
    ]
    
    # Count word frequency
    word_counts = Counter(filtered_words)
    
    # Get most common keywords
    top_keywords = [word for word, count in word_counts.most_common(max_keywords * 2)]
    
    # Filter out pure numbers
    keywords = [kw for kw in top_keywords if not kw.isdigit()][:max_keywords]
    
    return keywords


# ===== Streamlit UI Application =====
st.title("AI Image Similarity Search")
load_index_into_session()

view_mode = st.sidebar.radio(
    "Navigation",
    ["Search", "Indexing", "Metadata"],
    index=0,
)

# Reset temporary UI state when switching views
if "last_view_mode" not in st.session_state:
    st.session_state["last_view_mode"] = view_mode
elif st.session_state["last_view_mode"] != view_mode:
    st.session_state["last_view_mode"] = view_mode
    # Clear temporary state variables
    for key in ["removal_selection", "last_metrics"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

if view_mode == "Indexing":
    st.subheader("Index management")
    
    # ===== PDF Upload and Image Extraction =====
    if PDF_SUPPORT:
        st.markdown("### 📄 Extract Images from PDF")
        pdf_files = st.file_uploader(
            "Upload PDF files to extract images",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files. Images will be extracted from each page."
        )
        
        if pdf_files and st.button("Extract images from PDF"):
            output_folder = IMAGE_FOLDERS[0]
            
            with st.spinner("Processing PDF files..."):
                all_extracted_images = []
                all_pdf_data = []  # [(image_paths, pdf_filename, extracted_text)]
                
                for pdf_file in pdf_files:
                    try:
                        st.info(f"Processing: {pdf_file.name}")
                        
                        # Save PDF to catalog folder
                        pdf_path = save_pdf_to_catalog(pdf_file, PDF_CATALOG_FOLDER)
                        st.success(f"✓ Saved PDF to: {pdf_path}")
                        
                        # Extract images
                        pdf_file.seek(0)  # Reset file pointer
                        image_paths, pdf_filename = extract_images_from_pdf(pdf_file, output_folder)
                        
                        # Extract text
                        pdf_file.seek(0)  # Reset file pointer
                        extracted_text = extract_text_from_pdf(pdf_file)
                        
                        # Extract keywords
                        suggested_keywords = extract_keywords_from_text(extracted_text, max_keywords=PDF_MAX_KEYWORDS)
                        
                        all_extracted_images.extend(image_paths)
                        all_pdf_data.append({
                            "image_paths": image_paths,
                            "pdf_filename": pdf_filename,
                            "extracted_text": extracted_text,
                            "suggested_keywords": suggested_keywords
                        })
                        
                        st.success(f"✓ Extracted {len(image_paths)} images from {pdf_file.name}")
                    
                    except Exception as e:
                        st.error(f"✗ Error processing {pdf_file.name}: {str(e)}")
                
                if all_extracted_images:
                    st.session_state["pdf_extracted_data"] = all_pdf_data
                    st.session_state["pdf_keywords_input"] = {}
                    st.success(f"Total: {len(all_extracted_images)} images extracted. Scroll down to add keywords.")
        
        # ===== Keywords Selection Interface =====
        if "pdf_extracted_data" in st.session_state:
            st.markdown("### 🏷️ Add Keywords for Extracted Images")
            st.info("Review extracted text and enter keywords for each image. The PDF filename will be used as the caption.")
            
            for pdf_data in st.session_state["pdf_extracted_data"]:
                pdf_filename = pdf_data["pdf_filename"]
                image_paths = pdf_data["image_paths"]
                extracted_text = pdf_data["extracted_text"]
                
                st.markdown(f"#### PDF: `{pdf_filename}.pdf`")
                
                # Display extracted text (for user reference)
                with st.expander("📝 Extracted text from PDF (for reference)", expanded=False):
                    st.text_area(
                        "Text content",
                        value=extracted_text[:2000] + ("..." if len(extracted_text) > 2000 else ""),
                        height=200,
                        disabled=True,
                        key=f"text_preview_{pdf_filename}"
                    )
                
                # Enter keywords for each image
                cols = st.columns(2)
                suggested_keywords = pdf_data.get("suggested_keywords", [])
                default_keywords_str = ", ".join(suggested_keywords)
                
                for idx, img_path in enumerate(image_paths):
                    col = cols[idx % 2]
                    with col:
                        try:
                            st.image(str(img_path), caption=img_path.name, width=250)
                        except:
                            st.warning(f"Cannot preview: {img_path.name}")
                        
                        keywords_key = f"keywords_{pdf_filename}_{idx}"
                        keywords_input = st.text_input(
                            f"Keywords for {img_path.name}",
                            key=keywords_key,
                            value=default_keywords_str,
                            help="Enter comma-separated keywords (auto-suggested from PDF text)"
                        )
                        
                        st.session_state["pdf_keywords_input"][str(img_path)] = keywords_input
                
                st.divider()
            
            # Save all metadata
            if st.button("💾 Save all metadata and finish", type="primary"):
                metadata = load_all_metadata()
                
                for pdf_data in st.session_state["pdf_extracted_data"]:
                    pdf_filename = pdf_data["pdf_filename"]
                    image_paths = pdf_data["image_paths"]
                    
                    for img_path in image_paths:
                        img_path_str = str(img_path)
                        keywords_input = st.session_state["pdf_keywords_input"].get(img_path_str, "")
                        keywords_list = [k.strip() for k in keywords_input.split(",") if k.strip()]
                        
                        # Normalize path
                        norm_path = normalize_path_key(img_path_str)
                        
                        # Use PDF filename as caption
                        metadata[norm_path] = {
                            "caption": f"{pdf_filename}.pdf",
                            "keywords": keywords_list
                        }
                
                save_metadata_file(metadata)
                st.session_state["metadata"] = metadata
                
                # Clean up temporary data
                del st.session_state["pdf_extracted_data"]
                del st.session_state["pdf_keywords_input"]
                
                st.success("✓ All metadata saved successfully! You can now run 'Sync selected folders' to index these images.")
                st.rerun()
    else:
        st.warning("PDF support not available. Install PyPDF2 and pdf2image to enable this feature.")
    
    st.divider()
    
    # ===== Original Image Upload Feature =====
    upload_candidates = st.file_uploader(
        "Add new images to the gallery (multiple files allowed)",
        type=list({ext.replace(".", "") for ext in IMAGE_EXTS}),
        accept_multiple_files=True,
    )
    if st.button("Save uploaded images", disabled=not upload_candidates):
        saved_paths = save_library_uploads(upload_candidates or [])
        if saved_paths:
            st.success(f"Added {len(saved_paths)} images. Run sync below to update the index.")
        else:
            st.info("No images were saved. Please confirm the file types.")

    available_dirs = list_available_directories()
    default_dirs = [str(path) for path in IMAGE_FOLDERS if path.exists()]
    dir_options = [str(path) for path in available_dirs]
    selected_dir_labels = st.multiselect(
        "Choose folders to index",
        options=dir_options,
        default=default_dirs or dir_options,
        help="Sync all images in these folders. Re-run any time to pick up changes.",
    )
    if st.button("Sync selected folders"):
        if not selected_dir_labels:
            st.warning("Select at least one folder.")
        else:
            target_dirs = [Path(label) for label in selected_dir_labels]
            progress_bar = st.progress(0, text="Preparing sync...")

            def report_progress(done: int, total: int):
                total = max(total, 1)
                percent = int(min(done / total, 1) * 100)
                progress_bar.progress(percent, text=f"Embedding progress: {done}/{total}")

            with st.spinner("Syncing index..."):
                sync_directories(target_dirs, progress_reporter=report_progress)
            progress_bar.progress(100, text="Sync complete")
            progress_bar.empty()

    current_indexed_paths = st.session_state.get("indexed_paths", [])
    removal_selection = st.multiselect(
        "Remove images from index",
        options=current_indexed_paths,
        help="Select entries to delete from the index. Optional: remove files from disk.",
        key="removal_selection",
    )
    if removal_selection:
        preview_cols = st.columns(4)
        for idx, img_path in enumerate(removal_selection):
            col = preview_cols[idx % len(preview_cols)]
            with col:
                try:
                    st.image(
                        img_path,
                        caption=Path(img_path).name,
                        width="stretch",
                    )
                except Exception as exc:
                    st.warning(f"Unable to preview {img_path}: {exc}")

    delete_files = st.checkbox("Delete image files when removing from index", value=False)
    if st.button("Remove selected images", disabled=not removal_selection):
        with st.spinner("Removing selected images..."):
            remove_images_from_index(removal_selection, delete_files)

elif view_mode == "Metadata":
    st.subheader("Edit captions & keywords")
    current_indexed_paths = st.session_state.get("indexed_paths", [])
    if current_indexed_paths:
        edit_choice = st.selectbox(
            "Choose image to edit",
            options=current_indexed_paths,
            index=None,
            placeholder="Select an image..."
        )
        if edit_choice:
            try:
                st.image(edit_choice, caption=Path(edit_choice).name, width=300)
            except Exception as exc:
                st.warning(f"Unable to load image preview: {exc}")
            
            meta_map = st.session_state.get("metadata", {})
            current_meta = meta_map.get(edit_choice, {"caption": "", "keywords": []})
            new_caption = st.text_input("Caption", value=current_meta.get("caption", ""))
            kw_default = ", ".join(current_meta.get("keywords", []))
            new_keywords = st.text_input("Keywords (comma-separated)", value=kw_default)
            if st.button("Save metadata"):
                md = load_all_metadata()
                md[edit_choice] = {"caption": new_caption, "keywords": [k.strip() for k in new_keywords.split(",") if k.strip()]}
                save_metadata_file(md)
                st.session_state["metadata"] = md
                st.success("Saved metadata for selected image.")
        else:
            st.info("Select an image to view and edit its metadata.")
    else:
        st.info("No indexed images to edit. Run a sync first in Indexing.")

else:
    # Search mode (default)
    st.divider()
    
    if not st.session_state.get("indexed_paths"):
        st.warning("No index yet. Switch to \"Indexing\" and run a sync.")
    else:
        st.caption(f"Images available for search: {len(st.session_state['indexed_paths'])}")

    search_mode = st.radio("Select search mode", ["Text search", "Image search"], horizontal=True)

    metrics_container = st.container()

    if search_mode == "Text search":
        text_query = st.text_input("Describe what you need (e.g., glass roof, beach, sunset...)")
        if text_query.strip():
            with st.spinner("Searching..."):
                start_time = perf_counter()
                query_vector = embed_text(text_query.strip())
                matches = search_similar(query_vector)
                duration = perf_counter() - start_time
                record_query_metrics("Text search", duration)
            render_results(matches)
    elif search_mode == "Image search":
        uploaded_file = st.file_uploader("Upload an image to find similar ones", type=["jpg", "jpeg", "png", "bmp", "gif", "webp"])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Your uploaded image (thumbnail)", width="stretch")
            with st.spinner("Searching for similar images..."):
                start_time = perf_counter()
                query_vector = embed_uploaded_image(uploaded_file)
                matches = search_similar(query_vector)
                duration = perf_counter() - start_time
                record_query_metrics("Image search", duration)
            render_results(matches)

    with metrics_container:
        metrics = st.session_state.get("last_metrics")
        if metrics:
            st.divider()
            st.caption(
                f"{metrics['mode']} took {metrics['duration'] * 1000:.1f} ms · "
                f"scanned {metrics['comparisons']} images · "
                f"~{metrics['throughput']:.1f} images/sec"
            )