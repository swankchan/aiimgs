# 執行指令: streamlit run app.py
import os
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, List, Sequence, Tuple, Dict, Optional
import json

import faiss  # FAISS 向量搜尋庫
import numpy as np
import open_clip  # type: ignore  # CLIP 模型
import streamlit as st  # 網頁應用框架
import torch  # PyTorch 深度學習框架
from PIL import Image  # 圖像處理
from streamlit.runtime.uploaded_file_manager import UploadedFile

# ===== 設定常數 =====
IMAGE_FOLDERS = [Path("images")]  # 圖像儲存資料夾
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}  # 支援的圖像格式
MODEL_NAME = "clip-vit-b-32"  # 模型名稱
INDEX_DIR = Path("metadata-files") / MODEL_NAME  # 索引檔案目錄
PATHS_FILE = INDEX_DIR / "paths.npz"  # 圖像路徑檔案
FEATURES_FILE = INDEX_DIR / "features.npy"  # 特徵向量檔案
FAISS_INDEX_FILE = INDEX_DIR / "image_features.index"  # FAISS 索引檔案
METADATA_FILE = INDEX_DIR / "metadata.json"  # 中繼資料檔案 (標題、關鍵字)
TOP_K = 8  # 搜尋結果數量
BATCH_SIZE = 8  # 批次處理大小
CLIP_MODEL = "ViT-B-32"  # CLIP 模型架構
CLIP_PRETRAINED = "openai"  # CLIP 預訓練版本
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 或 CPU
EMBED_DIM = 512  # 嵌入維度

# ===== 載入模型相關函數 =====
@st.cache_resource(show_spinner=False)
def load_clip_components():
    """載入 CLIP 模型、前處理器和分詞器（使用快取加速）"""
    model, preprocess, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model = model.to(DEVICE)
    model.eval()
    return model, preprocess, tokenizer


def ensure_index_dir():
    """確保索引目錄存在"""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def list_available_directories(base: Path = Path(".")) -> List[Path]:
    """列出可用的目錄"""
    return sorted([p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")])


def list_image_paths(folders: Iterable[Path] | None = None) -> List[Path]:
    """遞迴搜尋指定資料夾中的所有圖像檔案"""
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
    """將向量正規化為單位向量（用於相似度搜尋）"""
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


# ===== 中繼資料相關函數 =====
def load_metadata_arrays() -> Tuple[List[str], np.ndarray]:
    """載入索引中的圖像路徑和特徵向量"""
    if not PATHS_FILE.exists() or not FEATURES_FILE.exists():
        return [], np.empty((0, EMBED_DIM), dtype=np.float32)
    paths = np.load(PATHS_FILE, allow_pickle=True)["paths"].tolist()
    features = np.load(FEATURES_FILE)
    return paths, features


def normalize_path_key(path_str: str) -> str:
    """正規化路徑以支援絕對路徑和相對路徑的互操作性"""
    p = Path(path_str).resolve()
    return str(p)


def load_all_metadata() -> Dict[str, Dict]:
    """載入所有圖像的標題和關鍵字中繼資料"""
    if not METADATA_FILE.exists():
        return {}
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                # 將所有相對路徑轉換為絕對路徑以確保一致性
                normalized = {}
                for key, val in data.items():
                    norm_key = normalize_path_key(key)
                    normalized[norm_key] = val
                return normalized
    except Exception:
        pass
    return {}


def save_metadata_file(metadata: Dict[str, Dict]):
    """儲存中繼資料到 JSON 檔案（使用相對路徑以增進可移植性）"""
    ensure_index_dir()
    try:
        # 將絕對路徑轉換為相對路徑以提高可移植性
        relative_meta: Dict[str, Dict] = {}
        for abs_path, data in metadata.items():
            try:
                rel_path = str(Path(abs_path).relative_to(Path.cwd()))
            except ValueError:
                # 若路徑無法相對化，則使用原始路徑
                rel_path = abs_path
            relative_meta[rel_path] = data
        
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(relative_meta, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        st.warning(f"儲存中繼資料失敗: {exc}")


def persist_index(paths: Sequence[str], features: np.ndarray, metadata: Optional[Dict[str, Dict]] = None):
    """保存索引檔案、特徵向量和中繼資料"""
    ensure_index_dir()
    if not paths or features.size == 0:
        # 清空所有索引檔案
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

    # 保存中繼資料（若提供則使用，否則載入既有資料並過濾）
    if metadata is None:
        existing_meta = load_all_metadata()
    else:
        existing_meta = metadata
    filtered_meta: Dict[str, Dict] = {}
    for p in paths:
        if p in existing_meta and isinstance(existing_meta[p], dict):
            filtered_meta[p] = existing_meta[p]
        else:
            # 為新圖像建立預設中繼資料項目
            filtered_meta[p] = {"caption": "", "keywords": []}
    save_metadata_file(filtered_meta)
    st.session_state["metadata"] = filtered_meta


def load_index_into_session():
    """從檔案載入索引到 session state"""
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

# ===== 嵌入和索引相關函數 =====
def encode_image_batch(tensors: List[torch.Tensor], model: torch.nn.Module) -> np.ndarray:
    """將一批圖像編碼為特徵向量"""
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        features = model.encode_image(batch)
    return features.cpu().numpy().astype(np.float32)


def extract_image_index(
    paths: Sequence[Path],
    progress_callback: Callable[[int, int], None] | None = None,
) -> Tuple[List[Path], np.ndarray]:
    """提取圖像的特徵向量並建立索引"""
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
            st.warning(f"處理圖像失敗 {path.name}: {exc}")
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
    """同步選定資料夾中的圖像到索引"""
    image_paths = list_image_paths(dir_paths)
    if not image_paths:
        st.warning("選定資料夾中沒有圖像；索引未更改。")
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
        # 為新索引的圖像新增預設中繼資料項目
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
        f"同步完成: 保留 {len(retained_paths)} 個檔案, "
        f"新增 {new_count} 個, 移除 {removed_count} 個。"
    )


def remove_images_from_index(paths_to_remove: Sequence[str], delete_files: bool = False):
    """從索引中移除圖像"""
    if not paths_to_remove:
        return
    existing_paths, existing_features = load_metadata_arrays()
    if not existing_paths:
        st.info("索引中沒有圖像可移除。")
        return
    remove_set = set(paths_to_remove)
    keep_indices = [idx for idx, path in enumerate(existing_paths) if path not in remove_set]
    new_paths = [existing_paths[idx] for idx in keep_indices]
    if keep_indices:
        new_features = existing_features[keep_indices]
    else:
        new_features = np.empty((0, EMBED_DIM), dtype=np.float32)
    # 過濾保留的路徑的中繼資料
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
                st.warning(f"刪除檔案失敗 {path}: {exc}")
    st.success(f"從索引中移除 {len(paths_to_remove)} 個圖像。")

# ===== 搜尋相關函數 =====
def embed_uploaded_image(uploaded_file) -> np.ndarray:
    """編碼上傳的圖像為向量"""
    model, preprocess, _ = load_clip_components()
    image = Image.open(uploaded_file).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model.encode_image(tensor)
    vector = features[0].cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / max(norm, 1e-12)


def embed_text(query: str) -> np.ndarray:
    """編碼文本搜尋查詢為向量"""
    model, _, tokenizer = load_clip_components()
    tokens = tokenizer([query])
    with torch.no_grad():
        features = model.encode_text(tokens.to(DEVICE))
    vector = features[0].cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / max(norm, 1e-12)


def search_similar(vector: np.ndarray, top_k: int = TOP_K) -> List[Tuple[str, float]]:
    """使用 FAISS 搜尋相似圖像"""
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
    """記錄搜尋性能指標"""
    comparisons = len(st.session_state.get("indexed_paths", []))
    throughput = comparisons / duration if duration > 0 else float("inf")
    st.session_state["last_metrics"] = {
        "mode": mode,
        "duration": duration,
        "comparisons": comparisons,
        "throughput": throughput,
    }


def render_results(results: List[Tuple[str, float]]):
    """顯示搜尋結果"""
    if not results:
        st.info("尚無結果。建立索引或調整查詢。")
        return
    cols = st.columns(4)
    for idx, (img_path, score) in enumerate(results):
        col = cols[idx % len(cols)]
        with col:
            meta = st.session_state.get("metadata", {}).get(img_path, {})
            caption_parts = [f"{Path(img_path).name}", f"相似度 {score:.2f}"]
            if isinstance(meta, dict):
                caption = meta.get("caption", "")
                keywords = meta.get("keywords", [])
                if caption:
                    caption_parts.insert(1, f"{caption}")
                if keywords:
                    kw = ", ".join(keywords)
                    caption_parts.append(f"關鍵字: {kw}")
            full_caption = " · ".join(caption_parts)
            st.image(
                img_path,
                caption=full_caption,
                width="stretch",
            )


def save_library_uploads(files: Sequence[UploadedFile]) -> List[Path]:
    """保存上傳的圖像到圖像資料夾"""
    saved: List[Path] = []
    if not files:
        return saved
    target_dir = IMAGE_FOLDERS[0]
    target_dir.mkdir(parents=True, exist_ok=True)
    for uploaded in files:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in IMAGE_EXTS:
            st.warning(f"{uploaded.name} 不是支援的圖像格式；已跳過。")
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


# ===== Streamlit UI 應用程式 =====
st.title("AI Image Similarity Search")
load_index_into_session()

view_mode = st.sidebar.radio(
    "Navigation",
    ["Search", "Indexing", "Metadata"],
    index=0,
)

# 切換視圖時重置臨時 UI 狀態
if "last_view_mode" not in st.session_state:
    st.session_state["last_view_mode"] = view_mode
elif st.session_state["last_view_mode"] != view_mode:
    st.session_state["last_view_mode"] = view_mode
    # 清除臨時狀態變數
    for key in ["removal_selection", "last_metrics"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

if view_mode == "Indexing":
    st.subheader("Index management")
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
    # 搜尋模式（預設）
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