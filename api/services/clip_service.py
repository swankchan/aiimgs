"""
CLIP model service for image embedding and search
"""
import numpy as np
import torch
import open_clip
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import faiss
import json

class ClipService:
    """Service for CLIP model operations"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ready = False
        
        # Configuration
        self.model_name = self.config["model"]["architecture"]
        self.pretrained = self.config["model"]["pretrained"]
        self.embed_dim = self.config["model"]["embedding_dim"]
        self.index_dir = Path(self.config["folders"]["metadata"]) / self.config["model"]["name"]
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            # Return default config
            return {
                "model": {
                    "name": "clip-vit-b-16",
                    "architecture": "ViT-B-16",
                    "pretrained": "openai",
                    "embedding_dim": 512
                },
                "folders": {
                    "images": "images",
                    "metadata": "metadata-files"
                }
            }
    
    async def initialize(self):
        """Initialize CLIP model"""
        print(f"Loading CLIP model: {self.model_name} ({self.pretrained})")
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.ready = True
        print(f"✓ CLIP model loaded on {self.device}")
    
    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self.ready and self.model is not None
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text query into vector"""
        if not self.is_ready():
            raise RuntimeError("CLIP service not initialized")
        
        tokens = self.tokenizer([text])
        with torch.no_grad():
            features = self.model.encode_text(tokens.to(self.device))
        
        vector = features[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / max(norm, 1e-12)
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode image into vector"""
        if not self.is_ready():
            raise RuntimeError("CLIP service not initialized")
        
        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(tensor)
        
        vector = features[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / max(norm, 1e-12)
    
    def encode_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Encode image from bytes into vector"""
        if not self.is_ready():
            raise RuntimeError("CLIP service not initialized")
        
        from io import BytesIO
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(tensor)
        
        vector = features[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / max(norm, 1e-12)
    
    def encode_image_batch(self, image_paths: List[str], batch_size: int = 8) -> np.ndarray:
        """Encode batch of images into vectors"""
        if not self.is_ready():
            raise RuntimeError("CLIP service not initialized")
        
        all_features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tensors = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert("RGB")
                    tensors.append(self.preprocess(image))
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    continue
            
            if not tensors:
                continue
            
            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(batch)
            
            all_features.append(features.cpu().numpy().astype(np.float32))
        
        if not all_features:
            return np.empty((0, self.embed_dim), dtype=np.float32)
        
        return np.vstack(all_features)
    
    def search_similar(self, query_vector: np.ndarray, top_k: int = 32) -> List[Tuple[str, float]]:
        """Search for similar images using FAISS index"""
        index_file = self.index_dir / "image_features.index"
        paths_file = self.index_dir / "paths.npz"
        
        if not index_file.exists() or not paths_file.exists():
            return []
        
        # Load index and paths
        index = faiss.read_index(str(index_file))
        paths_data = np.load(paths_file, allow_pickle=True)
        paths = paths_data["paths"].tolist()
        
        # Search
        query = query_vector.astype(np.float32)[None, :]
        scores, indices = index.search(query, min(top_k, len(paths)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(paths):
                continue
            results.append((paths[idx], float(score)))
        
        return results
    
    def normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit vectors"""
        if vectors.size == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms
    
    def create_index(self, paths: List[str], features: np.ndarray) -> bool:
        """Create FAISS index from features"""
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            
            if len(paths) == 0 or features.size == 0:
                return False
            
            # Normalize features
            normalized = self.normalize_vectors(features).astype(np.float32)
            
            # Create FAISS index
            index = faiss.IndexFlatIP(normalized.shape[1])
            index.add(normalized)
            
            # Save index and paths
            faiss.write_index(index, str(self.index_dir / "image_features.index"))
            np.savez_compressed(self.index_dir / "paths.npz", paths=np.array(paths))
            np.save(self.index_dir / "features.npy", normalized)
            
            return True
        except Exception as e:
            print(f"Error creating index: {e}")
            return False
    
    def load_index(self) -> Tuple[List[str], np.ndarray]:
        """Load existing index"""
        paths_file = self.index_dir / "paths.npz"
        features_file = self.index_dir / "features.npy"
        
        if not paths_file.exists() or not features_file.exists():
            return [], np.empty((0, self.embed_dim), dtype=np.float32)
        
        paths = np.load(paths_file, allow_pickle=True)["paths"].tolist()
        features = np.load(features_file)
        
        return paths, features
