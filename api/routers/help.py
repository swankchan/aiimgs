"""
FAQ Chatbot router using Ollama gemma3:1b
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import requests
import json
from pathlib import Path

router = APIRouter()

# Load config
def load_config():
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

CONFIG = load_config()
OLLAMA_URL = CONFIG.get("pdf", {}).get("ai_analysis", {}).get("llm_url", "http://localhost:11434")
MODEL = CONFIG.get("pdf", {}).get("ai_analysis", {}).get("model", "gemma3:1b")

SYSTEM_PROMPT = """You are a helpful assistant for the AI Imagery Search application. Your role is to help users understand and use the application effectively.

**Application Features:**
1. **Search** - Semantic image search using AI (CLIP model). Users can search by text description or upload an image to find similar images.
2. **Indexing** - Upload and index new images or PDFs. PDFs are automatically extracted into individual images.
3. **Library** - Browse all uploaded images and PDFs. View thumbnails, metadata, and AI-generated captions.
4. **Admin** - User management (admin only). Create, edit, and delete users.

**Key Capabilities:**
- Semantic search: Find images by describing what you see (e.g., "red building", "people walking")
- PDF support: Upload PDFs and extract images automatically. First page is shown as thumbnail.
- AI analysis: Automatic caption and metadata generation for images
- Metadata: View and edit image captions and keywords

**Common Questions:**
- How to upload images? Go to Indexing page, select images, and click Upload.
- How to search? Go to Search page, enter text description or upload a reference image.
- How to view PDFs? Go to Library, select "PDFs" tab to see all uploaded PDFs with thumbnails.
- What file formats? Supports JPG, PNG, BMP, GIF, WEBP images and PDF documents.

Keep answers concise and friendly. If you don't know something about the application, admit it and suggest contacting support."""


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = False


class ChatResponse(BaseModel):
    message: str
    model: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with FAQ assistant using Ollama
    """
    try:
        # Prepare messages with system prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        # Add conversation history
        for msg in request.messages:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Call Ollama API
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            },
            timeout=30
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=500, 
                detail=f"Ollama API error: {response.text}"
            )
        
        result = response.json()
        assistant_message = result.get("message", {}).get("content", "")
        
        return ChatResponse(
            message=assistant_message,
            model=MODEL
        )
        
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Ollama request timed out")
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503, 
            detail="Cannot connect to Ollama. Make sure Ollama is running."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/health")
async def check_ollama():
    """
    Check if Ollama is available
    """
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = response.json().get("models", [])
        model_names = [m.get("name") for m in models]
        
        return {
            "status": "ok",
            "ollama_url": OLLAMA_URL,
            "configured_model": MODEL,
            "model_available": MODEL in model_names,
            "available_models": model_names
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "ollama_url": OLLAMA_URL
        }
