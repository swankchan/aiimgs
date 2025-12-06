"""PDF processing utilities for image extraction and text analysis."""

import re
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import json

try:
    import streamlit as st
    from streamlit.runtime.uploaded_file_manager import UploadedFile as StreamlitUploadedFile
    STREAMLIT_SUPPORT = True
except ImportError:
    STREAMLIT_SUPPORT = False
    StreamlitUploadedFile = None

from PIL import Image

# PDF processing related
try:
    import PyPDF2
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# AI analysis support
try:
    import requests
    AI_SUPPORT = True
except ImportError:
    AI_SUPPORT = False


def generate_pdf_thumbnail(pdf_file: Union[BytesIO, 'StreamlitUploadedFile'], output_path: Path, max_width: int = 400) -> bool:
    """
    Generate a thumbnail image from the first page of a PDF
    
    Args:
        pdf_file: PDF file (BytesIO or Streamlit UploadedFile)
        output_path: Path to save thumbnail image
        max_width: Maximum width for thumbnail
    
    Returns:
        True if successful, False otherwise
    """
    if not PDF_SUPPORT:
        return False
    
    try:
        # Read PDF bytes
        if isinstance(pdf_file, BytesIO):
            pdf_bytes = pdf_file.getvalue()
        else:
            pdf_bytes = pdf_file.read()
            # Reset file pointer if possible
            if hasattr(pdf_file, 'seek'):
                pdf_file.seek(0)
        
        # Open PDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if len(doc) == 0:
            doc.close()
            return False
        
        # Get first page
        page = doc[0]
        
        # Render page to image (higher DPI for better quality)
        zoom = 2.0  # 2x zoom for better quality
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Resize if needed
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
        # Save thumbnail
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, "JPEG", quality=85)
        
        doc.close()
        return True
        
    except Exception as e:
        print(f"Error generating PDF thumbnail: {e}")
        return False


def save_pdf_to_catalog(pdf_file: Union[BytesIO, 'StreamlitUploadedFile'], catalog_folder: Path, filename: str = None) -> Path:
    """
    Save uploaded PDF to catalog folder and generate thumbnail
    
    Args:
        pdf_file: Uploaded PDF file (BytesIO or Streamlit UploadedFile)
        catalog_folder: Target folder for PDF storage
        filename: Original filename (required if pdf_file is BytesIO)
    
    Returns:
        Path to saved PDF file
    """
    catalog_folder.mkdir(parents=True, exist_ok=True)
    
    # Get filename
    if hasattr(pdf_file, 'name'):
        pdf_filename = Path(pdf_file.name).name
    elif filename:
        pdf_filename = filename
    else:
        pdf_filename = "uploaded.pdf"
    
    dest_path = catalog_folder / pdf_filename
    
    # Avoid duplicate filenames
    counter = 1
    stem = Path(pdf_filename).stem
    original_stem = stem
    while dest_path.exists():
        stem = f"{original_stem}_{counter}"
        dest_path = catalog_folder / f"{stem}.pdf"
        counter += 1
    
    # Save PDF
    with open(dest_path, "wb") as f:
        if isinstance(pdf_file, BytesIO):
            pdf_file.seek(0) if hasattr(pdf_file, 'seek') else None
            f.write(pdf_file.getvalue())
        else:
            f.write(pdf_file.read())
    
    # Generate thumbnail from first page
    thumbnail_path = catalog_folder / f"{stem}_preview.jpg"
    if isinstance(pdf_file, BytesIO):
        pdf_file.seek(0) if hasattr(pdf_file, 'seek') else None
    generate_pdf_thumbnail(pdf_file, thumbnail_path)
    
    return dest_path


def extract_images_from_pdf(
    pdf_file: Union[BytesIO, 'StreamlitUploadedFile'],
    output_folder: Path,
    jpeg_quality: int = 100,
    pdf_filename: str = None
) -> Tuple[List[Path], str]:
    """
    Extract all embedded images from PDF and save them
    
    Args:
        pdf_file: Uploaded PDF file (BytesIO or Streamlit UploadedFile)
        output_folder: Image output folder
        jpeg_quality: JPEG quality for saved images
        pdf_filename: Original PDF filename (required if pdf_file is BytesIO)
    
    Returns:
        (List of image paths, PDF filename)
    """
    if not PDF_SUPPORT:
        raise ImportError("PDF support not available. Install PyMuPDF (fitz).")
    
    # Get PDF filename
    if hasattr(pdf_file, 'name'):
        pdf_name = Path(pdf_file.name).stem
    elif pdf_filename:
        pdf_name = Path(pdf_filename).stem
    else:
        pdf_name = "uploaded"
    
    # Read PDF bytes
    if isinstance(pdf_file, BytesIO):
        pdf_bytes = pdf_file.getvalue()
    else:
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
                img_filename = f"{pdf_name}_img{img_counter}.jpg"
                img_path = output_folder / img_filename
                
                # Avoid duplicate filenames
                counter = 1
                while img_path.exists():
                    img_filename = f"{pdf_name}_img{img_counter}_{counter}.jpg"
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
                
                img.save(img_path, "JPEG", quality=jpeg_quality)
                image_paths.append(img_path)
                img_counter += 1
                
            except Exception as e:
                if STREAMLIT_SUPPORT:
                    st.warning(f"Unable to extract image (page {page_num + 1}, img {img_index + 1}): {str(e)}")
                else:
                    print(f"Unable to extract image (page {page_num + 1}, img {img_index + 1}): {str(e)}")
                continue
    
    doc.close()
    
    return image_paths, pdf_name


def extract_text_from_pdf(pdf_file: Union[BytesIO, 'StreamlitUploadedFile']) -> str:
    """
    Extract all text from PDF
    
    Args:
        pdf_file: Uploaded PDF file (BytesIO or Streamlit UploadedFile)
    
    Returns:
        Extracted text content
    """
    if not PDF_SUPPORT:
        raise ImportError("PDF support not available. Install PyPDF2.")
    
    # Read PDF bytes
    if isinstance(pdf_file, BytesIO):
        pdf_bytes = pdf_file.getvalue()
    else:
        pdf_bytes = pdf_file.read()
    
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    
    all_text = []
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            all_text.append(text)
    
    return "\n\n".join(all_text)


def extract_keywords_from_text(text: str, max_keywords: int = 5, ai_info: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Extract keywords from text using multiple fallback methods
    
    Args:
        text: Text content
        max_keywords: Maximum number of keywords to return
        ai_info: AI extracted information (Client, Location, Contractor, Date, Role) - optional
    
    Returns:
        List of keywords
    """
    keywords = []
    
    # Method 1: AI extracted structured information (if available)
    if ai_info:
        priority_fields = ["client", "location", "contractor", "role", "date_of_completion"]
        for field in priority_fields:
            value = ai_info.get(field, "")
            if value and value != "Not found":
                parts = value.replace(",", " ").replace("Ltd", "").replace("Limited", "").split()
                for part in parts:
                    cleaned = part.strip("()[]")
                    if len(cleaned) > 2 and cleaned.lower() not in ["not", "found"]:
                        keywords.append(cleaned)
    
    # Method 2: Pattern-based extraction (works without AI)
    # Look for common document patterns
    patterns = {
        "Project": r"(?:Project|Development|Building|Construction)[\s:]+([A-Z][A-Za-z\s&]+?)(?:\n|,|\.)",
        "Location": r"(?:Location|Address|Site|at)[\s:]+([A-Z][A-Za-z\s,]+?)(?:\n|Project|Client)",
        "Client": r"(?:Client|Owner|For)[\s:]+([A-Z][A-Za-z\s&]+?)(?:\n|Contractor|Project)",
        "Year": r"\b(19|20)\d{2}\b"
    }
    
    for label, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches[:2]:  # Limit to 2 matches per pattern
            if isinstance(match, tuple):
                match = match[0] if match[0] else match[1]
            cleaned = match.strip()
            if len(cleaned) > 3:
                keywords.append(cleaned)
    
    # Method 3: Capitalized phrases (likely to be proper nouns/names)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', text)
    keywords.extend(capitalized[:max_keywords])
    
    # Method 4: Word frequency (fallback)
    if len(keywords) < max_keywords:
        stop_words = {
            "the", "and", "for", "with", "this", "that", "from", "have", "been",
            "will", "page", "project", "building", "construction", "document"
        }
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        filtered = [w for w in words if w not in stop_words]
        word_counts = Counter(filtered)
        keywords.extend([w for w, c in word_counts.most_common(max_keywords * 2)])
    
    # Return unique keywords
    seen = set()
    unique_keywords = []
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen and len(kw) > 2:
            seen.add(kw_lower)
            unique_keywords.append(kw)
            if len(unique_keywords) >= max_keywords * 2:
                break
    
    return unique_keywords


def extract_keywords_by_method(text: str, max_keywords: int = 5, ai_info: Optional[Dict[str, str]] = None) -> Dict[str, List[str]]:
    """
    Extract keywords from text using multiple methods, returning them grouped by method
    
    Args:
        text: Text content
        max_keywords: Maximum number of keywords per method
        ai_info: AI extracted information - optional
    
    Returns:
        Dictionary with keywords grouped by extraction method
    """
    result = {
        "AI Analysis": [],
        "Pattern-based": [],
        "Capitalized Phrases": [],
        "Frequency Analysis": []
    }
    
    # Method 1: AI extracted structured information (if available)
    if ai_info:
        priority_fields = ["client", "location", "contractor", "role", "date_of_completion", "project_name"]
        for field in priority_fields:
            value = ai_info.get(field, "")
            if value and value != "Not found":
                parts = value.replace(",", " ").replace("Ltd", "").replace("Limited", "").split()
                for part in parts:
                    cleaned = part.strip("()[]")
                    if len(cleaned) > 2 and cleaned.lower() not in ["not", "found"]:
                        result["AI Analysis"].append(cleaned)
    
    # Method 2: Pattern-based extraction
    patterns = {
        "Project": r"(?:Project|Development|Building|Construction)[\s:]+([A-Z][A-Za-z\s&]+?)(?:\n|,|\.)",
        "Location": r"(?:Location|Address|Site|at)[\s:]+([A-Z][A-Za-z\s,]+?)(?:\n|Project|Client)",
        "Client": r"(?:Client|Owner|For)[\s:]+([A-Z][A-Za-z\s&]+?)(?:\n|Contractor|Project)",
        "Year": r"\b(19|20)\d{2}\b"
    }
    
    for label, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches[:2]:
            if isinstance(match, tuple):
                match = match[0] if match[0] else match[1]
            cleaned = match.strip()
            if len(cleaned) > 3:
                result["Pattern-based"].append(cleaned)
    
    # Method 3: Capitalized phrases
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', text)
    result["Capitalized Phrases"] = capitalized[:max_keywords * 2]
    
    # Method 4: Word frequency
    stop_words = {
        "the", "and", "for", "with", "this", "that", "from", "have", "been",
        "will", "page", "project", "building", "construction", "document"
    }
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in stop_words]
    word_counts = Counter(filtered)
    result["Frequency Analysis"] = [w for w, c in word_counts.most_common(max_keywords * 2)]
    
    # Remove duplicates within each method while preserving order
    for method in result:
        seen = set()
        unique = []
        for kw in result[method]:
            kw_lower = kw.lower()
            if kw_lower not in seen and len(kw) > 2:
                seen.add(kw_lower)
                unique.append(kw)
        result[method] = unique[:max_keywords * 2]
    
    return result


def analyze_pdf_with_ai(
    text: str,
    custom_fields: Optional[List[str]] = None,
    llm_url: str = "http://localhost:11434",
    model: str = "llama3.2:3b"
) -> Dict[str, str]:
    """
    Use local AI (Ollama) to intelligently analyze PDF content and extract information
    Works even without explicit labels like "Project Name:" or "Location:"
    
    Args:
        text: PDF text content
        custom_fields: List of fields to extract (e.g., ["Location", "Client", "Role", "Project Name"])
        llm_url: LLM API URL (default: http://localhost:11434)
        model: LLM model to use (default: llama3.2:3b, alternatives: llama3.1, mistral)
    
    Returns:
        Dictionary with extracted information
    """
    if not AI_SUPPORT:
        raise ImportError("AI support not available. Install: pip install requests")
    
    ###############################
    # Information to be extracted #
    ###############################
    if custom_fields is None:
        custom_fields = ["Project Title", "Location of the Buildings", "Client's Name", "Contractor's Name", "Role"]
    
    # Truncate text if too long
    max_chars = 6000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Text truncated...]"
    
    # Build intelligent prompt
    fields_list = ", ".join(custom_fields)
    prompt = f"""You are analyzing a PDF about Building Project of a Construction company. I need you to extract the following information: {fields_list} in order to build a database for making further reference. Please return the information.    

Document text:
{text}

Instructions:
- Read the document carefully and understand the context
- Extract the requested information even if there are no explicit labels
- For "Project Name": identify what project this document is about
- For "Location": find any place, city, or address mentioned
- For "Client": identify the client, customer, or company being served
- For "Contractor": identify the main contractor or construction company
- For "Date of Completion": find project completion date, year, or time period
- For "Role": determine the author's role or position in this project
- For "Description": provide a brief 1-2 sentence summary
- If information is not found, write "Not found"

Respond ONLY with valid JSON (no markdown, no explanation):
{{"project_name": "...", "location": "...", "client": "...", "contractor": "...", "date_of_completion": "...", "role": "...", "description": "..."}}

JSON:"""
    
    try:
        # Call Ollama API
        response = requests.post(
            f"{llm_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1, # Lower for more consistent extraction
                    "num_predict": 1000 # Number of Tokens for 700 English Wordings
                }
            },
            timeout=600
        )
        
        if response.status_code != 200:
            return {"error": f"Ollama error: {response.status_code}"}
        
        # Parse response
        result_text = response.json().get("response", "").strip()
        
        print(f"[AI Debug] Raw response length: {len(result_text)}")
        print(f"[AI Debug] First 500 chars: {result_text[:500]}")
        
        # Extract JSON from response (more robust)
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
            print(f"[AI Debug] Removed ```json markers")
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
            print(f"[AI Debug] Removed ``` markers")
        
        # Find JSON object - must be a complete object
        start = result_text.find("{")
        if start == -1:
            return {"error": "No JSON object found", "raw": result_text[:200]}
            
        # Count braces to find the matching closing brace
        brace_count = 0
        end = -1
        for i in range(start, len(result_text)):
            if result_text[i] == '{':
                brace_count += 1
            elif result_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        
        if end == -1:
            return {"error": "Incomplete JSON object", "raw": result_text[:500]}
        
        json_text = result_text[start:end].strip()
        print(f"[AI Debug] Extracted JSON length: {len(json_text)} chars")
        
        # Clean up the JSON text before parsing
        import re
        
        # Fix common issues with AI-generated JSON:
        # 1. Remove trailing commas before } and ]
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)
        
        # 2. Replace literal newlines within string values with spaces
        # This handles multiline strings that should be on one line
        # We need to be careful to only replace newlines INSIDE quoted strings
        def fix_newlines_in_strings(text):
            result = []
            in_string = False
            escape_next = False
            
            for i, char in enumerate(text):
                if escape_next:
                    result.append(char)
                    escape_next = False
                    continue
                    
                if char == '\\':
                    result.append(char)
                    escape_next = True
                    continue
                    
                if char == '"':
                    in_string = not in_string
                    result.append(char)
                    continue
                    
                if in_string and char in '\n\r\t':
                    # Replace control characters with space inside strings
                    result.append(' ')
                else:
                    result.append(char)
            
            return ''.join(result)
        
        json_text = fix_newlines_in_strings(json_text)
        
        # Try to parse JSON
        try:
            result_dict = json.loads(json_text)
            print(f"[AI Debug] Successfully parsed: {list(result_dict.keys())}")
            return result_dict
        except json.JSONDecodeError as je:
            print(f"[AI Debug] JSON parse error at position {je.pos}: {je.msg}")
            print(f"[AI Debug] Context around error: ...{json_text[max(0,je.pos-100):je.pos+100]}...")
            return {"error": "Invalid JSON response", "raw": result_text[:500]}
        
    except requests.exceptions.ConnectionError:
        return {"error": "Ollama not running. Start with: ollama serve"}
    except requests.exceptions.Timeout:
        return {"error": "Timeout. Try a shorter document"}
    except Exception as e:
        import traceback
        print(f"[AI Debug] Exception: {e}")
        traceback.print_exc()
        return {"error": str(e)}


def generate_smart_caption(
    pdf_info: Dict[str, str],
    template: Optional[str] = None
) -> str:
    """
    Generate a smart caption based on extracted PDF information
    
    Args:
        pdf_info: Dictionary with extracted information (from analyze_pdf_with_ai)
        template: Custom template string with placeholders like {project_name}, {location}
                 If None, will use default template
    
    Returns:
        Generated caption string
    """
    # Default template if not specified
    if template is None:
        # Use project_name as main caption, with other info as supplementary
        template = "{project_name}"
    
    # Replace placeholders with actual values
    caption = template
    for key, value in pdf_info.items():
        placeholder = "{" + key + "}"
        if placeholder in caption:
            caption = caption.replace(placeholder, value if value != "Not found" else "")
    
    # Clean up extra spaces
    caption = " ".join(caption.split())
    
    return caption
