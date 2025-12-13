"""Generate thumbnails for existing PDFs in catalog folder"""
from pathlib import Path
import sys

# Make sure we can import from current directory
sys.path.insert(0, str(Path(__file__).parent))

from pdf_utils import generate_pdf_thumbnail
from io import BytesIO

def generate_thumbnails_for_catalog():
    """Generate thumbnails for all PDFs in catalog folder"""
    catalog_folder = Path("catalog")
    
    if not catalog_folder.exists():
        print("Catalog folder not found")
        return
    
    pdf_files = list(catalog_folder.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_path in pdf_files:
        # Check if thumbnail already exists
        thumbnail_path = catalog_folder / f"{pdf_path.stem}_preview.jpg"
        
        if thumbnail_path.exists():
            print(f"✓ Thumbnail already exists: {thumbnail_path.name}")
            continue
        
        print(f"Generating thumbnail for: {pdf_path.name}...", end=" ")
        
        try:
            # Read PDF
            with open(pdf_path, 'rb') as f:
                pdf_bytes = BytesIO(f.read())
            
            # Generate thumbnail
            success = generate_pdf_thumbnail(pdf_bytes, thumbnail_path)
            
            if success:
                print("✓ Success")
            else:
                print("✗ Failed")
                
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        generate_thumbnails_for_catalog()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
