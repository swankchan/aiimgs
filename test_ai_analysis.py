"""
Simple test script for CLIP-based PDF analysis feature

Run this to test the analysis without running the full Streamlit app.
"""

import os
from pathlib import Path

# Test if required packages are available
try:
    import open_clip
    import torch
    print("✓ OpenCLIP and PyTorch installed")
except ImportError as e:
    print(f"✗ Required packages not installed: {e}")
    print("  Run: pip install open_clip_torch torch")
    exit(1)

# Test PDF utils import
try:
    from pdf_utils import analyze_pdf_with_clip, generate_smart_caption
    print("✓ PDF utils imported successfully")
except ImportError as e:
    print(f"✗ Failed to import pdf_utils: {e}")
    exit(1)

# Sample PDF text for testing
sample_text = """
PROJECT PORTFOLIO

Project Name: Hong Kong Central Tower Renovation
Location: Central, Hong Kong
Client: ABC Development Ltd.
Role: Senior Architect
Project Duration: January 2024 - December 2024
Budget: HK$ 50 million

Description:
This project involves the complete renovation of a historic building in Central Hong Kong.
The scope includes structural reinforcement, facade restoration, and interior modernization
while preserving the building's heritage value. The team consists of 15 professionals
including architects, engineers, and contractors.

Key Achievements:
- Completed ahead of schedule
- Reduced costs by 15%
- Received Green Building Award 2024
"""

print("\n" + "="*60)
print("Testing CLIP-based PDF Analysis...")
print("="*60)

try:
    # Load CLIP model
    print("\nLoading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"✓ CLIP model loaded on {device}")
    
    # Test analysis without CLIP
    print("\n--- Test 1: Pattern Matching Only ---")
    result = analyze_pdf_with_clip(
        sample_text,
        custom_fields=["Project Name", "Location", "Client", "Role", "Date", "Description"],
        clip_model=None,
        tokenizer=None
    )
    
    print("\n✓ Analysis Result (Pattern Matching):")
    for field, value in result.items():
        print(f"  {field}: {value}")
    
    # Test analysis with CLIP
    print("\n--- Test 2: Pattern Matching + CLIP Semantic Search ---")
    result2 = analyze_pdf_with_clip(
        sample_text,
        custom_fields=["Project Name", "Location", "Client", "Role", "Date", "Description"],
        clip_model=model,
        tokenizer=tokenizer
    )
    
    print("\n✓ Analysis Result (with CLIP):")
    for field, value in result2.items():
        print(f"  {field}: {value}")
    
    # Test caption generation
    caption = generate_smart_caption(result2, template="{project_name}")
    print(f"\n✓ Generated Caption: {caption}")
    
    # Test custom template
    caption2 = generate_smart_caption(result2, template="{project_name} - {location}")
    print(f"✓ Custom Template Caption: {caption2}")
    
    caption3 = generate_smart_caption(result2, template="[{location}] {project_name} | {client}")
    print(f"✓ Complex Template Caption: {caption3}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    print("\nTo enable in app: Set 'enabled': true in config.json")
    
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
