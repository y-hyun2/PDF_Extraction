"""
RAG Answer Generator using Multimodal LLM (Gemma-3n-E4B-it).

Logic:
1. Search Vector DB for query.
2. Retrieve Top-K chunks (Markdown + Metadata).
3. Fetch corresponding Page Images.
4. Construct Multimodal Prompt (Text + Images).
5. Generate Answer.

Handling "All different pages":
- We prioritize unique pages from the Top-K results.
- We limit to N images (e.g., 3) to prevent context overflow.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# Load environment variables (HF_TOKEN)
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Import existing search logic
sys.path.append(str(Path(__file__).parent))
from search_vector_db import search_vector_db  # We might need to refactor search_vector_db to be importable or copy logic
# actually, search_vector_db.py has a main(), let's extract the search function if possible or just import it if it has a clean function.
# Looking at search_vector_db.py content from memory/files... it has search_vector_db function.

def load_model(model_id: str):
    print(f"📦 Loading Model '{model_id}'...")
    try:
        # Assuming Gemma 3n uses standard AutoClasses or similar
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        print("✅ Model Loaded.")
        return processor, model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

def get_unique_pages_from_results(results: List[Dict], max_images: int = 3):
    """
    Extract unique page info (image path) from search results.
    Preserve order of relevance (Rank 1 page first).
    """
    seen_pages = set()
    unique_contexts = []
    
    start_dir = Path("data/pages_structured") # Base path for images
    
    for res in results:
        # metadata contains 'image_path' (ref to page image usually) or we infer it
        # Actually vector_db metadata might mostly have chunk text and IDs.
        # We need to look up the Page Image Path from the Doc/Page ID.
        # Format of ID: doc_{doc_id}_page_{page_no}_chunk_{chunk_id}
        # Metadata: 'source', 'page_no', 'doc_id', etc.
        
        meta = res.get('metadata', {})
        doc_name = meta.get('company', 'Unknown') + "_" + str(meta.get('year', '2023')) # Heuristic
        # Better: use the 'source' field which is formatted like "HDEC (2023) | p.11"
        # Or construct path: data/pages_structured/{ReportName}/page_{page_no}/page.png
        
        # We need the Report Directory Name. 
        # Metadata 'filename' is stored in Chroma? Let's check search_vector_db output structure.
        # It prints "Source: HDEC (2023) | p.11".
        
        # Let's assume we can map back to file path or we query DB.
        # For prototype, we will SEARCH for the page image based on page_no.
        # Assuming single report or simple mapping.
        
        # Let's try to pass the 'doc_name' (e.g. 2025_HDEC_Report) as arg or infer.
        # If we have multiple reports, we need to know WHICH report the chunk belongs to.
        # Our Chroma metadata SHOULD have 'doc_name' or 'filename'.
        
        # Assuming we can find the image:
        # For now, let's look at `res['metadata']`.
        pass 
        
    return []

# Refined Plan: 
# We need `search_vector_db` to return METADATA including file paths or doc names.
# Current `search_vector_db.py` prints results. We should import `query_vector_db` from `build_vector_db`? 
# or modify `search_vector_db.py` to return data.


def get_page_image_path(doc_name: str, page_no: int) -> Path:
    """
    Construct path to page image.
    Assumption: doc_name matches folder in data/pages_structured.
    Example: 2023_HDEC_Report -> data/pages_structured/2023_HDEC_Report/page_0011/page.png
    """
    base_dir = Path("data/pages_structured")
    
    # Try exact match first
    candidate = base_dir / doc_name / f"page_{page_no:04d}" / "page.png"
    if candidate.exists():
        return candidate
        
    # Try finding in sanitized/subdirs if exact mismatch (common issue)
    # The doc_name in metadata might be "HDEC_2023" but folder is "2023_HDEC_Report"
    # Metadata 'company_name'='HDEC', 'report_year'=2023.
    # We need to scan base_dir for folder matching year and company?
    # Heuristic: Scan for year
    for folder in base_dir.iterdir():
        if not folder.is_dir(): continue
        if str(page_no) in folder.name: # Wrong, folder is report name
             pass
        # Check if folder name contains the doc_name parts?
        # Let's try flexible match
        # Normalize: 2023 in name AND HDEC (or company) in name?
        pass # To simplify, we rely on metadata HAVING the exact folder name if possible?
             # Currently metadata has 'company_name' and 'report_year'.
             # We might need to map "HDEC" + "2023" -> "2023_HDEC_Report".
    
    # Fallback: check 2025_HDEC_Report (hardcoded check for common ones)
    fallback_folder = f"{2023 if '2023' in doc_name else 2025}_HDEC_Report"
    candidate = base_dir / fallback_folder / f"page_{page_no:04d}" / "page.png"
    if candidate.exists():
        return candidate
    candidate_sanitized = base_dir / f"{fallback_folder}_sanitized" / f"page_{page_no:04d}" / "page.png"
    if candidate_sanitized.exists():
        return candidate_sanitized
        
    return None

def main():
    parser = argparse.ArgumentParser(description="RAG Answer Generator")
    parser.add_argument("query", type=str, help="Question to ask")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it", help="Model ID") 
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    
    # Filter args
    parser.add_argument("--company", type=str, default=None, help="Filter by Company Name (e.g. HDEC)")
    parser.add_argument("--year", type=int, default=None, help="Filter by Report Year (e.g. 2023)")
    
    args = parser.parse_args()
    
    # 1. Search
    print(f"🔎 Searching for: '{args.query}'")
    if args.company or args.year:
        print(f"   Filters: Company='{args.company}', Year='{args.year}'")

    results = search_vector_db(
        args.query, 
        top_k=args.top_k, 
        filter_company=args.company, 
        filter_year=args.year
    )
    
    if not results:
        print("Test ended: No results found.")
        return

    # 2. Process Contexts & Images
    unique_pages = {} # page_key -> {image_path, texts}
    
    for res in results:
        meta = res['metadata']
        doc_year = meta.get('report_year', 2023)
        company = meta.get('company_name', 'HDEC')
        page_no = meta.get('page_no')
        chunk_text = res['content']
        
        # Key for unique page
        page_key = f"{company}_{doc_year}_{page_no}"
        
        if page_key not in unique_pages:
            # Find image
            doc_name_hint = f"{doc_year}_{company}_Report"
            img_path = get_page_image_path(doc_name_hint, page_no)
            
            unique_pages[page_key] = {
                "image_path": img_path,
                "texts": [],
                "page_info": f"{company} {doc_year} Sustainability Report (Page {page_no})"
            }
        
        unique_pages[page_key]["texts"].append(chunk_text)

    # 3. Load Model
    # ... (Model loading logic remains same) ...
    print(f"📦 Loading Model '{args.model}'...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # Changed to bfloat16 for stability
            trust_remote_code=True,
            token=HF_TOKEN 
        ).eval()
        processor = AutoProcessor.from_pretrained(
            args.model, 
            trust_remote_code=True,
            token=HF_TOKEN
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        if "gated" in str(e).lower() or "401" in str(e):
             print("💡 Tip: Ensure HF_TOKEN is set in .env and you have access to the model.")
        return

    # 4. Construct Prompt
    # We will feed images + text context.
    # Handling multiple images: Most VLMs support list of images or interleaved.
    # We will format as a chat.
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Answer the user's question based on the provided visual and text context."},
    ]
    
    user_content = []
    user_content.append({"type": "text", "text": f"Question: {args.query}\n\nContexts:\n"})
    
    images_loaded = []
    
    # Iterating pages (limit to top 3 unique pages to save VRAM)
    for i, (key, data) in enumerate(list(unique_pages.items())[:3]):
        if data["image_path"]:
            user_content.append({"type": "text", "text": f"--- Page Image ({data['page_info']}) ---\n"})
            user_content.append({"type": "image", "image": str(data["image_path"])}) # Processor handles path string or PIL
            images_loaded.append(data["image_path"])
        
        texts_combined = "\n... \n".join(data["texts"])
        user_content.append({"type": "text", "text": f"\n[Extracted Text for {data['page_info']}]:\n{texts_combined}\n\n"})

    user_content.append({"type": "text", "text": "Answer:"})
    messages.append({"role": "user", "content": user_content})

    # 5. Generate
    print("🤖 Generating Answer...")
    
    # Prepare inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Load PIL images
    pil_images = [Image.open(p) for p in images_loaded] if images_loaded else None
    
    inputs = processor(
        text=[text],
        images=pil_images,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("\n" + "="*40)
    print("📝 Answer:")
    print("="*40)
    print(output_text)
    print("="*40)

if __name__ == "__main__":
    main()
