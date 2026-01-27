"""PDF Sanitizer tool.

Docling 등의 도구가 인코딩 문제로 PDF를 제대로 읽지 못할 때,
화면에 보이는 요소를 기반으로 PDF를 재생성(Reconstruction)하여 문제를 해결하는 도구.
텍스트를 깨끗한 폰트(NanumGothic)로 다시 쓰고, 선/도형/이미지를 복원한다.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import fitz  # PyMuPDF
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def sanitize_pdf(pdf_path: str | Path, page_range: Optional[Tuple[int, int]] = None) -> Path:
    """
    Rebuilds the PDF by copying text with a clean font (NanumGothic)
    and faithfully reconstructing lines, shapes, and images to support Vision AI.
    
    Args:
        pdf_path: Path to source PDF.
        page_range: (start, end) 0-based page indices.
        
    Returns:
        Path to the sanitized PDF.
    """
    pdf_path = Path(pdf_path)
    print(f"--- [Sanitization] Starting Full Reconstruction for: {pdf_path.name} ---")
    
    src_doc = fitz.open(pdf_path)
    clean_doc = fitz.open()
    
    # Determine output path
    sanitized_path = pdf_path.with_stem(pdf_path.stem + "_sanitized").with_suffix(".pdf")
    
    # Optimization: Reuse existing sanitized PDF if recently created
    if sanitized_path.exists():
        print(f"  [Sanitization] Found existing sanitized file: {sanitized_path}. Reusing it.")
        return sanitized_path

    # Determine range
    start_i = page_range[0] if page_range else 0
    end_i = page_range[1] if page_range else len(src_doc)
    
    # Validations
    start_i = max(0, start_i)
    end_i = min(len(src_doc), end_i)
    
    print(f"  Processing pages {start_i} to {end_i}...")

    # Font configuration
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    font_name = "nanum"
    has_font = os.path.exists(font_path)
    if not has_font:
         print(f"  Warning: Korean font not found at {font_path}, fallback to helv.")
         font_name = "helv"

    for i in range(start_i, end_i):
        page = src_doc[i]
        # Create new page
        new_page = clean_doc.new_page(width=page.rect.width, height=page.rect.height)
        
        # --- 1. Reconstruct Drawings (Lines, Rects for Table Borders) ---
        try:
            paths = page.get_drawings()
            for path in paths:
                color = path.get("color")
                fill = path.get("fill")
                width = path.get("width", 1)
                
                # SMART CONTRAST ENHANCEMENT
                # Diagnosis: PDF uses 0.25pt (Thin) + ~0.5 (Gray) RGB, making lines invisible.
                # Fix: If it's a "line" (stroke) and the color is "Gray-ish" (not White/Light), snap to BLACK.
                # This boosts visibility without changing width (hierarchy).
                if color: 
                    # Check if it's not too light (e.g. avoid turning light gray fills into black)
                    if max(color) < 0.9: 
                        color = (0, 0, 0)

                for item in path["items"]:
                    if item[0] == "l": # Line
                        new_page.draw_line(item[1], item[2], color=color, width=width)
                    elif item[0] == "re": # Rect
                        new_page.draw_rect(item[1], color=color, fill=fill, width=width)
                    elif item[0] == "c": # Curve
                        new_page.draw_bezier(item[1], item[2], item[3], item[4], color=color, fill=fill, width=width)
        except Exception:
            pass # Suppress minor drawing errors

        # --- 2. Reconstruct Images ---
        try:
            image_infos = page.get_image_info()
            for img in image_infos:
                xref = img.get("xref")
                if xref:
                    try:
                        base_img = src_doc.extract_image(xref)
                        if base_img:
                            new_page.insert_image(img["bbox"], stream=base_img["image"])
                    except Exception:
                        continue
        except Exception:
            pass

        # --- 3. Reconstruct Text (Clean Font) ---
        # Register Font Per Page if needed (to be safe)
        if has_font:
            try:
                new_page.insert_font(fontname=font_name, fontfile=font_path)
            except Exception:
                font_name = "helv"
        
        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        try:
                            clean_text = span["text"]
                            # Fix specific artifacts
                            # U+302E (Hangul Single Dot Tone Mark) -> Middle Dot
                            clean_text = clean_text.replace("\u302e", "·")
                            # U+2219 (Bullet Operator) -> Middle Dot
                            clean_text = clean_text.replace("\u2219", "·")
                            
                            new_page.insert_text(
                                fitz.Point(span["origin"]), 
                                clean_text, 
                                fontsize=span["size"] * 0.9, # Reduce size to prevent overlap (Nanum is wider)
                                fontname=font_name 
                            )
                        except Exception:
                            pass

    clean_doc.save(sanitized_path)
    clean_doc.close()
    src_doc.close()
    print(f"--- [Sanitization] Completed: {sanitized_path} ---")
    return sanitized_path


def check_docling_compatibility(pdf_path: Path) -> bool:
    """
    Docling이 이 PDF를 정상적으로 읽을 수 있는지 테스트한다.
    - OCR을 끄고 빠르게 변환 시도
    - 인코딩 오류가 발생하거나 텍스트가 텅 비었으면 False 반환
    - 첫 페이지만 체크하면 일부만 정상이고 뒤가 깨지는 경우를 놓칠 수 있으므로,
      [1페이지, 중간 페이지, 마지막 페이지]를 샘플링하여 검사한다.
    """
    print(f"[Check] Verifying Docling compatibility for: {pdf_path.name}")
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False 
    pipeline_options.do_table_structure = False 
    
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    
    try:
        # Get total page count first
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            
        # Sample strategy: First, Middle, Last
        if total_pages < 3:
            test_pages = list(range(1, total_pages + 1))
        else:
            test_pages = [1, total_pages // 2, total_pages]
            
        print(f"  Sampling pages {test_pages} for validation...")
        
        # Docling accepts specific page numbers, but standard convert() takes range or we call it multiple times.
        # To be efficient, we can check them one by one or in a batch. 
        # Docling's page_range arg is (start, end) inclusive, doesn't support list.
        # So we iterate.
        
        for p in test_pages:
            res = converter.convert(pdf_path, page_range=(p, p))
            markdown = res.document.export_to_markdown().strip()
            
            if not markdown:
                print(f"  ⚠️ Content missing on page {p}. Likely encoding issue.")
                return False
            
            # Additional Heuristic: Check for replacement characters if needed
            # if "\ufffd" in markdown: ...
            
        print("  ✅ Docling can read sampled pages successfully.")
        return True
        
    except Exception as e:
        print(f"  ❌ Docling extraction failed: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="PDF Reconstruction Tool for Encoding/Vision Issues")
    parser.add_argument("--pdf", type=Path, required=True, help="Path to the source PDF file")
    parser.add_argument("--force", action="store_true", help="Skip validation and force sanitization")
    
    args = parser.parse_args()
    
    if not args.pdf.exists():
        print(f"Error: File not found: {args.pdf}")
        return 1
        
    # Check compatibility unless forced
    if not args.force:
        is_compatible = check_docling_compatibility(args.pdf)
        if is_compatible:
            print("\n[Pass] This PDF seems fine. You can skip sanitization.")
            # Note: The user pipeline expects a file. 
            # If we don't sanitize, we don't produce the _sanitized.pdf.
            # Users should use the original file then.
            return 0
        else:
            print("\n[Issue] Encoding issues detected. Proceeding with sanitization...")

    try:
        result_path = sanitize_pdf(args.pdf)
        print(f"Successfully created: {result_path}")
        return 0
    except Exception as e:
        print(f"Errors occurred during sanitization: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

