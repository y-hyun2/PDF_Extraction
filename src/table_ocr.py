"""표 영역 텍스트 추출 유틸.

RapidOCR를 이용해 이미지에서 직접 OCR 하거나, PyMuPDF로 PDF 텍스트를 그대로
가져오는 두 가지 방식을 지원한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import fitz
from rapidocr import RapidOCR

# Add src to path to allow importing sibling modules if run from root
import sys
sys.path.append(str(Path(__file__).parent))


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRUCTURED_DIR = REPO_ROOT / "data" / "pages_structured"
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "input"
PDF_WORD_TOLERANCE = 2.0


def infer_default_pdf() -> Path:
    candidates = sorted(DEFAULT_INPUT_DIR.glob("*.pdf"))
    if not candidates:
        raise FileNotFoundError(
            f"{DEFAULT_INPUT_DIR}에서 PDF를 찾지 못했습니다. --pdf 옵션으로 직접 지정하세요."
        )
    return candidates[0]


def find_available_pages(structured_dir: Path) -> tuple[list[int], Path]:
    pages: list[int] = []
    actual_root = structured_dir
    
    # Check if root has page folders
    root_has_pages = any(c.is_dir() and c.name.startswith("page_") for c in structured_dir.iterdir())
    
    if root_has_pages:
        scan_dir = structured_dir
    else:
        # Try finding a subdirectory that looks like a report
        # Prefer "sanitized" folder if multiple exist
        candidates = [d for d in structured_dir.iterdir() if d.is_dir()]
        sanitized = [d for d in candidates if "_sanitized" in d.name]
        others = [d for d in candidates if "_sanitized" not in d.name]
        
        target_sub = None
        
        # Check sanitized first
        for sub in sanitized:
            if any(c.is_dir() and c.name.startswith("page_") for c in sub.iterdir()):
                target_sub = sub
                break
        
        # Then others
        if not target_sub:
            for sub in others:
                 if any(c.is_dir() and c.name.startswith("page_") for c in sub.iterdir()):
                    target_sub = sub
                    break
        
        if target_sub:
            print(f"INFO: Automatically detected report directory: {target_sub.name}")
            scan_dir = target_sub
            actual_root = target_sub
        else:
            scan_dir = structured_dir # Fallback to empty root

    # Collect pages
    for child in scan_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("page_"):
            continue
        try:
            page_no = int(child.name.split("_")[-1])
        except ValueError:
            continue
        if (child / "page.json").exists():
            pages.append(page_no)

    return sorted(pages), actual_root


def parse_pages_arg(pages_arg: str, available: Iterable[int]) -> list[int]:
    avail_set = set(available)
    selected: set[int] = set()
    for raw in pages_arg.split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if start > end:
                start, end = end, start
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    final = sorted(page for page in selected if page in avail_set)
    if not final:
        raise ValueError("요청한 페이지가 pages_structured에 없습니다.")
    return final


def serialize_ocr_output(out) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    boxes = out.boxes if out.boxes is not None else []
    txts = out.txts if out.txts is not None else []
    for text, box in zip(txts, boxes):
        entries.append(
            {
                "text": text,
                "box": [[float(pt[0]), float(pt[1])] for pt in box],
            }
        )
    return entries


def process_table_image(
    ocr: RapidOCR,
    image_path: Path,
    output_json: Path,
) -> list[dict[str, object]]:
    out = ocr(str(image_path))
    entries = serialize_ocr_output(out)
    output_json.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    return entries


def update_page_metadata(page_json_path: Path, table_id: str, ocr_rel_path: str, preview: str) -> None:
    data = json.loads(page_json_path.read_text(encoding="utf-8"))
    updated = False
    for table in data.get("tables", []):
        if table.get("id") == table_id:
            table["ocr_path"] = ocr_rel_path
            table["ocr_preview"] = preview
            updated = True
            break
    if updated:
        page_json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def bbox_to_pdf_rect(bbox: dict[str, float], page_height: float) -> tuple[float, float, float, float]:
    left = float(bbox.get("left", 0.0))
    right = float(bbox.get("right", 0.0))
    top = float(bbox.get("top", 0.0))
    bottom = float(bbox.get("bottom", 0.0))
    y0 = page_height - top
    y1 = page_height - bottom
    if y0 > y1:
        y0, y1 = y1, y0
    return left, y0, right, y1


def rect_contains_center(rect: tuple[float, float, float, float],
                         x0: float,
                         y0: float,
                         x1: float,
                         y1: float,
                         tolerance: float) -> bool:
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    return (
        rect[0] - tolerance <= cx <= rect[2] + tolerance
        and rect[1] - tolerance <= cy <= rect[3] + tolerance
    )


def extract_table_text_with_pymupdf(
    page,
    bbox: dict[str, float] | None,
    page_height: float,
) -> list[dict[str, object]]:
    if not bbox:
        return []
    rect = bbox_to_pdf_rect(bbox, page_height)
    words = page.get_text("words")
    entries: list[dict[str, object]] = []
    for word in words:
        x0, y0, x1, y1, text, *_ = word
        text = (text or "").strip()
        if not text:
            continue
        if not rect_contains_center(rect, x0, y0, x1, y1, PDF_WORD_TOLERANCE):
            continue
        entries.append(
            {
                "text": text,
                "box": [
                    [float(x0), float(y0)],
                    [float(x1), float(y0)],
                    [float(x1), float(y1)],
                    [float(x0), float(y1)],
                ],
            }
        )
    return entries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="pages_structured 내 표 이미지를 RapidOCR로 텍스트화한다.",
    )
    parser.add_argument(
        "--structured-dir",
        type=Path,
        default=DEFAULT_STRUCTURED_DIR,
        help="structured_extract.py가 생성한 페이지 폴더 경로.",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="처리할 페이지 목록/범위. 생략하면 모든 페이지를 대상으로 함.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 존재하는 ocr.json이 있어도 다시 생성.",
    )
    parser.add_argument(
        "--backend",
        choices=("pymupdf", "rapidocr"),
        default="pymupdf",
        help="텍스트 추출 방식. 기본값은 PyMuPDF로 PDF 텍스트를 그대로 사용.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="--backend pymupdf일 때 사용할 원본 PDF 경로. 생략하면 data/input의 첫 PDF를 사용.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    structured_dir = args.structured_dir.resolve()
    if not structured_dir.exists():
        parser.error(f"구조화 폴더를 찾을 수 없습니다: {structured_dir}")

    available_pages, actual_structured_dir = find_available_pages(structured_dir)
    if not available_pages:
        parser.error("처리할 페이지가 없습니다. 먼저 structured_extract.py를 실행하세요.")

    # Update structured_dir to the detected one so downstream logic works
    structured_dir = actual_structured_dir 

    if args.pages:
        target_pages = parse_pages_arg(args.pages, available_pages)
    else:
        target_pages = available_pages

    pdf_doc = None
    if args.backend == "pymupdf":
        if args.pdf is None:
            pdf_path = infer_default_pdf()
            print(f"기본 PDF 사용: {pdf_path}")
        else:
            pdf_path = args.pdf.expanduser().resolve()
            if not pdf_path.exists():
                parser.error(f"PDF를 찾을 수 없습니다: {pdf_path}")
        pdf_doc = fitz.open(pdf_path)
    else:
        ocr = RapidOCR()

    try:
        for page_no in target_pages:
            page_dir = structured_dir / f"page_{page_no:04d}"
            page_json_path = page_dir / "page.json"
            tables_dir = page_dir / "tables"
            if not tables_dir.exists() or not page_json_path.exists():
                continue

            page_data = json.loads(page_json_path.read_text(encoding="utf-8"))
            tables = page_data.get("tables", [])
            if not tables:
                continue

            if args.backend == "pymupdf":
                if pdf_doc is None:
                    parser.error("PyMuPDF 백엔드를 사용하려면 PDF 문서를 열 수 있어야 합니다.")
                page_index = page_no - 1
                if page_index < 0 or page_index >= len(pdf_doc):
                    print(f"[SKIP] 페이지 {page_no} (PDF 범위 밖)")
                    continue
                pdf_page = pdf_doc[page_index]
                page_height = float(
                    (page_data.get("page_dimensions") or {}).get("height")
                    or pdf_page.rect.height
                )
            else:
                pdf_page = None
                page_height = 0.0

            for table in tables:
                table_id = table.get("id")
                image_rel = table.get("image_path")
                bbox = table.get("bbox")
                if not table_id or not image_rel or not bbox:
                    continue
                image_path = structured_dir / image_rel
                ocr_json_path = image_path.with_suffix(".ocr.json")
                if ocr_json_path.exists() and not args.overwrite:
                    print(f"[SKIP] {ocr_json_path} (이미 존재)")
                    continue

                if args.backend == "pymupdf" and pdf_page is not None:
                    entries = extract_table_text_with_pymupdf(pdf_page, bbox, page_height)
                else:
                    if not image_path.exists():
                        print(f"[SKIP] {image_path} (이미지 없음)")
                        continue
                    entries = process_table_image(ocr, image_path, ocr_json_path)

                # Save standard OCR result
                if args.backend == "pymupdf":
                    ocr_json_path.write_text(
                        json.dumps(entries, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                
                # Update metadata with OCR path
                preview = " ".join(item["text"] for item in entries[:5]) if entries else ""
                relative_ocr_path = ocr_json_path.relative_to(structured_dir)
                update_page_metadata(page_json_path, table_id, str(relative_ocr_path), preview)
                
                origin = "PDF" if args.backend == "pymupdf" else "RapidOCR"
                print(f"텍스트 추출 완료({origin}): {table_id} -> {ocr_json_path}")
    finally:
        if pdf_doc is not None:
            pdf_doc.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
