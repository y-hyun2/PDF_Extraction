"""Docling 기반으로 페이지를 구조화해 저장하는 스크립트.

페이지별 Markdown, 표 JSON/이미지, 그림 이미지를 한 폴더에 모으고
후속 GPT 해석 단계에서 사용할 수 있도록 메타데이터(JSON)를 작성한다.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pypdfium2 as pdfium
from docling.datamodel.base_models import ConversionStatus
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "input"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "pages_structured"
GPT_API_KEY_PLACEHOLDER = "PASTE_YOUR_GPT_API_KEY"
MIN_FIGURE_AREA_RATIO = 0.01
FIGURE_HEADER_RATIO = 0.12

load_dotenv()


@dataclass
class TextBlock:
    text: str
    bbox: dict[str, float]


def infer_default_pdf() -> Path:
    candidates = sorted(DEFAULT_INPUT_DIR.glob("*.pdf"))
    if not candidates:
        raise FileNotFoundError(
            f"{DEFAULT_INPUT_DIR}에서 PDF를 찾지 못했습니다. --pdf 옵션으로 직접 지정하세요."
        )
    return candidates[0]


def parse_page_selection(selection: str, total_pages: int) -> list[int]:
    pages: set[int] = set()
    for raw in selection.split(","):
        part = raw.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if start > end:
                start, end = end, start
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))

    normalized = sorted(p for p in pages if 1 <= p <= total_pages)
    if not normalized:
        raise ValueError("요청한 페이지가 문서 범위를 벗어났습니다.")
    return normalized


def pages_from_count(total_pages: int, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("--count 값은 양수여야 합니다.")
    upper = min(total_pages, count)
    return list(range(1, upper + 1))


def chunk_consecutive(pages: Iterable[int]) -> list[tuple[int, int]]:
    sorted_pages = sorted(pages)
    if not sorted_pages:
        return []

    groups: list[tuple[int, int]] = []
    start = prev = sorted_pages[0]
    for page in sorted_pages[1:]:
        if page == prev + 1:
            prev = page
            continue
        groups.append((start, prev))
        start = prev = page
    groups.append((start, prev))
    return groups


def bbox_to_dict(bbox) -> dict[str, float]:
    return {
        "left": float(bbox.l),
        "right": float(bbox.r),
        "top": float(bbox.t),
        "bottom": float(bbox.b),
    }


def collect_text_blocks(doc, page_no: int) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for text in doc.texts:
        for prov in text.prov:
            if prov.page_no != page_no or prov.bbox is None:
                continue
            content = (text.text or "").strip()
            if not content:
                continue
            blocks.append(TextBlock(text=content, bbox=bbox_to_dict(prov.bbox)))
            break
    return blocks


def horizontal_overlap_ratio(a: dict[str, float], b: dict[str, float]) -> float:
    left = max(a["left"], b["left"])
    right = min(a["right"], b["right"])
    if right <= left:
        return 0.0
    width_a = max(1e-3, a["right"] - a["left"])
    width_b = max(1e-3, b["right"] - b["left"])
    overlap = right - left
    return overlap / min(width_a, width_b)


def detect_table_title(
    table_bbox: dict[str, float],
    text_blocks: list[TextBlock],
    vertical_threshold: float = 120.0,
    overlap_threshold: float = 0.5,
    max_chars: int = 60,
) -> str | None:
    best_text: str | None = None
    best_score = float("inf")
    for block in text_blocks:
        overlap = horizontal_overlap_ratio(block.bbox, table_bbox)
        if overlap < overlap_threshold:
            continue
        text_len = len(block.text)
        if text_len == 0 or text_len > max_chars:
            continue
        vertical_gap = block.bbox["bottom"] - table_bbox["top"]
        if vertical_gap < 0 or vertical_gap > vertical_threshold:
            continue
        score = vertical_gap - overlap * 10
        if score < best_score:
            best_score = score
            best_text = block.text
    return best_text


def table_cells_to_json(table, doc) -> list[list[dict[str, object]]]:
    rows: list[list[dict[str, object]]] = []
    for row_idx, row in enumerate(table.data.grid):
        row_cells: list[dict[str, object]] = []
        for col_idx, cell in enumerate(row):
            row_cells.append(
                {
                    "row": row_idx,
                    "col": col_idx,
                    "text": cell.text if cell.text is not None else cell._get_text(doc=doc),
                    "row_span": cell.row_span,
                    "col_span": cell.col_span,
                    "row_header": cell.row_header,
                    "column_header": cell.column_header,
                }
            )
        rows.append(row_cells)
    return rows


def render_page_image(pdf_doc: pdfium.PdfDocument, page_no: int, scale: float):
    page = pdf_doc[page_no - 1]
    try:
        pil_image = page.render(scale=scale).to_pil()
    finally:
        page.close()
    return pil_image


def bbox_to_pixels(
    bbox: dict[str, float],
    page_width: float,
    page_height: float,
    image_width: int,
    image_height: int,
):
    scale_x = image_width / page_width
    scale_y = image_height / page_height

    left = max(0, int(round(bbox["left"] * scale_x)))
    right = min(image_width, int(round(bbox["right"] * scale_x)))
    top = max(0, int(round((page_height - bbox["top"]) * scale_y)))
    bottom = min(image_height, int(round((page_height - bbox["bottom"]) * scale_y)))

    if bottom <= top or right <= left:
        return None
    return left, top, right, bottom


def crop_region(image, crop_box, output_path: Path) -> Path | None:
    if crop_box is None:
        return None
    region = image.crop(crop_box)
    region.save(output_path)
    return output_path


def summarize_with_gpt(page_payload: dict, api_key: str | None, model_name: str) -> str:
    """GPT 호출 자리. API 키를 설정한 뒤 원하는 로직으로 교체하세요."""

    if not api_key or api_key == GPT_API_KEY_PLACEHOLDER:
        return "GPT 요약 미실행: OPENAI_API_KEY 환경변수를 설정하고 summarize_with_gpt를 구현하세요."

    client = OpenAI(api_key=api_key)

    page_no = page_payload["page_number"]
    markdown = page_payload["markdown"]
    trimmed_markdown = markdown[:6000]

    table_notes = []
    for table in page_payload.get("tables", []):
        title = table.get("title") or table.get("id")
        detail = f"- {title}: 구조화 데이터={table.get('json_path')}"
        if table.get("ocr_path"):
            detail += f", OCR={table.get('ocr_path')}"
        table_notes.append(detail)
    tables_text = "\n".join(table_notes) if table_notes else "(이 페이지에는 표가 없습니다.)"

    visual_hint = "이 페이지는 이미지 비중이 높으니 주요 메시지를 서술하고, 수치 비교는 table_json을 참고하세요."
    prompt = (
        f"다음은 ESG 보고서 {page_no}페이지의 Markdown 본문입니다. 요약과 해석을 제공하되, "
        f"구체적인 수치는 아래 table_json 파일에서 검증 후 언급하세요.\n\n"
        f"[본문]\n{trimmed_markdown}\n\n[표 메타]\n{tables_text}\n\n{visual_hint}"
    )

    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "당신은 ESG 보고서를 해석하는 분석가입니다. 표 JSON의 숫자를 신뢰하고, Markdown 텍스트의 맥락을 설명하세요.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                },
            ],
            temperature=0.2,
            max_output_tokens=600,
        )
    except Exception as exc:  # pragma: no cover - runtime safeguard
        return f"GPT 호출 실패: {exc}"

    output_fragments: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text_val = getattr(content, "text", None)
            if text_val:
                output_fragments.append(text_val)
    return "".join(output_fragments).strip() or "(GPT 응답이 비어 있습니다.)"


def process_page(
    doc,
    pdf_doc: pdfium.PdfDocument,
    page_no: int,
    output_root: Path,
    render_scale: float,
    gpt_api_key: str | None,
    enable_gpt: bool,
    gpt_model: str,
    visual_threshold: float,
):
    page_dir = output_root / f"page_{page_no:04d}"
    tables_dir = page_dir / "tables"
    figures_dir = page_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    markdown = doc.export_to_markdown(
        page_no=page_no,
        image_placeholder="[IMAGE]",
        include_annotations=False,
        page_break_placeholder=None,
    ).strip()
    page_md_path = page_dir / "page.md"
    page_md_path.write_text(markdown, encoding="utf-8")

    page_image = render_page_image(pdf_doc, page_no, render_scale)
    page_image_path = page_dir / "page.png"
    page_image.save(page_image_path)

    page_size = doc.pages[page_no].size
    page_width = float(page_size.width)
    page_height = float(page_size.height)

    text_blocks = collect_text_blocks(doc, page_no)

    tables_meta: list[dict[str, object]] = []
    table_area = 0.0
    page_tables = [t for t in doc.tables if any(prov.page_no == page_no for prov in t.prov)]
    for idx, table in enumerate(page_tables, start=1):
        bbox = bbox_to_dict(table.prov[0].bbox)
        title = detect_table_title(bbox, text_blocks)
        table_id = f"table_{idx:03d}"
        markdown_path = tables_dir / f"{table_id}.md"
        markdown_path.write_text(table.export_to_markdown(doc=doc), encoding="utf-8")

        table_json = {
            "id": table_id,
            "title": title,
            "cells": table_cells_to_json(table, doc),
        }
        json_path = tables_dir / f"{table_id}.json"
        json_path.write_text(json.dumps(table_json, ensure_ascii=False, indent=2), encoding="utf-8")

        crop_box = bbox_to_pixels(bbox, page_width, page_height, page_image.width, page_image.height)
        image_path = tables_dir / f"{table_id}.png"
        saved_image = crop_region(page_image, crop_box, image_path)

        tables_meta.append(
            {
                "id": table_id,
                "title": title,
                "markdown_path": str(markdown_path.relative_to(output_root)),
                "json_path": str(json_path.relative_to(output_root)),
                "image_path": str(saved_image.relative_to(output_root)) if saved_image else "",
                "bbox": bbox,
            }
        )
        table_area += max(0.0, (bbox["right"] - bbox["left"])) * max(0.0, (bbox["top"] - bbox["bottom"]))

    figures_meta: list[dict[str, object]] = []
    figure_area = 0.0
    header_cutoff = page_height * (1 - FIGURE_HEADER_RATIO) if page_height else None
    page_pictures = [p for p in doc.pictures if any(prov.page_no == page_no for prov in p.prov)]
    for idx, picture in enumerate(page_pictures, start=1):
        bbox = bbox_to_dict(picture.prov[0].bbox)
        figure_id = f"figure_{idx:03d}"
        width = max(0.0, bbox["right"] - bbox["left"])
        height = max(0.0, bbox["top"] - bbox["bottom"])
        page_area = max(1e-3, page_width * page_height)
        area_ratio = (width * height) / page_area
        if area_ratio < MIN_FIGURE_AREA_RATIO:
            print(
                f"[SKIP ICON] page {page_no} {figure_id} (area ratio={area_ratio:.4f})"
            )
            continue
        if header_cutoff and bbox["bottom"] >= header_cutoff:
            print(f"[SKIP HEADER] page {page_no} {figure_id} (header zone)")
            continue
        crop_box = bbox_to_pixels(bbox, page_width, page_height, page_image.width, page_image.height)
        image_path = figures_dir / f"{figure_id}.png"
        saved_image = crop_region(page_image, crop_box, image_path)

        caption_texts: list[str] = []
        for ref in picture.captions:
            node = ref.resolve(doc)
            text = getattr(node, "text", "")
            if text:
                caption_texts.append(text.strip())

        figures_meta.append(
            {
                "id": figure_id,
                "caption": " ".join(caption_texts) if caption_texts else None,
                "image_path": str(saved_image.relative_to(output_root)) if saved_image else "",
                "bbox": bbox,
            }
        )
        figure_area += width * height

    visual_density = (table_area + figure_area) / (page_width * page_height)
    needs_visual_review = visual_density >= visual_threshold or bool(page_pictures)

    summary_path = None
    if enable_gpt:
        summary_text = summarize_with_gpt(
            {
                "page_number": page_no,
                "markdown": markdown,
                "tables": tables_meta,
            },
            gpt_api_key,
            gpt_model,
        )
        summary_path = page_dir / "summary.md"
        summary_path.write_text(summary_text, encoding="utf-8")

    page_payload = {
        "page_number": page_no,
        "markdown": markdown,
        "markdown_path": str(page_md_path.relative_to(output_root)),
        "page_image_path": str(page_image_path.relative_to(output_root)),
        "page_dimensions": {"width": page_width, "height": page_height},
        "tables": tables_meta,
        "figures": figures_meta,
        "needs_visual_review": needs_visual_review,
        "visual_density": visual_density,
        "summary_path": str(summary_path.relative_to(output_root)) if summary_path else None,
    }

    page_json_path = page_dir / "page.json"
    page_json_path.write_text(json.dumps(page_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"페이지 {page_no} 처리 완료 -> {page_json_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Docling 결과를 페이지별 Markdown/표/이미지로 구조화한다.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="대상 PDF 경로. 생략하면 data/input의 첫 번째 파일을 사용.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="페이지별 산출물을 저장할 디렉터리.",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="1-based 페이지 목록/범위(예: 25,27-29).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="--pages를 생략하면 앞쪽에서 몇 페이지를 처리할지 지정.",
    )
    parser.add_argument(
        "--render-scale",
        type=float,
        default=2.0,
        help="PDF 이미지를 렌더링할 배율(기본 2.0=약 144DPI).",
    )
    parser.add_argument(
        "--visual-threshold",
        type=float,
        default=0.35,
        help="표+그림 면적 비율이 이 값을 넘으면 이미지 해석이 필요하다고 표시.",
    )
    parser.add_argument(
        "--gpt-summary",
        action="store_true",
        help="페이지 단위 GPT 요약을 생성할 준비를 한다(실제 호출은 summarize_with_gpt를 구현해야 함).",
    )
    parser.add_argument(
        "--gpt-api-key",
        type=str,
        default=None,
        help="GPT 호출에 사용할 API 키. 생략하면 OPENAI_API_KEY 환경변수를 사용.",
    )
    parser.add_argument(
        "--gpt-model",
        type=str,
        default="gpt-4o-mini",
        help="GPT 호출에 사용할 모델 ID.",
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default=None,
        help="pages_structured 하위에 생성할 보고서 폴더 이름. 생략하면 PDF 파일명을 기반으로 자동 생성.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.pdf is None:
        pdf_path = infer_default_pdf()
        print(f"기본 PDF 사용: {pdf_path}")
    else:
        pdf_path = args.pdf.expanduser().resolve()
        if not pdf_path.exists():
            parser.error(f"PDF를 찾을 수 없습니다: {pdf_path}")

    # Check for sanitized version automatically
    # Pattern 1: report.sanitized.pdf (Original tool default)
    # Pattern 2: report_sanitized.pdf (User modified convention)
    if "_sanitized" not in pdf_path.name and ".sanitized" not in pdf_path.name:
        candidates = [
            pdf_path.with_suffix(".sanitized.pdf"),
            pdf_path.with_stem(pdf_path.stem + "_sanitized").with_suffix(".pdf")
        ]
        
        for cand in candidates:
            if cand.exists():
                print(f"⚠️  [Auto-Switch] Sanitized PDF found: {cand.name}")
                print(f"    Switching input to use the sanitized version for better extraction.")
                pdf_path = cand
                break

    pdf_doc = pdfium.PdfDocument(str(pdf_path))
    total_pages = len(pdf_doc)

    if args.pages:
        pages = parse_page_selection(args.pages, total_pages)
    else:
        pages = pages_from_count(total_pages, args.count)

    target_pages = sorted(set(pages))

    def sanitize_report_name(raw: str) -> str:
        cleaned_chars = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw]
        sanitized = "".join(cleaned_chars).strip("_")
        return sanitized or "report"

    base_output = args.output_dir.resolve()
    default_base = DEFAULT_OUTPUT_DIR.resolve()
    report_name = args.report_name
    if not report_name and base_output == default_base:
        report_name = sanitize_report_name(pdf_path.stem)
    if report_name:
        output_root = base_output / report_name
    else:
        output_root = base_output
    output_root.mkdir(parents=True, exist_ok=True)

    gpt_api_key = (
        args.gpt_api_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPEN_AI_API_KEY")
        or GPT_API_KEY_PLACEHOLDER
    )

    converter = DocumentConverter()
    try:
        for start, end in chunk_consecutive(target_pages):
            result = converter.convert(pdf_path, page_range=(start, end))
            if result.status not in {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}:
                errors = ", ".join(err.error_message for err in result.errors)
                raise RuntimeError(f"Docling 변환 실패 ({start}-{end}): {result.status}. {errors}")
            if result.document is None:
                raise RuntimeError("Docling 문서가 반환되지 않았습니다.")

            for page_no in range(start, end + 1):
                if page_no not in target_pages:
                    continue
                process_page(
                    result.document,
                    pdf_doc,
                    page_no,
                    output_root,
                    args.render_scale,
                    gpt_api_key,
                    args.gpt_summary,
                    args.gpt_model,
                    args.visual_threshold,
                )
    finally:
        pdf_doc.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
