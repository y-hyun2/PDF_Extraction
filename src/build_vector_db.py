"""MySQL 데이터를 이용해 페이지/청크 2단계 벡터 DB를 구축하는 스크립트."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from load_to_db import get_connection

# ===== 설정 =====
BASE_DIR = Path("vector_db")
PAGE_COLLECTION = "esg_pages"
CHUNK_COLLECTION = "esg_chunks"
EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
BATCH_SIZE = 32
FIGURE_SUMMARY_LIMIT = 1500  # 페이지 대표 텍스트에서 그림 설명 총 길이 제한


def get_or_create_collections(client: chromadb.PersistentClient, reset: bool):
    if reset:
        for name in (PAGE_COLLECTION, CHUNK_COLLECTION):
            try:
                client.delete_collection(name)
            except Exception:
                pass
    page_col = client.get_or_create_collection(PAGE_COLLECTION, metadata={"hnsw:space": "cosine"})
    chunk_col = client.get_or_create_collection(CHUNK_COLLECTION, metadata={"hnsw:space": "cosine"})
    return page_col, chunk_col


def fetch_pages(conn) -> List[Dict[str, Any]]:
    sql = """
        SELECT d.id AS doc_id,
               d.filename,
               d.company_name,
               d.report_year,
               p.id AS page_id,
               p.page_no,
               p.full_markdown,
               p.image_path
        FROM pages p
        JOIN documents d ON p.doc_id = d.id
        WHERE p.full_markdown IS NOT NULL AND p.full_markdown != ''
        ORDER BY d.id, p.page_no
    """
    with conn.cursor() as cursor:
        cursor.execute(sql)
        return cursor.fetchall()


def fetch_figures(conn) -> List[Dict[str, Any]]:
    sql = """
        SELECT f.id AS figure_id,
               f.doc_id,
               f.page_id,
               p.page_no,
               f.caption,
               f.description,
               f.image_path,
               d.company_name,
               d.report_year,
               d.filename
        FROM doc_figures f
        JOIN pages p ON f.page_id = p.id
        JOIN documents d ON f.doc_id = d.id
        WHERE f.description IS NOT NULL AND CHAR_LENGTH(f.description) > 0
    """
    with conn.cursor() as cursor:
        cursor.execute(sql)
        return cursor.fetchall()


def fetch_tables(conn) -> List[Dict[str, Any]]:
    sql = """
        SELECT t.id AS table_id,
               t.doc_id,
               t.page_id,
               t.page_no,
               t.title,
               t.image_path,
               t.diff_data,
               d.company_name,
               d.report_year,
               d.filename
        FROM doc_tables t
        JOIN documents d ON t.doc_id = d.id
        ORDER BY t.doc_id, t.page_no, t.id
    """
    with conn.cursor() as cursor:
        cursor.execute(sql)
        return cursor.fetchall()


def fetch_table_cells(conn, table_ids: Iterable[int]) -> Dict[int, List[Dict[str, Any]]]:
    table_ids = list(table_ids)
    if not table_ids:
        return {}
    placeholders = ",".join(["%s"] * len(table_ids))
    sql = f"""
        SELECT table_id, row_idx, col_idx, content, is_header
        FROM table_cells
        WHERE table_id IN ({placeholders})
        ORDER BY table_id, row_idx, col_idx
    """
    with conn.cursor() as cursor:
        cursor.execute(sql, table_ids)
        rows = cursor.fetchall()
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["table_id"]].append(row)
    return grouped


def build_table_text(table_meta: Dict[str, Any], cells: List[Dict[str, Any]]) -> str:
    rows: Dict[int, List[str]] = defaultdict(list)
    for cell in cells:
        text = (cell.get("content") or "").strip()
        rows[cell["row_idx"]].append((cell["col_idx"], text))

    ordered_lines: List[str] = []
    for row_idx in sorted(rows.keys()):
        cols = [text for _, text in sorted(rows[row_idx], key=lambda pair: pair[0])]
        ordered_lines.append(" | ".join(cols).strip())

    title = table_meta.get("title") or "(제목 없음)"
    lines = [f"표 제목: {title}", "표 내용:"] + ordered_lines
    if table_meta.get("diff_data"):
        diff_str = table_meta["diff_data"]
        if isinstance(diff_str, dict):
            diff_repr = json.dumps(diff_str, ensure_ascii=False)
        else:
            diff_repr = str(diff_str)
        lines.append(f"검증 정보: {diff_repr}")
    return "\n".join(lines).strip()


def build_page_summary(page_row: Dict[str, Any], figure_texts: List[str], table_titles: List[str]) -> str:
    base = [
        f"페이지 {page_row['page_no']} 본문:",
        (page_row.get("full_markdown") or "").strip(),
    ]
    if table_titles:
        base.append("[이 페이지의 표]")
        base.extend(f"- {title}" for title in table_titles)
    if figure_texts:
        base.append("[이 페이지의 그림 요약]")
        base.extend(figure_texts)
    return "\n".join(line for line in base if line).strip()


def collect_page_metadata(page_row: Dict[str, Any], table_ids: List[int], figure_ids: List[int]) -> Dict[str, Any]:
    return {
        "doc_id": page_row["doc_id"],
        "page_id": page_row["page_id"],
        "page_no": page_row["page_no"],
        "company_name": page_row.get("company_name") or "Unknown",
        "report_year": page_row.get("report_year") or 0,
        "filename": page_row["filename"],
        "page_image_path": page_row.get("image_path") or "",
        "table_ids": json.dumps([str(tid) for tid in table_ids]),
        "figure_ids": json.dumps([str(fid) for fid in figure_ids]),
        "created_at": datetime.now().isoformat(),
    }


def chunk_text(text: str, splitter: RecursiveCharacterTextSplitter) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return splitter.split_text(text)


def embed_and_upsert(collection, model, ids, documents, metadatas):
    if not ids:
        return
    for start in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[start:start + BATCH_SIZE]
        batch_docs = documents[start:start + BATCH_SIZE]
        batch_metas = metadatas[start:start + BATCH_SIZE]
        embeddings = model.encode(batch_docs).tolist()
        collection.upsert(ids=batch_ids, documents=batch_docs, embeddings=embeddings, metadatas=batch_metas)


def build_vector_db(reset: bool = False) -> None:
    print(f"🚀 2단계 벡터 DB 구축 시작 (모델: {EMBEDDING_MODEL})")
    client = chromadb.PersistentClient(path=str(BASE_DIR.resolve()))
    page_collection, chunk_collection = get_or_create_collections(client, reset)

    print("📦 임베딩 모델 로딩 중...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "]
    )

    conn = get_connection()
    try:
        pages = fetch_pages(conn)
        figures = fetch_figures(conn)
        tables = fetch_tables(conn)
    finally:
        conn.close()

    if not pages:
        print("MySQL에서 페이지 데이터를 찾을 수 없습니다. load_to_db.py 실행 여부를 확인하세요.")
        return

    print(f"📄 페이지 {len(pages)}건 / 그림 {len(figures)}건 / 표 {len(tables)}건 로드 완료")

    figures_by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for fig in figures:
        figures_by_page[fig["page_id"]].append(fig)

    tables_by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for tbl in tables:
        tables_by_page[tbl["page_id"]].append(tbl)

    # 페이지 대표 텍스트 생성
    page_ids: List[str] = []
    page_docs: List[str] = []
    page_metas: List[Dict[str, Any]] = []

    for page in pages:
        fig_texts: List[str] = []
        fig_ids: List[int] = []
        remaining = FIGURE_SUMMARY_LIMIT
        for fig in figures_by_page.get(page["page_id"], []):
            desc = (fig.get("description") or "").strip()
            if desc and remaining > 0:
                snippet = desc if len(desc) <= remaining else desc[:remaining]
                fig_texts.append(f"- {snippet}")
                remaining -= len(snippet)
            fig_ids.append(fig["figure_id"])

        table_titles = []
        tbl_ids = []
        for tbl in tables_by_page.get(page["page_id"], []):
            title = tbl.get("title") or f"표 {tbl['table_id']}"
            table_titles.append(title)
            tbl_ids.append(tbl["table_id"])

        summary_text = build_page_summary(page, fig_texts, table_titles)
        page_ids.append(f"page_repr_{page['page_id']}")
        page_docs.append(summary_text)
        page_metas.append(collect_page_metadata(page, tbl_ids, fig_ids))

    print(f"🧾 페이지 대표 텍스트 {len(page_ids)}건 임베딩")
    embed_and_upsert(page_collection, model, page_ids, page_docs, page_metas)

    # 정밀 청크 처리
    chunk_ids: List[str] = []
    chunk_docs: List[str] = []
    chunk_metas: List[Dict[str, Any]] = []

    for page in pages:
        # 페이지 본문 청크
        chunks = chunk_text(page["full_markdown"], splitter)
        for idx, chunk in enumerate(chunks):
            chunk_ids.append(f"page_{page['page_id']}_chunk_{idx}")
            chunk_docs.append(chunk)
            chunk_metas.append({
                "source_type": "page_text",
                "doc_id": page["doc_id"],
                "page_id": page["page_id"],
                "page_no": page["page_no"],
                "chunk_index": idx,
                "company_name": page.get("company_name") or "Unknown",
                "report_year": page.get("report_year") or 0,
                "filename": page["filename"],
                "created_at": datetime.now().isoformat(),
            })

        # 페이지 내 테이블 텍스트
        page_tables = tables_by_page.get(page["page_id"], [])
        table_cells_map = fetch_table_cells(get_connection(), [tbl["table_id"] for tbl in page_tables])
        for tbl in page_tables:
            cells = table_cells_map.get(tbl["table_id"], [])
            table_text = build_table_text(tbl, cells)
            chunk_ids.append(f"table_{tbl['table_id']}")
            chunk_docs.append(table_text)
            chunk_metas.append({
                "source_type": "table",
                "doc_id": tbl["doc_id"],
                "page_id": tbl["page_id"],
                "page_no": tbl["page_no"],
                "table_id": tbl["table_id"],
                "table_title": tbl.get("title") or "",
                "company_name": tbl.get("company_name") or "Unknown",
                "report_year": tbl.get("report_year") or 0,
                "filename": tbl["filename"],
                "image_path": tbl.get("image_path") or "",
                "diff_present": bool(tbl.get("diff_data")),
                "created_at": datetime.now().isoformat(),
            })

        # 페이지 내 그림 설명
        for fig in figures_by_page.get(page["page_id"], []):
            desc = (fig.get("description") or "").strip()
            if not desc:
                continue
            figure_text = f"캡션: {fig.get('caption') or ''}\n\n{desc}"
            chunk_ids.append(f"figure_{fig['figure_id']}")
            chunk_docs.append(figure_text)
            chunk_metas.append({
                "source_type": "figure",
                "doc_id": fig["doc_id"],
                "page_id": fig["page_id"],
                "page_no": fig["page_no"],
                "figure_id": fig["figure_id"],
                "company_name": fig.get("company_name") or "Unknown",
                "report_year": fig.get("report_year") or 0,
                "filename": fig["filename"],
                "image_path": fig.get("image_path") or "",
                "created_at": datetime.now().isoformat(),
            })

    print(f"🔍 정밀 청크 {len(chunk_ids)}건 임베딩")
    embed_and_upsert(chunk_collection, model, chunk_ids, chunk_docs, chunk_metas)

    print(f"✅ 페이지 컬렉션 벡터 수: {page_collection.count()}")
    print(f"✅ 청크 컬렉션 벡터 수: {chunk_collection.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="기존 벡터 DB를 초기화하고 재구축")
    args = parser.parse_args()

    build_vector_db(reset=args.reset)
