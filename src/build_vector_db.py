"""MySQL 데이터를 이용해 페이지/청크 2단계 벡터 DB를 구축하는 스크립트."""

from __future__ import annotations

import argparse
import base64
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# GPT 요약을 위해 OpenAI 클라이언트 사용
from openai import OpenAI

from load_to_db import get_connection

# ===== 설정 =====
REPO_ROOT = Path(__file__).resolve().parents[1]
STRUCTURED_ROOT = REPO_ROOT / "data" / "pages_structured"
BASE_DIR = Path("vector_db")
PAGE_COLLECTION = "esg_pages"
CHUNK_COLLECTION = "esg_chunks"
EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
BATCH_SIZE = 32
PAGE_SUMMARY_PROMPT = """
You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Give a concise summary of the image that is well optimized for retrieval.

Instructions

- Image is page of sustainability report, and this RAG system will be used to QA task for ESG analyst.

- Summary should be in Korean.

- RAG 목적에 맞게 제목, 키워드, 주요 데이터 를 추출해주고, 내용을 빠짐없이 모두 마크다운 형식으로 정리해주고, 특히 삽입된 이미지와 표를 잘 인식해서 텍스트로 변환해줘

- 페이지가 섹션을 구분하는 페이지 이거나, 제목, 목차와 같은 페이지라면, 내용을 짧게 작성하도록해.

- 페이지에 정보가 적다면, 내용이 적어도 괜찮고, 정보가 많다면, 내용이 누락되지 않게 자세하게 작성해.

- 요약 결과 포맷은 다음과 같아.


예시 포맷

---


제목: 기후변화 리스크 및 기회 의사결정 프로세스와 전사 대응, 온실가스 감축 목표

키워드: 기후변화 리스크, 기후변화 기회, 중대성 평가, ESG위원회, CEO, CLO(ESG 총괄), ESG추진 실무단, 전사 리스크 관리, 온실가스 (Scope1, Scope2, Scope3), 재생에너지 사용 비율, 대응 로드맵

주요 데이터:

- 의사결정 프로세스 표

- 이사회(ESG위원회) → 경영진(CEO, CLO) → 실무단(ESG추진)

- 온실가스 배출량 (Scope1·2·3)

- 재생에너지 사용 비율

- 연간 목표(중대성 평가 결과 반영)

상세 내용:

1. 기후변화 리스크 및 기회 의사결정 프로세스

- 중대성 평가 및 리스크 인식

- 조직 내 검토되는 기후변화 리스크를 다른 리스크와 함께 평가, 우선순위 결정

- 평가 과정: 내부 영향 분석 → 이슈 식별·분류 → 중대성 평가 → 핵심 이슈 도출

- 중대한 리스크·기회 요인 발생 시 CLO에게 즉시 보고 후 필요시 CEO, ESG위원회까지 단계별 보고

- 의사결정 구조

- 이사회(ESG위원회)

- 핵심 기후변화 리스크·기회에 대한 최종 의결

- SK텔레콤에 심각한 영향을 미치는 사안에 대한 심의

- 경영진(CEO)

- 중대한 리스크·기회 요인에 대한 의사결정

- 사업·재무에 심각한 영향을 미치는 핵심 이슈 발생 시 ESG위원회 보고

- 경영진(CLO, ESG 총괄)

- 기후변화 리스크·기회가 중대한 사안일 경우 CEO 보고

- 영향 범위 및 대응 방안 검토, 필요시 이사회(ESG위원회)와 심의

- 실무단(ESG추진)

- 리스크·기회 요소 모니터링 및 중대성 평가(내부 영향 분석) 수행

- 평가 결과를 토대로 대응 시나리오 수립, ESG위원회·경영진 보고

2. 기후변화 리스크 및 기회 전사 대응 프로세스

- 의사결정 이후 후속 조치

- ESG 추진 유관 부서가 대응 방향 설정 → 중장기 관점의 대안 마련

- 매년 자체 계획 반영 및 이행 성과 모니터링, 필요 시 공공기관 등 이해관계자와 협력

- 전사 리스크 대응 체계에 기후변화 리스크·기회를 포함하여 상시 관리

- 주요 추진 내용

- 정기·수시 모니터링: 경영·재무적 영향, 조직 운영 리스크 검토

- 중대한 이슈 발생 시 ESG위원회 보고와 함께 내부 프로세스 강화

- 중대성 평가 결과 반영 후 전사 전략 및 목표 재조정

3. 기후변화 리스크 및 기회 관련 지표와 목표

- 온실가스 배출량 (Scope1)

- 단위: tCO₂e

- 2020: 1,039,979

- 2021: 1,101,340

- 2022: 1,132,090

- 간접 배출량 (Scope2)

- 단위: tCO₂e

- 2020: 1,031,338

- 2021: 1,094,967

- 2022: 1,126,600

- 기타 간접 배출량 (Scope3)

- 단위: tCO₂e

- 2020: 6,918,286

- 2021: 6,925,159

- 2022: 7,059,192

- 재생에너지 사용 비율

- 2020: 1.0%

- 2021: 2.0%

- 2022: 2.5%

- 목표: 5.0%

- 참고 사항

- Scope1: 직접 배출(연료 사용 등)

- Scope2: 전기·열 사용 등 간접 배출

- Scope3: 밸류체인 전반(구매, 물류 등)에서 발생하는 기타 간접 배출

- 재생에너지 사용 비율: 연간 전력 사용량 대비 재생에너지 비중

4. 요약

- 이 문서는 SK텔레콤이 기후변화 리스크 및 기회를 중대 리스크로 인식하고, 이사회(ESG위원회)·경영진·실무단 간 협업을 통해 식별·우선순위화·대응 방안을 결정하는 과정을 상세히 다룸

- 온실가스 배출(Scope1·2·3) 추이와 재생에너지 사용 목표 등 주요 지표를 공개하고, 매년 중대성 평가와 ESG 전략 수정·이행 모니터링을 반복하여 리스크를 체계적으로 관리하는 내용을 제시함
"""



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


def build_page_context(page_row: Dict[str, Any], figure_texts: List[str], table_titles: List[str]) -> str:
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


def summarize_page_with_gpt(client: OpenAI, page_no: int, context: str, image_path: Path | None) -> str:
    if client is None:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    user_content = (
        f"{PAGE_SUMMARY_PROMPT}\n\n"
        f"[페이지 번호]\n{page_no}\n\n"
        f"[페이지 본문 및 부가 정보]\n{context}"
    )
    content_payload: list[dict] = [{"type": "input_text", "text": user_content}]
    if image_path and image_path.exists():
        image_b64 = encode_image_base64(image_path)
        content_payload.append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image_b64}",
            }
        )
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": content_payload}],
        temperature=0.3,
        max_output_tokens=800,
    )
    for item in resp.output or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                return text
    raise RuntimeError(f"GPT 응답이 비었습니다. 페이지 {page_no}")


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


def encode_image_base64(image_path: Path) -> str:
    data = image_path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


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

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 필요합니다 (페이지 GPT 요약 단계).")
    gpt_client = OpenAI(api_key=api_key)

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
        for fig in figures_by_page.get(page["page_id"], []):
            desc = (fig.get("description") or "").strip()
            if desc:
                fig_texts.append(f"- {desc}")
            fig_ids.append(fig["figure_id"])

        table_titles = []
        tbl_ids = []
        for tbl in tables_by_page.get(page["page_id"], []):
            title = tbl.get("title") or f"표 {tbl['table_id']}"
            table_titles.append(title)
            tbl_ids.append(tbl["table_id"])

        doc_folder = Path(page["filename"]).stem
        image_rel = page.get("image_path")
        image_abs = STRUCTURED_ROOT / doc_folder / image_rel if image_rel else None

        context_text = build_page_context(page, fig_texts, table_titles)
        summary_text = summarize_page_with_gpt(gpt_client, page["page_no"], context_text, image_abs)
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
def summarize_page_with_gpt(client: OpenAI, page_no: int, context: str, image_path: Path | None) -> str:
    """GPT-4o에게 페이지 요약을 요청한다. 이미지도 함께 첨부."""
    if client is None:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    user_content = (
        f"{PAGE_SUMMARY_PROMPT}\n\n"
        f"[페이지 번호]\n{page_no}\n\n"
        f"[페이지 본문 및 부가 정보]\n{context}"
    )
    content_payload: list[dict] = [{"type": "text", "text": user_content}]
    if image_path and image_path.exists():
        image_b64 = encode_image_base64(image_path)
        content_payload.append(
            {
                "type": "input_image",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        )
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "user", "content": content_payload}],
        temperature=0.3,
        max_output_tokens=800,
    )
    for item in resp.output or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                return text
    raise RuntimeError(f"GPT 응답이 비었습니다. 페이지 {page_no}")
