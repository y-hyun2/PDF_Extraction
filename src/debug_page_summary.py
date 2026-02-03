"""esg_pages 컬렉션에서 doc_id/page_no로 GPT 요약 확인."""

from __future__ import annotations

import argparse
from typing import Any

import chromadb

VECTOR_DB_DIR = "vector_db"
COLLECTION_NAME = "esg_pages"


def main() -> None:
    parser = argparse.ArgumentParser(description="doc_id/page_no로 페이지 요약을 출력")
    parser.add_argument("doc_id", type=int, help="documents.id")
    parser.add_argument("page_no", type=int, help="페이지 번호")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    filters = {"$and": [{"doc_id": args.doc_id}, {"page_no": args.page_no}]}
    data: dict[str, Any] = collection.get(where=filters)

    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    ids = data.get("ids") or []

    if not docs:
        print("해당 조건에 맞는 페이지를 찾지 못했습니다.")
        return

    for idx, (doc, meta, cid) in enumerate(zip(docs, metas, ids), start=1):
        print("=" * 80)
        print(f"[결과 {idx}] id={cid}")
        print("--- 메타데이터 ---")
        print(meta)
        print("--- 요약 텍스트 ---")
        print(doc)


if __name__ == "__main__":
    main()
