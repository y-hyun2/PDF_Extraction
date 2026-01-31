"""Chroma 벡터 DB 검색 스크립트.

Semantic(코사인 기반) 검색과 키워드(BM25) 검색을 모두 지원하며,
필요 시 두 결과를 혼합(Hybrid)할 수 있다.

Usage:
    python src/search_vector_db.py "query string" [--top-k 5] [--mode semantic|keyword|hybrid]
"""

import argparse
import sys
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

try:
    from kiwipiepy import Kiwi  # 선택적 한국어 형태소 분석기
    KIWI = Kiwi()
except Exception:  # pylint: disable=broad-except
    KIWI = None

import math
import re
from collections import Counter

# Configuration (Must match build_vector_db.py)
VECTOR_DB_DIR = "vector_db"
# 기본 구조 변경: 페이지/청크 2개 컬렉션을 모두 검사할 수 있도록 리스트로 보관
COLLECTIONS = ["esg_pages", "esg_chunks", "esg_documents"]  # 존재하는 컬렉션만 사용
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


def tokenize(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if KIWI is not None:
        return [token.form for token in KIWI.tokenize(text) if token.form.strip()]
    return re.findall(r"[0-9A-Za-z가-힣]+", text)


def bm25_scores(corpus_tokens: list[list[str]], query_tokens: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not corpus_tokens:
        return []
    N = len(corpus_tokens)
    doc_lens = [len(tokens) for tokens in corpus_tokens]
    avgdl = sum(doc_lens) / max(N, 1)
    df: Counter[str] = Counter()
    for tokens in corpus_tokens:
        unique = set(tokens)
        for term in unique:
            df[term] += 1

    scores = []
    for tokens, doc_len in zip(corpus_tokens, doc_lens):
        freq = Counter(tokens)
        score = 0.0
        for term in query_tokens:
            if term not in df:
                continue
            idf = math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
            tf = freq.get(term, 0)
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * doc_len / avgdl)
            score += idf * ((tf * (k1 + 1)) / denom)
        scores.append(score)
    return scores

def load_collections(client: chromadb.PersistentClient):
    active = []
    for name in COLLECTIONS:
        try:
            active.append(client.get_collection(name))
        except Exception:
            continue
    return active


def semantic_search(collections: list, model, query: str, top_k: int) -> list[tuple[str, str, dict, float]]:
    query_vec = model.encode([query]).tolist()
    results = []
    for collection in collections:
        resp = collection.query(query_embeddings=query_vec, n_results=top_k)
        docs = resp.get("documents") or []
        if not docs:
            continue
        for cid, doc, meta, dist in zip(resp["ids"][0], docs[0], resp["metadatas"][0], resp["distances"][0]):
            results.append((collection.name, doc, meta, dist))
    # distance는 cosine distance(0에 가까울수록 유사)
    results.sort(key=lambda item: item[3])
    return results[:top_k]


def keyword_search(collections: list, query: str, top_k: int) -> list[tuple[str, str, dict, float]]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    docs_all: list[tuple[str, str, dict, list[str]]] = []
    for collection in collections:
        data = collection.get(include=["documents", "metadatas", "ids"])
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        ids = data.get("ids") or []
        for doc_id, text, meta in zip(ids, docs, metas):
            tokens = tokenize(text)
            docs_all.append((collection.name, doc_id, text, meta, tokens))

    corpus_tokens = [item[4] for item in docs_all]
    scores = bm25_scores(corpus_tokens, query_tokens)
    ranked = sorted(zip(docs_all, scores), key=lambda x: x[1], reverse=True)
    results = []
    for (col_name, doc_id, text, meta, _), score in ranked[:top_k]:
        results.append((col_name, text, meta, score))
    return results


def format_result(rank: int, mode: str, collection_name: str, doc: str, meta: dict, score: float) -> None:
    company = meta.get("company_name") or "?"
    year = meta.get("report_year")
    page_no = meta.get("page_no")
    chunk = meta.get("chunk_index")
    preview = doc[:200].replace("\n", " ")
    print(f"[{mode.upper()} Rank {rank}] ({collection_name}) Score: {score:.4f}")
    print(f"   Source: {company} ({year}) | p.{page_no} | chunk={chunk}")
    print(f"   Content: {preview}...")
    print("-" * 80)


def search_vector_db(query: str, top_k: int = 5, mode: str = "semantic"):
    print(f"🔎 Query='{query}' | Mode={mode} | Top {top_k}")
    abs_path = os.path.abspath(VECTOR_DB_DIR)
    if not os.path.exists(abs_path):
        print(f"❌ Vector DB '{abs_path}' 를 찾을 수 없습니다. build_vector_db.py 실행 필요")
        return

    client = chromadb.PersistentClient(path=abs_path)
    collections = load_collections(client)
    if not collections:
        print("❌ 사용 가능한 컬렉션이 없습니다.")
        return

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    results_sem = results_kw = []
    if mode in {"semantic", "hybrid"}:
        results_sem = semantic_search(collections, model, query, top_k)
    if mode in {"keyword", "hybrid"}:
        results_kw = keyword_search(collections, query, top_k)

    # hybrid는 두 결과 리스트를 순차적으로 병합
    combined: list[tuple[str, str, dict, float, str]] = []
    if mode == "semantic":
        combined = [(name, doc, meta, score, "semantic") for name, doc, meta, score in results_sem]
    elif mode == "keyword":
        combined = [(name, doc, meta, score, "keyword") for name, doc, meta, score in results_kw]
    else:
        seen = set()
        for name, doc, meta, score in results_sem:
            key = (name, meta.get("doc_id"), meta.get("page_id"), meta.get("chunk_index"))
            combined.append((name, doc, meta, score, "semantic"))
            seen.add(key)
        for name, doc, meta, score in results_kw:
            key = (name, meta.get("doc_id"), meta.get("page_id"), meta.get("chunk_index"))
            if key in seen:
                continue
            combined.append((name, doc, meta, score, "keyword"))
    if not combined:
        print("검색 결과가 없습니다.")
        return

    for idx, (col_name, doc, meta, score, mode_tag) in enumerate(combined[:top_k], start=1):
        format_result(idx, mode_tag, col_name, doc, meta or {}, score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chroma 기반 ESG Vector 검색기")
    parser.add_argument("query", type=str, help="검색 질의어")
    parser.add_argument("--top-k", type=int, default=5, help="반환 결과 수")
    parser.add_argument(
        "--mode",
        choices=("semantic", "keyword", "hybrid"),
        default="semantic",
        help="검색 방식 선택 (semantic=임베딩, keyword=BM25, hybrid=두 방식 병합)",
    )

    args = parser.parse_args()

    search_vector_db(args.query, top_k=args.top_k, mode=args.mode)
