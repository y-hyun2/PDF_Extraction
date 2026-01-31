"""Chroma 벡터 DB 검색 스크립트 (Semantic + Keyword + 로컬 Reranker)."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

try:
    from kiwipiepy import Kiwi

    KIWI = Kiwi()
except Exception:  # pylint: disable=broad-except
    KIWI = None

try:
    RERANKER = SentenceTransformer("BAAI/bge-reranker-v2-m3")
except Exception:
    RERANKER = None

VECTOR_DB_DIR = "vector_db"
COLLECTIONS = ["esg_pages", "esg_chunks"]
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
MAX_KEYWORD_DOCS = 2000
RERANK_CANDIDATES = 50


def tokenize(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if KIWI is not None:
        return [token.form for token in KIWI.tokenize(text) if token.form.strip()]
    return re.findall(r"[0-9A-Za-z가-힣]+", text)


def bm25_scores(corpus_tokens: List[List[str]], query_tokens: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    if not corpus_tokens:
        return []
    N = len(corpus_tokens)
    doc_lens = [len(tokens) for tokens in corpus_tokens]
    avgdl = sum(doc_lens) / max(N, 1)
    df: Counter[str] = Counter()
    for tokens in corpus_tokens:
        for term in set(tokens):
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


def semantic_search(collections, model, query: str, top_k: int) -> List[Tuple[str, str, Dict, float]]:
    query_vec = model.encode([query]).tolist()
    results = []
    for collection in collections:
        resp = collection.query(query_embeddings=query_vec, n_results=top_k)
        docs = resp.get("documents") or []
        if not docs:
            continue
        for doc, meta, dist in zip(docs[0], resp["metadatas"][0], resp["distances"][0]):
            results.append((collection.name, doc, meta, dist))
    results.sort(key=lambda item: item[3])
    return results[:top_k]


def keyword_search(collections, query: str, top_k: int) -> List[Tuple[str, str, Dict, float]]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    docs_all = []
    for collection in collections:
        data = collection.get(include=["documents", "metadatas"], limit=MAX_KEYWORD_DOCS)
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        for text, meta in zip(docs, metas):
            tokens = tokenize(text)
            docs_all.append((collection.name, text, meta, tokens))
    corpus_tokens = [item[3] for item in docs_all]
    scores = bm25_scores(corpus_tokens, query_tokens)
    ranked = sorted(zip(docs_all, scores), key=lambda x: x[1], reverse=True)
    return [(name, text, meta, score) for (name, text, meta, _), score in ranked[:top_k]]


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


def search_vector_db(query: str, top_k: int = 5, mode: str = "hybrid"):
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

    semantic_res = keyword_res = []
    if mode in {"semantic", "hybrid"}:
        semantic_res = semantic_search(collections, model, query, top_k)
    if mode in {"keyword", "hybrid"}:
        keyword_res = keyword_search(collections, query, top_k)

    combined = []
    if mode == "semantic":
        combined = [(name, doc, meta, score, "semantic") for name, doc, meta, score in semantic_res]
    elif mode == "keyword":
        combined = [(name, doc, meta, score, "keyword") for name, doc, meta, score in keyword_res]
    else:
        seen = set()
        for name, doc, meta, score in semantic_res:
            key = (name, meta.get("doc_id"), meta.get("page_id"), meta.get("chunk_index"))
            combined.append((name, doc, meta or {}, score, "semantic"))
            seen.add(key)
        for name, doc, meta, score in keyword_res:
            key = (name, meta.get("doc_id"), meta.get("page_id"), meta.get("chunk_index"))
            if key in seen:
                continue
            combined.append((name, doc, meta or {}, score, "keyword"))
    if not combined:
        print("검색 결과가 없습니다.")
        return

    if RERANKER is not None:
        docs = [doc for _, doc, _, _, _ in combined]
        scores = RERANKER.compute_score([[query, d] for d in docs], batch_size=16)
        combined = sorted(zip(combined, scores), key=lambda x: x[1], reverse=True)
    else:
        combined = [(item, item[3]) for item in combined]

    for idx, ((col_name, doc, meta, _, mode_tag), score) in enumerate(combined[:top_k], start=1):
        format_result(idx, mode_tag, col_name, doc, meta, float(score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chroma 기반 ESG Vector 검색기")
    parser.add_argument("query", type=str, help="검색 질의어")
    parser.add_argument("--top-k", type=int, default=5, help="반환 결과 수")
    parser.add_argument(
        "--mode",
        choices=("semantic", "keyword", "hybrid"),
        default="hybrid",
        help="검색 방식 선택",
    )
    args = parser.parse_args()
    search_vector_db(args.query, top_k=args.top_k, mode=args.mode)
