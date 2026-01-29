"""
Search Vector Database (Phase 1).
Query the ChromaDB collection for relevant document chunks.

Usage:
    python src/search_vector_db.py "query string" [--top-k 5]
"""

import argparse
import sys
import os
import chromadb
from sentence_transformers import SentenceTransformer

# Configuration (Must match build_vector_db.py)
VECTOR_DB_DIR = "vector_db"
COLLECTION_NAME = "esg_documents"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

def search_vector_db(query: str, top_k: int = 5):
    print(f"🔎 Search Query: '{query}' (Top {top_k})")
    
    # 1. Initialize Client & Collection
    abs_path = os.path.abspath(VECTOR_DB_DIR)
    if not os.path.exists(abs_path):
        print(f"❌ Vector DB directory '{abs_path}' not found. Run build_vector_db.py first.")
        return

    client = chromadb.PersistentClient(path=abs_path)
    print(f"DEBUG: Vector DB Path: {abs_path}")
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Collection '{COLLECTION_NAME}' not found or error occurred: {e}")
        return

    # 2. Embed Query
    print(f"📦 Loading Model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_vec = model.encode([query]).tolist()
    
    # 3. Query ChromaDB
    results = collection.query(
        query_embeddings=query_vec,
        n_results=top_k
        # where={"company_name": "HDEC"} # Example filter
    )
    
    # 4. Print Results
    if not results['documents']:
        print("No results found.")
        return

    print(f"\n✅ Found {len(results['documents'][0])} matches:\n")
    
    for idx, doc in enumerate(results['documents'][0]):
        meta = results['metadatas'][0][idx]
        score = results['distances'][0][idx] 
        # Note: Chroma default distance is often L2 or Cosine distance (lower is better or higher is better depending on metric).
        # We used {"hnsw:space": "cosine"} in build script. 
        # Cosine distance = 1 - Cosine Similarity. So lower is better (0 means identical).
        
        print(f"[Rank {idx+1}] Distance: {score:.4f}")
        print(f"   Source: {meta.get('company_name')} ({meta.get('report_year')}) | p.{meta.get('page_no')}")
        print(f"   Chunk: {meta.get('chunk_index')} | ID: {results['ids'][0][idx]}")
        print(f"   Content: {doc[:200].replace(chr(10), ' ')}...") # Preview 200 chars, remove newlines
        print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search ESG Vector Database")
    parser.add_argument("query", type=str, help="The search query string")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    search_vector_db(args.query, top_k=args.top_k)
