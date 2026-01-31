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

def search_vector_db(query: str, top_k: int = 5, filter_company: str = None, filter_year: int = None):
    print(f"🔎 Search Query: '{query}' (Top {top_k})")
    
    # 1. Initialize Client & Collection
    abs_path = os.path.abspath(VECTOR_DB_DIR)
    if not os.path.exists(abs_path):
        print(f"❌ Vector DB directory '{abs_path}' not found. Run build_vector_db.py first.")
        return []

    client = chromadb.PersistentClient(path=abs_path)
    # print(f"DEBUG: Vector DB Path: {abs_path}") # Reduce noise
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Collection '{COLLECTION_NAME}' not found or error occurred: {e}")
        return []

    # 2. Embed Query
    print(f"📦 Loading Model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_vec = model.encode([query]).tolist()
    
    # Build Filter
    conditions = []
    if filter_company:
        conditions.append({"company_name": filter_company})
    if filter_year:
        conditions.append({"report_year": filter_year})
        
    if len(conditions) > 1:
        final_where = {"$and": conditions}
    elif len(conditions) == 1:
        final_where = conditions[0]
    else:
        final_where = None

    # 3. Query ChromaDB
    results = collection.query(
        query_embeddings=query_vec,
        n_results=top_k,
        where=final_where
    )
    
    # 4. Process Results
    structured_results = []
    if results['documents']:
        for idx, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][idx]
            score = results['distances'][0][idx]
            doc_id = results['ids'][0][idx]
            
            structured_results.append({
                "id": doc_id,
                "content": doc,
                "metadata": meta,
                "score": score
            })

    # Print Results (only if run as main script or specific flag, but for now always print is fine for debug)
    if not structured_results:
        print("No results found.")
    else:
        print(f"\n✅ Found {len(structured_results)} matches:\n")
        for i, res in enumerate(structured_results):
            print(f"[Rank {i+1}] Distance: {res['score']:.4f}")
            print(f"   Source: {res['metadata'].get('company_name')} ({res['metadata'].get('report_year')}) | p.{res['metadata'].get('page_no')}")
            print(f"   Chunk: {res['metadata'].get('chunk_index')} | ID: {res['id']}")
            print(f"   Content: {res['content'][:200].replace(chr(10), ' ')}...")
            print("-" * 60)
            
    return structured_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search ESG Vector Database")
    parser.add_argument("query", type=str, help="The search query string")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--company", type=str, default=None, help="Filter by company name")
    parser.add_argument("--year", type=int, default=None, help="Filter by report year")
    
    args = parser.parse_args()
    
    search_vector_db(
        args.query, 
        top_k=args.top_k, 
        filter_company=args.company, 
        filter_year=args.year
    )
