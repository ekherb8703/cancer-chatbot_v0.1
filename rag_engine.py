"""
rag_engine.py
─────────────
문서를 벡터화하여 저장하고, 질문과 관련된 내용을 검색하는 RAG 엔진.
Streamlit Cloud 호환을 위해 인메모리 DB를 사용합니다.
"""

import os
import hashlib
import chromadb

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
COLLECTION_NAME = "cancer_info"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5


def _split_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - CHUNK_OVERLAP
    return chunks


def _file_hash(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_documents() -> list[dict]:
    docs = []
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        return docs

    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = _split_text(text)
        file_id = _file_hash(filepath)[:8]
        for i, chunk in enumerate(chunks):
            docs.append({
                "text": chunk,
                "source": filename,
                "chunk_id": f"{file_id}_{i:04d}",
            })
    return docs


def build_vector_store():
    """
    문서를 인메모리 벡터 DB에 저장하고 (client, collection) 튜플을 반환합니다.
    st.cache_resource로 캐시되므로 앱 실행 중 유지됩니다.
    """
    client = chromadb.Client()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    docs = load_documents()
    if not docs:
        return collection

    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        collection.add(
            ids=[d["chunk_id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[{"source": d["source"]} for d in batch],
        )

    return collection


def search(collection, query: str, top_k: int = TOP_K) -> list[dict]:
    """
    질문과 가장 관련 있는 문서 청크를 검색합니다.
    collection은 build_vector_store()에서 반환된 객체를 넣어주세요.
    """
    try:
        if collection is None or collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
        )

        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "distance": results["distances"][0][i] if results.get("distances") else 0,
            })
        return search_results

    except Exception:
        return []


def format_context(results: list[dict]) -> str:
    if not results:
        return ""
    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"[참고자료 {i}] (출처: {r['source']})\n{r['text']}"
        )
    return "\n\n".join(context_parts)
