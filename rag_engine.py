"""
rag_engine.py
─────────────
문서를 벡터화하여 저장하고, 질문과 관련된 내용을 검색하는 RAG 엔진.
국립암센터 암정보센터 자료 등을 넣으면 답변 품질이 크게 올라갑니다.
"""

import os
import hashlib
import chromadb
from chromadb.utils import embedding_functions
import streamlit as st

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "cancer_info"
CHUNK_SIZE = 500       # 한 청크의 글자 수
CHUNK_OVERLAP = 100    # 청크 간 겹치는 글자 수
TOP_K = 5              # 검색 시 가져올 문서 수


def _get_embedding_function():
    """임베딩 함수를 반환합니다. OpenAI 키가 있으면 OpenAI, 없으면 무료 로컬 모델."""
    api_key = st.secrets.get("OPENAI_API_KEY")
    if api_key:
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small",
        )
    else:
        # OpenAI 키가 없으면 무료 로컬 임베딩 사용
        return embedding_functions.DefaultEmbeddingFunction()


def _split_text(text: str) -> list[str]:
    """긴 텍스트를 작은 청크로 나눕니다."""
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
    """파일의 해시값 (변경 감지용)"""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_documents() -> list[dict]:
    """
    data/ 폴더의 .txt 파일들을 읽어 청크로 나눕니다.

    Returns:
        [{"text": "...", "source": "파일명", "chunk_id": "..."}, ...]
    """
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
            docs.append(
                {
                    "text": chunk,
                    "source": filename,
                    "chunk_id": f"{file_id}_{i:04d}",
                }
            )

    return docs


def build_vector_store(force_rebuild: bool = False) -> chromadb.Collection:
    """
    문서를 벡터 DB에 저장합니다. 이미 존재하면 건너뜁니다.

    Args:
        force_rebuild: True면 기존 DB를 삭제하고 처음부터 다시 만듭니다.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = _get_embedding_function()

    # 기존 컬렉션 삭제 (강제 재구축 시)
    if force_rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    # 이미 데이터가 있으면 건너뜀
    if collection.count() > 0 and not force_rebuild:
        return collection

    # 문서 로드 & 저장
    docs = load_documents()
    if not docs:
        return collection

    # ChromaDB에 배치 추가 (한 번에 최대 100개씩)
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        collection.add(
            ids=[d["chunk_id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[{"source": d["source"]} for d in batch],
        )

    return collection


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    질문과 가장 관련 있는 문서 청크를 검색합니다.

    Returns:
        [{"text": "...", "source": "...", "distance": 0.xx}, ...]
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        embed_fn = _get_embedding_function()
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )

        if collection.count() == 0:
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
        )

        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append(
                {
                    "text": results["documents"][0][i],
                    "source": results["metadatas"][0][i]["source"],
                    "distance": results["distances"][0][i] if results.get("distances") else 0,
                }
            )

        return search_results

    except Exception:
        return []


def format_context(results: list[dict]) -> str:
    """검색 결과를 LLM 프롬프트에 넣을 수 있는 형태로 정리합니다."""
    if not results:
        return ""

    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"[참고자료 {i}] (출처: {r['source']})\n{r['text']}"
        )

    return "\n\n".join(context_parts)
