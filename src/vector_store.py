"""ChromaDB persistent vector store: build and query collections."""
import os
from pathlib import Path
from typing import Dict, List

import chromadb
from tqdm import tqdm

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document  # type: ignore

from .embedder import encode_texts

_ROOT = Path(__file__).parent.parent
_DB_PATH = str(_ROOT / "chroma_db")

client = chromadb.PersistentClient(path=_DB_PATH)

COLLECTIONS: Dict[str, str] = {
    "fixed_no_overlap":          "rag_fixed_no_overlap",
    "fixed_with_overlap":        "rag_fixed_overlap",
    "recursive_character":       "rag_recursive",
    "semantic_chunking":         "rag_semantic",
    "hierarchical_parent_child": "rag_hierarchical",
}

_UPSERT_BATCH = 100


def _clean_meta(meta: dict) -> dict:
    """Ensure all metadata values are ChromaDB-compatible primitives."""
    cleaned = {}
    for k, v in meta.items():
        if v is None:
            cleaned[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
        # Always guarantee source_file is a bare filename
    if "source_file" in cleaned:
        cleaned["source_file"] = os.path.basename(str(cleaned["source_file"]))
    return cleaned


def build_collection(strategy_name: str, documents: List[Document]) -> None:
    """Delete any existing collection, re-embed all chunks, and upsert."""
    col_name = COLLECTIONS[strategy_name]

    # Drop and recreate
    try:
        client.delete_collection(col_name)
    except Exception:
        pass
    collection = client.create_collection(col_name)

    texts = [doc.page_content for doc in documents]
    metadatas = [_clean_meta(dict(doc.metadata)) for doc in documents]
    ids = [m["chunk_id"] for m in metadatas]

    # Embed all at once (encode_texts handles internal batching)
    print(f"  Embedding {len(texts):,} chunks …")
    vectors = encode_texts(texts, is_query=False)

    # Upsert in batches of 100
    for i in tqdm(
        range(0, len(ids), _UPSERT_BATCH),
        desc=f"  Upserting {col_name}",
        unit="batch",
    ):
        sl = slice(i, i + _UPSERT_BATCH)
        collection.upsert(
            ids=ids[sl],
            documents=texts[sl],
            embeddings=vectors[sl],
            metadatas=metadatas[sl],
        )

    print(f"✅ Built collection '{col_name}': {len(ids):,} chunks")


def query_collection(strategy_name: str, query_text: str, k: int) -> List[dict]:
    """Query a collection and return ranked results."""
    col_name = COLLECTIONS[strategy_name]
    try:
        collection = client.get_collection(col_name)
    except Exception as exc:
        raise RuntimeError(
            f"Collection '{col_name}' not found. Run --ingest first."
        ) from exc

    count = collection.count()
    n_results = min(k, count)
    if n_results == 0:
        return []

    query_vec = encode_texts([query_text], is_query=True)[0]
    raw = collection.query(
        query_embeddings=[query_vec],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output: List[dict] = []
    for rank, (text, meta, dist) in enumerate(
        zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]),
        start=1,
    ):
        output.append({"text": text, "metadata": meta, "distance": dist, "rank": rank})
    return output
