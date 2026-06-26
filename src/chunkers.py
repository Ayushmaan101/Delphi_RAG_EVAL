"""Five chunking strategies, all returning LangChain Documents."""
import os
import re
from typing import Dict, List

import numpy as np
from tqdm import tqdm

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document  # type: ignore

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

# Reuse the single model instance from embedder to avoid loading weights twice.
from .embedder import _model


# ── helpers ───────────────────────────────────────────────────────────────────

def _group_by_source(documents: List[Document]) -> Dict[str, str]:
    """Concatenate all pages for each source file into one text string."""
    groups: Dict[str, str] = {}
    for doc in documents:
        key = os.path.basename(doc.metadata.get("source_file", ""))
        groups[key] = groups.get(key, "") + doc.page_content + "\n"
    return groups


def _make_doc(
    text: str, source_file: str, strategy: str, chunk_index: int
) -> Document:
    chunk_id = f"{source_file}_{strategy}_{chunk_index}"
    return Document(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "source_file": source_file,
            "strategy": strategy,
            "chunk_index": chunk_index,
        },
    )


# ── Strategy 1: Fixed-size, no overlap ────────────────────────────────────────

def chunk_fixed_no_overlap(documents: List[Document]) -> List[Document]:
    strategy = "fixed_no_overlap"
    chunk_size = 500
    groups = _group_by_source(documents)
    result: List[Document] = []

    for source_file, text in groups.items():
        idx = 0
        for start in range(0, len(text), chunk_size):
            chunk = text[start : start + chunk_size]
            if chunk.strip():
                result.append(_make_doc(chunk, source_file, strategy, idx))
                idx += 1

    return result


# ── Strategy 2: Fixed-size with 10% overlap ───────────────────────────────────

def chunk_fixed_with_overlap(documents: List[Document]) -> List[Document]:
    strategy = "fixed_with_overlap"
    chunk_size = 500
    step = 450  # overlap = 50 chars (10%)
    groups = _group_by_source(documents)
    result: List[Document] = []

    for source_file, text in groups.items():
        idx = 0
        start = 0
        while start < len(text):
            chunk = text[start : start + chunk_size]
            if chunk.strip():
                result.append(_make_doc(chunk, source_file, strategy, idx))
                idx += 1
            start += step

    return result


# ── Strategy 3: Recursive character splitter ──────────────────────────────────

def chunk_recursive_character(documents: List[Document]) -> List[Document]:
    strategy = "recursive_character"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    groups = _group_by_source(documents)
    result: List[Document] = []

    for source_file, text in groups.items():
        chunks = splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            if chunk.strip():
                result.append(_make_doc(chunk, source_file, strategy, idx))

    return result


# ── Strategy 4: Semantic chunking (BGE cosine similarity) ────────────────────

def _sub_split(sentences: List[str], max_chars: int) -> List[str]:
    """Pack sentences into sub-segments not exceeding max_chars."""
    segments: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    for s in sentences:
        addition = len(s) + (2 if current_parts else 0)  # ". " separator
        if current_len + addition <= max_chars:
            current_parts.append(s)
            current_len += addition
        else:
            if current_parts:
                segments.append(". ".join(current_parts))
            current_parts = [s]
            current_len = len(s)

    if current_parts:
        segments.append(". ".join(current_parts))
    return segments


def chunk_semantic(documents: List[Document]) -> List[Document]:
    strategy = "semantic_chunking"
    threshold = 0.60
    max_segment_chars = 400
    groups = _group_by_source(documents)
    result: List[Document] = []

    for source_file, text in tqdm(groups.items(), desc="Semantic chunking", unit="doc"):
        # Tokenise into sentences
        raw = re.split(r"\. |\n", text)
        sentences = [s.strip() for s in raw if len(s.strip()) >= 20]
        if not sentences:
            continue

        # Embed in batches of 64
        embeddings = _model.encode(
            sentences,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        )  # shape: (n, dim), already normalized

        # Group sentences into semantic segments
        segments: List[List[str]] = []
        current: List[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = float(np.dot(embeddings[i - 1], embeddings[i]))
            if sim < threshold:
                segments.append(current)
                current = [sentences[i]]
            else:
                current.append(sentences[i])
        segments.append(current)

        # Emit chunks; further split any segment > max_segment_chars
        idx = 0
        for seg_sentences in segments:
            segment = ". ".join(seg_sentences)
            if len(segment) <= max_segment_chars:
                if segment.strip():
                    result.append(_make_doc(segment.strip(), source_file, strategy, idx))
                    idx += 1
            else:
                for sub in _sub_split(seg_sentences, max_segment_chars):
                    if sub.strip():
                        result.append(_make_doc(sub.strip(), source_file, strategy, idx))
                        idx += 1

    return result


# ── Strategy 5: Hierarchical parent-child ─────────────────────────────────────

def chunk_hierarchical(documents: List[Document]) -> List[Document]:
    strategy = "hierarchical_parent_child"
    parent_size = 1000
    child_size = 200
    groups = _group_by_source(documents)
    result: List[Document] = []
    global_child_idx = 0

    for source_file, text in groups.items():
        # Carve parent chunks
        parents = [
            text[i : i + parent_size]
            for i in range(0, len(text), parent_size)
            if text[i : i + parent_size].strip()
        ]

        for parent_text in parents:
            # Carve child chunks from this parent
            for j in range(0, len(parent_text), child_size):
                child = parent_text[j : j + child_size]
                if not child.strip():
                    continue
                chunk_id = f"{source_file}_{strategy}_{global_child_idx}"
                result.append(
                    Document(
                        page_content=child,
                        metadata={
                            "chunk_id": chunk_id,
                            "source_file": source_file,
                            "strategy": strategy,
                            "chunk_index": global_child_idx,
                            "parent_text": parent_text,
                        },
                    )
                )
                global_child_idx += 1

    return result
