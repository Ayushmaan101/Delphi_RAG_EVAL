"""Single BGE model instance shared across the project."""
from typing import List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_BATCH_SIZE = 64


def encode_texts(texts: List[str], is_query: bool = False) -> List[List[float]]:
    if not texts:
        return []

    if is_query:
        texts = [_QUERY_PREFIX + t for t in texts]

    n_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE
    show_bar = len(texts) > _BATCH_SIZE

    all_embeddings: List[List[float]] = []
    for i in tqdm(
        range(0, len(texts), _BATCH_SIZE),
        total=n_batches,
        desc="  Embedding batches",
        unit="batch",
        disable=not show_bar,
    ):
        batch = texts[i : i + _BATCH_SIZE]
        embs = _model.encode(
            batch,
            batch_size=_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.extend(embs.tolist())

    return all_embeddings
