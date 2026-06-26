"""Thin wrapper around vector_store.query_collection."""
from typing import List

from .vector_store import query_collection


def retrieve(strategy_name: str, query: str, k: int) -> List[dict]:
    return query_collection(strategy_name, query, k)
