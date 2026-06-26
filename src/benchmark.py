"""Evaluate all chunking strategies against the answer key."""
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from .retriever import retrieve

_ROOT = Path(__file__).parent.parent
_ANSWER_KEY = _ROOT / "answer_key" / "answer_key.json"
_RESULTS_DIR = _ROOT / "results"
_RESULTS_FILE = _RESULTS_DIR / "benchmark_results_chunk_level.csv"

STRATEGIES: List[str] = [
    "fixed_no_overlap",
    "fixed_with_overlap",
    "recursive_character",
    "semantic_chunking",
    "hierarchical_parent_child",
]

K_VALUES: List[int] = [1, 3, 5]


def _is_hit(retrieved: dict, relevant_doc: str, answer_text: str) -> bool:
    """Chunk-level hit: source file matches AND answer prefix found in retrieved text."""
    meta = retrieved.get("metadata", {})
    doc_match = os.path.basename(meta.get("source_file", "")) == os.path.basename(relevant_doc)
    text_match = answer_text[:80] in retrieved.get("text", "")
    return doc_match and text_match


def run_benchmark() -> pd.DataFrame:
    with open(_ANSWER_KEY, encoding="utf-8") as fh:
        answer_key: List[dict] = json.load(fh)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []

    for strategy in STRATEGIES:
        for k in K_VALUES:
            hits = 0
            reciprocal_ranks: List[float] = []

            for qa in tqdm(
                answer_key,
                desc=f"{strategy} K={k}",
                unit="query",
                leave=False,
            ):
                query = qa["query"]
                relevant_doc = qa["relevant_doc_id"]
                answer_text = qa["answer"]

                results = retrieve(strategy, query, k)

                hit = False
                rr = 0.0
                for res in results:
                    if _is_hit(res, relevant_doc, answer_text):
                        hit = True
                        rr = 1.0 / res["rank"]
                        break

                hits += int(hit)
                reciprocal_ranks.append(rr)

            n = len(answer_key)
            hit_rate = hits / n if n else 0.0
            mrr = sum(reciprocal_ranks) / n if n else 0.0
            records.append(
                {
                    "strategy": strategy,
                    "k": k,
                    "hit_rate": round(hit_rate, 4),
                    "mrr": round(mrr, 4),
                    "hits": hits,
                    "total": n,
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(_RESULTS_FILE, index=False)
    print(f"\n📊 Results saved to {_RESULTS_FILE}")
    return df


def _pad(s: str, width: int) -> str:
    return s.ljust(width)


def print_results_table(df: pd.DataFrame) -> None:
    col_widths = {
        "Strategy": 30,
        "K": 4,
        "Hit Rate": 10,
        "MRR": 10,
    }
    header = (
        f"{'Strategy':<30} {'K':>4}  {'Hit Rate':>10}  {'MRR':>10}"
    )
    border = "═" * len(header)

    print(f"\n╔{border}╗")
    print(f"║ {header} ║")
    print(f"╠{border}╣")

    for _, row in df.iterrows():
        line = (
            f"{row['strategy']:<30} {int(row['k']):>4}  "
            f"{row['hit_rate']:>10.4f}  {row['mrr']:>10.4f}"
        )
        print(f"║ {line} ║")

    print(f"╚{border}╝")

    # Winner at K=1
    k1 = df[df["k"] == 1]
    if not k1.empty:
        winner_row = k1.loc[k1["hit_rate"].idxmax()]
        print(
            f"\n🏆 Winner at K=1: {winner_row['strategy']} "
            f"with Hit Rate = {winner_row['hit_rate']:.2f}"
        )
