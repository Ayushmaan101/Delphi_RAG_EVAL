"""Evaluate all chunking strategies against the answer key."""
import json
import os
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from .retriever import retrieve

_ROOT = Path(__file__).parent.parent
_ANSWER_KEY = _ROOT / "answer_key" / "answer_key.json"
_RESULTS_DIR = _ROOT / "results"
_CHUNK_CSV = _RESULTS_DIR / "benchmark_results_chunk_level.csv"
_DOC_CSV   = _RESULTS_DIR / "benchmark_results_doc_level.csv"

STRATEGIES: List[str] = [
    "fixed_no_overlap",
    "fixed_with_overlap",
    "recursive_character",
    "semantic_chunking",
    "hierarchical_parent_child",
]

K_VALUES: List[int] = [1, 3, 5]


def _eval_loop(
    answer_key: List[dict],
    strategy: str,
    k: int,
    eval_type: str,
) -> dict:
    hits = 0
    reciprocal_ranks: List[float] = []

    for qa in tqdm(
        answer_key,
        desc=f"  [{eval_type}] {strategy} K={k}",
        unit="query",
        leave=False,
    ):
        query        = qa["query"]
        relevant_doc = os.path.basename(qa["relevant_doc_id"])
        answer_text  = qa["answer"].lower().strip()

        results = retrieve(strategy, query, k)

        found_rank = None
        for res in results:
            retrieved_doc = os.path.basename(
                res["metadata"].get("source_file", "")
            )

            if eval_type == "chunk_level":
                # For hierarchical, child chunks are too small — check parent_text
                if res["metadata"].get("strategy") == "hierarchical_parent_child":
                    search_text = res["metadata"].get("parent_text", "").lower().strip()
                else:
                    search_text = res["text"].lower().strip()

                if retrieved_doc == relevant_doc and answer_text[:80] in search_text:
                    found_rank = res["rank"]
                    break

            else:  # doc_level
                if retrieved_doc == relevant_doc:
                    found_rank = res["rank"]
                    break

        hits += int(found_rank is not None)
        reciprocal_ranks.append(1.0 / found_rank if found_rank else 0.0)

    n = len(answer_key)
    return {
        "eval_type": eval_type,
        "strategy":  strategy,
        "k":         k,
        "hit_rate":  round(hits / n, 4) if n else 0.0,
        "mrr":       round(sum(reciprocal_ranks) / n, 4) if n else 0.0,
        "hits":      hits,
        "total":     n,
    }


def run_benchmark() -> pd.DataFrame:
    with open(_ANSWER_KEY, encoding="utf-8") as fh:
        answer_key: List[dict] = json.load(fh)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []

    for strategy in STRATEGIES:
        for k in K_VALUES:
            for eval_type in ("chunk_level", "doc_level"):
                records.append(_eval_loop(answer_key, strategy, k, eval_type))

    df = pd.DataFrame(records)

    df[df["eval_type"] == "chunk_level"].drop(columns="eval_type").to_csv(_CHUNK_CSV, index=False)
    df[df["eval_type"] == "doc_level"].drop(columns="eval_type").to_csv(_DOC_CSV, index=False)

    print(f"\n📊 Results saved to:")
    print(f"   {_CHUNK_CSV}")
    print(f"   {_DOC_CSV}")
    return df


def _print_table(df: pd.DataFrame, title: str) -> None:
    S, K, H, M = 29, 3, 9, 7
    top    = f"┌{'─'*(S+2)}┬{'─'*(K+2)}┬{'─'*(H+2)}┬{'─'*(M+2)}┐"
    head   = f"│ {'Strategy':<{S}} │ {'K':>{K}} │ {'Hit Rate':>{H}} │ {'MRR':>{M}} │"
    sep    = f"├{'─'*(S+2)}┼{'─'*(K+2)}┼{'─'*(H+2)}┼{'─'*(M+2)}┤"
    bottom = f"└{'─'*(S+2)}┴{'─'*(K+2)}┴{'─'*(H+2)}┴{'─'*(M+2)}┘"

    print(f"\n── {title} {'─' * max(0, 54 - len(title))}")
    print(top)
    print(head)
    print(sep)
    for _, row in df.iterrows():
        print(
            f"│ {row['strategy']:<{S}} │ {int(row['k']):>{K}} │ "
            f"{row['hit_rate']:>{H}.4f} │ {row['mrr']:>{M}.4f} │"
        )
    print(bottom)

    k1 = df[df["k"] == 1]
    if not k1.empty:
        winner = k1.loc[k1["hit_rate"].idxmax()]
        print(f"🏆 Winner at K=1: {winner['strategy']} with Hit Rate = {winner['hit_rate']:.2f}")


def _print_gap_table(doc_df: pd.DataFrame, chunk_df: pd.DataFrame) -> None:
    S, D, C, G = 29, 10, 11, 9
    top    = f"┌{'─'*(S+2)}┬{'─'*(D+2)}┬{'─'*(C+2)}┬{'─'*(G+2)}┐"
    head   = f"│ {'Strategy':<{S}} │ {'Doc-Level':>{D}} │ {'Chunk-Level':>{C}} │ {'Gap':>{G}} │"
    sep    = f"├{'─'*(S+2)}┼{'─'*(D+2)}┼{'─'*(C+2)}┼{'─'*(G+2)}┤"
    bottom = f"└{'─'*(S+2)}┴{'─'*(D+2)}┴{'─'*(C+2)}┴{'─'*(G+2)}┘"

    k1_doc   = doc_df[doc_df["k"] == 1].set_index("strategy")
    k1_chunk = chunk_df[chunk_df["k"] == 1].set_index("strategy")

    print(f"\n── EVALUATION GAP ANALYSIS {'─' * 30}")
    print(top)
    print(head)
    print(sep)
    for strat in STRATEGIES:
        doc_hr   = k1_doc.loc[strat, "hit_rate"]   if strat in k1_doc.index   else 0.0
        chunk_hr = k1_chunk.loc[strat, "hit_rate"] if strat in k1_chunk.index else 0.0
        gap      = chunk_hr - doc_hr
        print(
            f"│ {strat:<{S}} │ {doc_hr:>{D}.4f} │ {chunk_hr:>{C}.4f} │ {gap:>+{G}.4f} │"
        )
    print(bottom)
    print(
        "\n  The gap represents how often the retriever found the right\n"
        "  document but failed to surface the exact answer passage."
    )


def print_results_table(df: pd.DataFrame) -> None:
    doc_df   = df[df["eval_type"] == "doc_level"].drop(columns="eval_type").reset_index(drop=True)
    chunk_df = df[df["eval_type"] == "chunk_level"].drop(columns="eval_type").reset_index(drop=True)

    _print_table(doc_df,   "DOCUMENT-LEVEL EVALUATION (did retriever find the right document?)")
    _print_table(chunk_df, "CHUNK-LEVEL EVALUATION (does retrieved chunk contain exact answer?)")
    _print_gap_table(doc_df, chunk_df)
