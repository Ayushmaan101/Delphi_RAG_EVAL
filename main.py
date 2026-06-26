"""Entry point for the RAG benchmarking pipeline."""
import argparse
import sys

# Windows UTF-8 safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def print_header(title: str) -> None:
    border = "═" * (len(title) + 4)
    print(f"\n╔{border}╗")
    print(f"║  {title}  ║")
    print(f"╚{border}╝\n")


def run_ingest() -> None:
    print_header("INGESTION")
    from src.ingest import load_documents
    from src.chunkers import (
        chunk_fixed_no_overlap,
        chunk_fixed_with_overlap,
        chunk_recursive_character,
        chunk_semantic,
        chunk_hierarchical,
    )
    from src.vector_store import build_collection, COLLECTIONS

    documents = load_documents("diverse_docs")

    strategy_fns = {
        "fixed_no_overlap":          chunk_fixed_no_overlap,
        "fixed_with_overlap":        chunk_fixed_with_overlap,
        "recursive_character":       chunk_recursive_character,
        "semantic_chunking":         chunk_semantic,
        "hierarchical_parent_child": chunk_hierarchical,
    }

    for strategy_name, fn in strategy_fns.items():
        print(f"\n── Chunking: {strategy_name} ──")
        chunks = fn(documents)
        print(f"   Produced {len(chunks):,} chunks")
        build_collection(strategy_name, chunks)

    print("\n✅ Ingestion complete.")


def run_benchmark() -> None:
    print_header("BENCHMARK")
    from src.benchmark import run_benchmark as _run, print_results_table

    df = _run()
    print_results_table(df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Benchmarking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --ingest\n"
            "  python main.py --benchmark\n"
            "  python main.py --all\n"
        ),
    )
    parser.add_argument("--ingest", action="store_true", help="Chunk and ingest documents into ChromaDB")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark evaluation")
    parser.add_argument("--all", action="store_true", help="Run ingest then benchmark")

    args = parser.parse_args()

    if not any([args.ingest, args.benchmark, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.ingest or args.all:
        run_ingest()

    if args.benchmark or args.all:
        run_benchmark()

    if args.all:
        print("\n🎉 Pipeline complete.")
        print("📊 Results saved to results/benchmark_results_chunk_level.csv")


if __name__ == "__main__":
    main()
