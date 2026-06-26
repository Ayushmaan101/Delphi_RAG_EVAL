# Delphi RAG Benchmarking Pipeline

A local, end-to-end RAG (Retrieval-Augmented Generation) benchmarking system that evaluates five chunking strategies using ChromaDB and BGE embeddings.

---

## Project Structure

```
Delphi_RAG_EVAL/
├── diverse_docs/                   # 25 source documents (PDF + TXT)
├── answer_key/
│   └── answer_key.json             # 75 Q&A pairs (3 per document)
├── chroma_db/                      # Persistent ChromaDB storage (auto-created)
├── results/                        # Benchmark output CSVs (auto-created)
├── src/
│   ├── __init__.py
│   ├── embedder.py                 # Single BGE model instance
│   ├── ingest.py                   # PDF/TXT loader → LangChain Documents
│   ├── chunkers.py                 # Five chunking strategy implementations
│   ├── vector_store.py             # ChromaDB client, build/query helpers
│   ├── retriever.py                # Thin retrieval wrapper
│   └── benchmark.py                # Evaluation loop, metrics, CSV output
├── main.py                         # CLI entry point
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> The first run downloads the `BAAI/bge-small-en-v1.5` model (~130 MB). Subsequent runs use the local cache.

---

## Usage

All commands are run from the project root directory.

### Ingest documents into ChromaDB

```bash
python main.py --ingest
```

Loads all documents from `diverse_docs/`, applies all five chunking strategies, embeds each chunk locally, and stores them in five separate ChromaDB collections under `chroma_db/`.

Re-running `--ingest` **deletes and recreates** all collections from scratch.

### Run the benchmark

```bash
python main.py --benchmark
```

Evaluates each strategy at K = 1, 3, 5 against the 75 Q&A pairs in `answer_key/answer_key.json`. Prints a results table and saves `results/benchmark_results_chunk_level.csv`.

Requires ingestion to have been run first.

### Run everything end-to-end

```bash
python main.py --all
```

Equivalent to `--ingest` followed by `--benchmark`.

---

## Chunking Strategies

| Strategy | Description |
|---|---|
| `fixed_no_overlap` | 500-character windows, zero overlap |
| `fixed_with_overlap` | 500-character windows, 50-char overlap (step = 450) |
| `recursive_character` | LangChain `RecursiveCharacterTextSplitter` (500/50, `["\n\n","\n",". "," ",""]`) |
| `semantic_chunking` | Sentence-level BGE cosine similarity segmentation (threshold = 0.60, max 400 chars) |
| `hierarchical_parent_child` | Parent windows of 1000 chars split into 200-char child chunks |

---

## Evaluation Metrics

Hit detection is **chunk-level**:

- **Hit**: a retrieved chunk satisfies both:
  1. `source_file` metadata matches `relevant_doc_id` in the answer key (bare filename comparison)
  2. The first 80 characters of the answer text appear verbatim in the chunk text

- **Hit Rate @ K**: fraction of queries with at least one hit in the top-K results
- **MRR @ K**: Mean Reciprocal Rank — average of 1/rank of the first hit (0 if no hit)

---

## Embedding Model

**BAAI/bge-small-en-v1.5** via `sentence-transformers`

- Query prefix: `"Represent this sentence for searching relevant passages: "`
- `normalize_embeddings=True` — dot product equals cosine similarity
- Single model instance shared across `embedder.py` and `chunkers.py` (no double loading)

---

## Output

`results/benchmark_results_chunk_level.csv` columns:

| Column | Description |
|---|---|
| `strategy` | Chunking strategy name |
| `k` | Retrieval depth (1, 3, or 5) |
| `hit_rate` | Fraction of queries with a hit |
| `mrr` | Mean Reciprocal Rank |
| `hits` | Raw hit count |
| `total` | Total queries evaluated (75) |
