"""Load .pdf and .txt documents from diverse_docs/ into LangChain Documents."""
import logging
import os
from pathlib import Path
from typing import List

import pdfplumber
from tqdm import tqdm

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document  # type: ignore

logger = logging.getLogger(__name__)


def load_documents(doc_dir: str = "diverse_docs") -> List[Document]:
    """Scan doc_dir recursively for .pdf and .txt files and return LangChain Documents.

    PDFs: one Document per page  — metadata: source_file, page (int)
    TXTs: one Document per file  — metadata: source_file, page=0
    source_file is always os.path.basename(filepath).
    """
    doc_path = Path(doc_dir)
    if not doc_path.exists():
        raise FileNotFoundError(f"Document directory not found: {doc_dir}")

    all_files = sorted(
        list(doc_path.rglob("*.pdf")) + list(doc_path.rglob("*.txt"))
    )

    documents: List[Document] = []
    pdf_count = 0
    txt_count = 0

    for fpath in tqdm(all_files, desc="Loading documents", unit="file"):
        source_file = os.path.basename(fpath)
        try:
            if fpath.suffix.lower() == ".pdf":
                with pdfplumber.open(str(fpath)) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        if text.strip():
                            documents.append(
                                Document(
                                    page_content=text,
                                    metadata={
                                        "source_file": source_file,
                                        "page": page_num,
                                    },
                                )
                            )
                pdf_count += 1
            else:
                text = fpath.read_text(encoding="utf-8", errors="replace")
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source_file": source_file,
                            "page": 0,
                        },
                    )
                )
                txt_count += 1
        except Exception as exc:
            logger.warning("Skipping %s: %s", source_file, exc)

    total = len(documents)
    print(f"\n✅ Loaded {total} documents from {doc_dir}/")
    print(f"   PDFs: {pdf_count}  |  TXT: {txt_count}  |  Total pages/docs: {total}")
    return documents
