#!/usr/bin/env python3
"""
diverse_downloader.py — Downloads 25 documents across 5 categories.
Saves everything to diverse_docs/ at the project root.
Does NOT modify any existing project files.

Usage: py diverse_downloader.py
"""

import sys
import re
import json
import time
import gzip
import ssl
import zipfile
import pathlib
import urllib.request
import urllib.error
import urllib.parse

# Force UTF-8 output on Windows so box-drawing / emoji chars render correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from tqdm import tqdm
except ImportError:
    raise SystemExit("ERROR: tqdm not installed. Run: pip install tqdm")

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent
DIVERSE_DOCS = ROOT / "diverse_docs"

DIRS = {
    "financial": DIVERSE_DOCS / "financial",
    "wikipedia": DIVERSE_DOCS / "wikipedia",
    "academic":  DIVERSE_DOCS / "academic",
    "legal":     DIVERSE_DOCS / "legal",
    "technical": DIVERSE_DOCS / "technical",
}

# ─── Stats ────────────────────────────────────────────────────────────────────
stats = {
    "Financial": {"total": 5, "success": 0, "failed": 0, "failures": []},
    "Wikipedia": {"total": 5, "success": 0, "failed": 0, "failures": []},
    "Academic":  {"total": 5, "success": 0, "failed": 0, "failures": []},
    "Legal":     {"total": 5, "success": 0, "failed": 0, "failures": []},
    "Technical": {"total": 5, "success": 0, "failed": 0, "failures": []},
}

# ─── SSL helpers ──────────────────────────────────────────────────────────────
def _ssl_ctx(verify: bool = True) -> ssl.SSLContext:
    if verify:
        return ssl.create_default_context()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


# ─── HTTP helpers ─────────────────────────────────────────────────────────────
def fetch(url: str, headers: dict = None, timeout: int = 60) -> bytes:
    """Download URL bytes; falls back to no-SSL-verify on SSLError."""
    req = urllib.request.Request(url, headers=headers or {})
    raw = None
    content_encoding = ""

    for verify in (True, False):
        try:
            ctx = _ssl_ctx(verify)
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                content_encoding = resp.headers.get("Content-Encoding", "")
                raw = resp.read()
            break
        except ssl.SSLError:
            if not verify:
                raise
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} {exc.reason}")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"URL error: {exc.reason}")

    if raw is None:
        raise RuntimeError("All fetch attempts exhausted")

    if content_encoding.lower() == "gzip" or raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except Exception:
            pass  # Already decoded or not actually gzip
    return raw


def fetch_str(url: str, headers: dict = None, timeout: int = 60) -> str:
    return fetch(url, headers, timeout).decode("utf-8", errors="replace")


def strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


# ─── Validation ───────────────────────────────────────────────────────────────
def validate(path: pathlib.Path):
    """Returns (ok: bool, reason: str)."""
    if not path.exists():
        return False, "does not exist"
    size = path.stat().st_size
    if size < 10_000:
        return False, f"too small ({size:,} bytes)"
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with open(path, "rb") as fh:
            if fh.read(4) != b"%PDF":
                return False, "invalid PDF header"
    elif suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) < 5_000:
            return False, f"too short ({len(text):,} chars)"
    return True, "ok"


def already_valid(path: pathlib.Path) -> bool:
    ok, _ = validate(path)
    return ok


# ─── Folder setup ─────────────────────────────────────────────────────────────
def make_dirs():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"  Folder structure ready under: {DIVERSE_DOCS}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 1 — Financial (SEC 10-K Filings)
# ═══════════════════════════════════════════════════════════════════════════════
COMPANIES = [
    {"name": "apple",      "cik": "0000320193", "industry": "Technology"},
    {"name": "jpmorgan",   "cik": "0000019617", "industry": "Banking"},
    {"name": "pfizer",     "cik": "0000078003", "industry": "Pharma"},
    {"name": "tesla",      "cik": "0001318605", "industry": "Automotive"},
    {"name": "exxonmobil", "cik": "0000034088", "industry": "Energy"},
]

SEC_HEADERS = {
    "User-Agent": "RAG-Benchmark-Research student@university.com",
    "Accept-Encoding": "gzip, deflate",
}


def download_financial():
    print("\n" + "═" * 45)
    print("  DOWNLOADING: FINANCIAL (SEC 10-Ks)")
    print("═" * 45)

    for company in tqdm(COMPANIES, desc="Financial", unit="filing"):
        name = company["name"]
        cik_raw = company["cik"]
        cik_int = str(int(cik_raw))

        pdf_path = DIRS["financial"] / f"{name}_10k.pdf"
        txt_path = DIRS["financial"] / f"{name}_10k.txt"

        if already_valid(pdf_path) or already_valid(txt_path):
            tqdm.write(f"  ✓ {name}: already downloaded, skipping")
            stats["Financial"]["success"] += 1
            continue

        tqdm.write(f"  → {name}: fetching submission list…")
        try:
            # 1. Submissions JSON
            sub_url = f"https://data.sec.gov/submissions/CIK{cik_raw}.json"
            sub_data = json.loads(fetch_str(sub_url, SEC_HEADERS))
            time.sleep(0.75)

            # 2. Find most recent 10-K + primaryDocument (no extra HTTP call needed)
            recent = sub_data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            accessions = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])

            ten_k_idx = next((i for i, f in enumerate(forms) if f == "10-K"), None)
            if ten_k_idx is None:
                raise RuntimeError("No 10-K filing found in recent filings")

            accession = accessions[ten_k_idx]
            accession_nodash = accession.replace("-", "")
            primary_doc = (
                primary_docs[ten_k_idx]
                if ten_k_idx < len(primary_docs)
                else ""
            )

            # Use www.sec.gov for archives (data.sec.gov/Archives index.json may 404)
            base_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_int}/{accession_nodash}/"
            )
            tqdm.write(f"  → {name}: 10-K {accession}, primary: {primary_doc}")

            # 3. Try PDF (swap .htm/.html → .pdf in the primary document name)
            saved = False
            if primary_doc and primary_doc.lower().endswith((".htm", ".html")):
                pdf_candidate = re.sub(
                    r"\.html?$", ".pdf", primary_doc, flags=re.IGNORECASE
                )
                try:
                    tqdm.write(f"  → {name}: trying PDF '{pdf_candidate}'…")
                    pdf_data = fetch(base_url + pdf_candidate, SEC_HEADERS, timeout=120)
                    time.sleep(0.75)
                    pdf_path.write_bytes(pdf_data)
                    tqdm.write(f"  ✓ {name}: PDF saved ({len(pdf_data):,} B)")
                    stats["Financial"]["success"] += 1
                    saved = True
                except Exception:
                    tqdm.write(f"  → {name}: no PDF, downloading HTML…")

            # 4. Fall back: HTML primary doc → strip tags → save as .txt
            if not saved:
                if not primary_doc:
                    raise RuntimeError("No primaryDocument found in submissions JSON")
                html = fetch_str(base_url + primary_doc, SEC_HEADERS, timeout=180)
                time.sleep(0.75)
                text = strip_html(html)[:500_000]
                txt_path.write_text(text, encoding="utf-8")
                tqdm.write(f"  ✓ {name}: HTML→TXT saved ({len(text):,} chars)")
                stats["Financial"]["success"] += 1

        except Exception as exc:
            msg = str(exc)
            tqdm.write(f"  ✗ {name}: FAILED — {msg}")
            stats["Financial"]["failed"] += 1
            stats["Financial"]["failures"].append((f"{name}_10k.*", msg))


# ═══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 2 — Wikipedia (Long Articles)
# ═══════════════════════════════════════════════════════════════════════════════
WIKI_ARTICLES = [
    "World_War_II",
    "Human_genome",
    "Climate_change",
    "Ancient_Rome",
    "Quantum_mechanics",
]

WIKI_HEADERS = {"User-Agent": "RAG-Benchmark-Research student@university.com"}


def download_wikipedia():
    print("\n" + "═" * 45)
    print("  DOWNLOADING: WIKIPEDIA (Articles as TXT)")
    print("═" * 45)

    for article in tqdm(WIKI_ARTICLES, desc="Wikipedia", unit="article"):
        out_path = DIRS["wikipedia"] / f"{article.lower()}.txt"

        if already_valid(out_path):
            tqdm.write(f"  ✓ {article}: already downloaded, skipping")
            stats["Wikipedia"]["success"] += 1
            continue

        tqdm.write(f"  → {article}: fetching…")
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/html/{article}"
            html = fetch_str(url, WIKI_HEADERS)
            text = strip_html(html)[:300_000]
            out_path.write_text(text, encoding="utf-8")
            tqdm.write(f"  ✓ {article}: saved ({len(text):,} chars)")
            stats["Wikipedia"]["success"] += 1
        except Exception as exc:
            msg = str(exc)
            tqdm.write(f"  ✗ {article}: FAILED — {msg}")
            stats["Wikipedia"]["failed"] += 1
            stats["Wikipedia"]["failures"].append((out_path.name, msg))

        time.sleep(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 3 — Academic (arXiv Papers)
# ═══════════════════════════════════════════════════════════════════════════════
ARXIV_PAPERS = [
    {"id": "2303.08774", "name": "gpt4_technical_report",     "field": "AI"},
    {"id": "1706.03762", "name": "attention_is_all_you_need", "field": "ML"},
    {"id": "2103.00020", "name": "clip_paper",                "field": "Vision"},
    {"id": "1810.04805", "name": "bert_paper",                "field": "NLP"},
    {"id": "2005.14165", "name": "gpt3_paper",                "field": "LLM"},
]

ARXIV_HEADERS = {"User-Agent": "RAG-Benchmark-Research student@university.com"}


def download_academic():
    print("\n" + "═" * 45)
    print("  DOWNLOADING: ACADEMIC (arXiv Papers)")
    print("═" * 45)

    for paper in tqdm(ARXIV_PAPERS, desc="Academic", unit="paper"):
        out_path = DIRS["academic"] / f"{paper['name']}.pdf"

        if already_valid(out_path):
            tqdm.write(f"  ✓ {paper['name']}: already downloaded, skipping")
            stats["Academic"]["success"] += 1
            continue

        tqdm.write(f"  → {paper['name']} [{paper['field']}] ({paper['id']})…")
        try:
            url = f"https://arxiv.org/pdf/{paper['id']}.pdf"
            pdf_data = fetch(url, ARXIV_HEADERS, timeout=120)
            out_path.write_bytes(pdf_data)
            tqdm.write(f"  ✓ {paper['name']}: saved ({len(pdf_data):,} B)")
            stats["Academic"]["success"] += 1
        except Exception as exc:
            msg = str(exc)
            tqdm.write(f"  ✗ {paper['name']}: FAILED — {msg}")
            stats["Academic"]["failed"] += 1
            stats["Academic"]["failures"].append((out_path.name, msg))

        time.sleep(3)


# ═══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 4 — Legal (Public Domain Documents)
# ═══════════════════════════════════════════════════════════════════════════════
LEGAL_DOCS = [
    {
        "name": "us_constitution",
        "url": "https://www.govinfo.gov/content/pkg/GPO-CONAN-2002/pdf/GPO-CONAN-2002.pdf",
        "save_as": "us_constitution.pdf",
    },
    {
        "name": "gdpr_regulation",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679",
        "save_as": "gdpr_regulation.pdf",
    },
    {
        "name": "ftc_act",
        "url": (
            "https://www.ftc.gov/sites/default/files/documents/statutes/"
            "federal-trade-commission-act/"
            "ftc_act_incorporatingus_safe_web_act.pdf"
        ),
        "save_as": "ftc_act.pdf",
    },
    {
        "name": "paris_agreement",
        "url": "https://unfccc.int/sites/default/files/english_paris_agreement.pdf",
        "save_as": "paris_agreement.pdf",
    },
    {
        "name": "dodd_frank_act",
        "url": "https://www.govinfo.gov/content/pkg/PLAW-111publ203/pdf/PLAW-111publ203.pdf",
        "save_as": "dodd_frank_act.pdf",
    },
]

LEGAL_HEADERS = {"User-Agent": "RAG-Benchmark-Research student@university.com"}


def download_legal():
    print("\n" + "═" * 45)
    print("  DOWNLOADING: LEGAL (Public Documents)")
    print("═" * 45)

    for doc in tqdm(LEGAL_DOCS, desc="Legal", unit="doc"):
        out_path = DIRS["legal"] / doc["save_as"]
        txt_path = out_path.with_suffix(".txt")

        if already_valid(out_path) or already_valid(txt_path):
            tqdm.write(f"  ✓ {doc['name']}: already downloaded, skipping")
            stats["Legal"]["success"] += 1
            continue

        tqdm.write(f"  → {doc['name']}: downloading PDF…")
        saved = False
        last_err = ""

        # Primary: PDF
        try:
            pdf_data = fetch(doc["url"], LEGAL_HEADERS, timeout=90)
            out_path.write_bytes(pdf_data)
            tqdm.write(f"  ✓ {doc['name']}: saved {out_path.name} ({len(pdf_data):,} B)")
            stats["Legal"]["success"] += 1
            saved = True
        except Exception as exc:
            last_err = str(exc)
            tqdm.write(f"  ! {doc['name']}: PDF failed ({last_err}), trying HTML fallback…")

        # Fallback: strip HTML from the same path
        if not saved:
            try:
                parsed = urllib.parse.urlparse(doc["url"])
                fallback_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                html = fetch_str(fallback_url, LEGAL_HEADERS, timeout=30)
                text = strip_html(html)[:300_000]
                if len(text) < 5_000:
                    raise RuntimeError(f"HTML fallback too short ({len(text)} chars)")
                txt_path.write_text(text, encoding="utf-8")
                tqdm.write(
                    f"  ✓ {doc['name']}: HTML fallback saved {txt_path.name} ({len(text):,} chars)"
                )
                stats["Legal"]["success"] += 1
                saved = True
            except Exception as exc2:
                final_msg = f"{last_err} | fallback: {exc2}"
                tqdm.write(f"  ✗ {doc['name']}: FAILED — {final_msg}")
                stats["Legal"]["failed"] += 1
                stats["Legal"]["failures"].append((doc["save_as"], final_msg))

        time.sleep(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  CATEGORY 5 — Technical (Manuals & Specs)
# ═══════════════════════════════════════════════════════════════════════════════
TECHNICAL_DOCS = [
    {
        "name": "nist_cybersecurity",
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf",
        "save_as": "nist_sp800_53.pdf",
        "type": "pdf",
    },
    {
        "name": "linux_kernel_docs",
        "url": "https://www.gnu.org/software/make/manual/make.html",
        "save_as": "linux_kernel_docs.txt",
        "type": "html",
    },
    {
        "name": "postgres_manual",
        "url": "https://www.postgresql.org/files/documentation/pdf/15/postgresql-15-A4.pdf",
        "save_as": "postgresql_manual.pdf",
        "type": "pdf",
    },
    {
        "name": "rfc_http2",
        "url": "https://www.rfc-editor.org/rfc/rfc7540.txt",
        "save_as": "rfc_http2.txt",
        "type": "rawtext",
    },
    {
        "name": "docker_docs",
        "url": "https://docs.docker.com/get-started/overview/",
        "save_as": "docker_overview.txt",
        "type": "html",
    },
]

TECH_HEADERS = {"User-Agent": "RAG-Benchmark-Research student@university.com"}


def download_technical():
    print("\n" + "═" * 45)
    print("  DOWNLOADING: TECHNICAL (Manuals & Specs)")
    print("═" * 45)

    for doc in tqdm(TECHNICAL_DOCS, desc="Technical", unit="doc"):
        out_path = DIRS["technical"] / doc["save_as"]
        zip_tmp = DIRS["technical"] / f"_temp_{doc['name']}.zip"

        if already_valid(out_path):
            tqdm.write(f"  ✓ {doc['name']}: already downloaded, skipping")
            stats["Technical"]["success"] += 1
            continue

        tqdm.write(f"  → {doc['name']}: downloading ({doc['type']})…")
        try:
            if doc["type"] == "pdf":
                data = fetch(doc["url"], TECH_HEADERS, timeout=180)
                out_path.write_bytes(data)
                tqdm.write(f"  ✓ {doc['name']}: saved {out_path.name} ({len(data):,} B)")
                stats["Technical"]["success"] += 1

            elif doc["type"] == "html":
                html = fetch_str(doc["url"], TECH_HEADERS, timeout=30)
                text = strip_html(html)[:300_000]
                out_path.write_text(text, encoding="utf-8")
                tqdm.write(f"  ✓ {doc['name']}: saved {out_path.name} ({len(text):,} chars)")
                stats["Technical"]["success"] += 1

            elif doc["type"] == "rawtext":
                text = fetch_str(doc["url"], TECH_HEADERS, timeout=60)[:300_000]
                out_path.write_text(text, encoding="utf-8")
                tqdm.write(f"  ✓ {doc['name']}: saved {out_path.name} ({len(text):,} chars)")
                stats["Technical"]["success"] += 1

            elif doc["type"] == "zip":
                zip_data = fetch(doc["url"], TECH_HEADERS, timeout=300)
                zip_tmp.write_bytes(zip_data)
                tqdm.write(f"  → {doc['name']}: extracting from ZIP ({len(zip_data):,} B)…")

                extract_target = doc.get("extract_file", "").lower()
                with zipfile.ZipFile(zip_tmp, "r") as zf:
                    members = zf.namelist()
                    # 1) exact suffix match  2) partial name match  3) first PDF
                    found = None
                    for m in members:
                        if m.lower().endswith(extract_target):
                            found = m
                            break
                    if not found:
                        stem = extract_target.replace(".pdf", "")
                        for m in members:
                            if stem in m.lower():
                                found = m
                                break
                    if not found:
                        for m in members:
                            if m.lower().endswith(".pdf"):
                                found = m
                                break
                    if not found:
                        raise RuntimeError(
                            f"No matching PDF in ZIP. Members: {members[:10]}"
                        )
                    pdf_bytes = zf.read(found)
                    out_path.write_bytes(pdf_bytes)
                    tqdm.write(
                        f"  ✓ {doc['name']}: extracted '{found}' ({len(pdf_bytes):,} B)"
                    )
                    stats["Technical"]["success"] += 1

                zip_tmp.unlink()

        except Exception as exc:
            msg = str(exc)
            tqdm.write(f"  ✗ {doc['name']}: FAILED — {msg}")
            stats["Technical"]["failed"] += 1
            stats["Technical"]["failures"].append((doc["save_as"], msg))
            if zip_tmp.exists():
                try:
                    zip_tmp.unlink()
                except Exception:
                    pass

        time.sleep(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  POST-DOWNLOAD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
def validate_all():
    print("\n" + "═" * 45)
    print("  POST-DOWNLOAD VALIDATION")
    print("═" * 45)

    any_failed = False
    for folder_path in DIRS.values():
        if not folder_path.exists():
            continue
        for fpath in sorted(folder_path.iterdir()):
            if not fpath.is_file() or fpath.name.startswith("_temp"):
                continue
            ok, reason = validate(fpath)
            if not ok:
                print(f"  ⚠️  {fpath.name} failed validation — {reason}")
                any_failed = True

    if not any_failed:
        print("  All downloaded files passed validation.")


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
def print_summary():
    total_size = sum(
        f.stat().st_size
        for d in DIRS.values()
        if d.exists()
        for f in d.iterdir()
        if f.is_file() and not f.name.startswith("_temp")
    )
    total_mb = total_size / (1024 * 1024)

    print("\n")
    print("  ┌────────────────┬───────┬─────────┬────────┐")
    print("  │ Category       │ Total │ Success │ Failed │")
    print("  ├────────────────┼───────┼─────────┼────────┤")

    grand_total = grand_success = grand_failed = 0
    for cat, s in stats.items():
        t, su, fa = s["total"], s["success"], s["failed"]
        grand_total += t
        grand_success += su
        grand_failed += fa
        print(f"  │ {cat:<14} │   {t}   │    {su:<4} │  {fa:<4}  │")

    print("  ├────────────────┼───────┼─────────┼────────┤")
    print(f"  │ {'TOTAL':<14} │  {grand_total}   │    {grand_success:<4} │  {grand_failed:<4}  │")
    print("  └────────────────┴───────┴─────────┴────────┘")
    print(f"\n  Files saved to: {DIVERSE_DOCS}")
    print(f"  Total size: {total_mb:.2f} MB")

    all_failures = [
        (fname, reason)
        for s in stats.values()
        for fname, reason in s["failures"]
    ]
    if all_failures:
        print("\n  Failed downloads:")
        for fname, reason in all_failures:
            print(f"    - {fname}: {reason}")
    else:
        print("\n  All downloads succeeded!")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 45)
    print("  DIVERSE DOCS DOWNLOADER")
    print("  25 documents  ·  5 categories")
    print("=" * 45)

    make_dirs()

    download_financial()
    download_wikipedia()
    download_academic()
    download_legal()
    download_technical()

    validate_all()
    print_summary()


if __name__ == "__main__":
    main()
