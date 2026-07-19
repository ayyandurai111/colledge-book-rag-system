---
title: College RAG
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.37.1"
app_file: apps/streamlit_app.py
pinned: false
license: mit
short_description: Pure-retrieval semantic-chunking RAG chatbot for college textbooks (no LLM)
---

# 📚 College RAG — Pure-Retrieval RAG Chatbot for College Textbooks

Semantic Chunking + FAISS Vector Search, wrapped in a chatbot-style
Streamlit UI. Upload PDF/DOCX textbooks, ask questions in a chat window,
and get the most relevant full passages back — instantly.

**This is retrieval-only — no LLM/Claude/OpenAI dependency.** No API key
is required, no `anthropic` package is installed, and answers are never
generated — every response is the actual, unmodified text pulled from
your books.

74 automated tests, 89%+ code coverage — tests pass even with `anthropic`
completely uninstalled (proof of zero LLM dependency).

## 🏗️ Architecture

```
college_rag/
├── src/college_rag/          ← installable Python package (pip install -e .)
│   ├── models.py               — Shared dataclasses: TextBlock, Chunk, SearchResult, IndexStats
│   ├── config.py                — Centralized, env-var-overridable configuration
│   ├── exceptions.py            — Custom exception hierarchy
│   ├── pipeline.py              — RAGPipeline: orchestrates every layer below
│   ├── ingestion/                — PDF/DOCX → TextBlock extraction
│   │   ├── base.py               — Abstract BaseExtractor interface
│   │   ├── heading_utils.py      — Shared chapter/heading-detection heuristics
│   │   ├── pdf_extractor.py      — pypdf-based extractor
│   │   ├── docx_extractor.py     — python-docx-based extractor
│   │   └── factory.py            — Routes to the right extractor by extension
│   ├── chunking/
│   │   └── semantic_chunker.py  — Sentence-embedding-similarity-based chunking
│   ├── embeddings/
│   │   └── embedder.py          — Lazy-loaded sentence-transformers wrapper
│   ├── vectorstore/
│   │   └── faiss_store.py       — FAISS index: build / search / save / load
│   ├── retrieval/
│   │   └── retriever.py         — Convenience layer + sentence-level highlight (Python-API-only)
│   └── utils/
│       └── logging_config.py    — Centralized logging setup
│
├── apps/
│   └── streamlit_app.py        — Chatbot-style web UI (see below)
│
├── tests/                     ← 74 tests, offline (no model download needed)
│   ├── conftest.py             — Shared fixtures: FakeSentenceTransformer + inline sample-file generators
│   ├── test_models.py
│   ├── test_config.py
│   ├── test_ingestion.py
│   ├── test_chunking.py
│   ├── test_embedder.py
│   ├── test_vectorstore.py
│   ├── test_retriever.py
│   └── test_pipeline.py        — Full end-to-end integration tests
│
├── data/
│   ├── uploads/                 — scratch space for uploaded books (gitignored)
│   └── indexes/                 — saved FAISS indexes (gitignored)
│
├── pyproject.toml              — pip-installable package config + pytest config
├── requirements.txt             — no `anthropic`, no CLI-only deps
├── requirements-dev.txt
├── .env.example
└── .gitignore
```

### Data flow

```
PDF/DOCX  →  ingestion  →  TextBlock[]  →  chunking  →  Chunk[]
                                                            │
                                                            ▼
                                                  embeddings + vectorstore
                                                     (FAISS index)
                                                            │
   question ────────────► retrieval ─────────────────────► SearchResult[]
   (chat input)                                            │
                                                            ▼
                                              full chunk.text + similarity
                                               score shown as a chat reply
                                               — no LLM call anywhere
```

## ⚙️ Installation

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# for development/testing:
pip install -r requirements-dev.txt

# or, as an installable package:
pip install -e ".[ui,dev]"
```

> On first use, the `sentence-transformers` model
> (`paraphrase-multilingual-MiniLM-L12-v2`, ~400MB) auto-downloads. An
> internet connection is required. No API key is needed beyond that.

## 🚀 Running the app

```bash
streamlit run apps/streamlit_app.py
```

### UI layout

- **Left — Chat**: a chatbot-style conversation window. Ask a question,
  get the most relevant full book passages back as a chat reply, with
  source (file / chapter / page) and similarity score for each.
- **Right — Upload**: drag in PDF/DOCX files, tune chunking settings,
  and click **🔨 Build index**. Index stats (chunk count, source files)
  are shown once built.
- **🪵 Live logger**: a toggle on the right panel that streams backend
  log messages (text extraction, chunking, indexing progress) into a
  live-updating panel as they happen — useful for watching large-book
  indexing progress in real time.

### Python API

```python
from college_rag.pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.build_index_from_files(["physics.pdf", "biology.docx"])
pipeline.save_index("./data/indexes/my_index")

results = pipeline.query("What is entropy?")   # List[SearchResult] — no LLM
for r in results:
    print(f"{r.score:.3f}  {r.chunk.location_label()}")
    print(r.chunk.text)                # full chunk (what the UI shows)
    # r.highlight -> optional sentence-level snippet, for advanced use only
```

## 🧪 Testing & Results

```bash
pip install -r requirements-dev.txt
pytest tests/ -v                                                # 74 tests
pytest tests/ --cov=src/college_rag --cov-report=term-missing   # coverage report
```

**Latest test run:**
```
74 passed in 0.84s
```

This suite was run with `anthropic` **completely uninstalled**, proving
zero LLM dependency:
```bash
pip uninstall anthropic
python -c "import anthropic"   # ModuleNotFoundError — package not present
pytest tests/ -v                # 74 passed ✅
```

Tests **have no network dependency** — `conftest.py` provides a
deterministic, topic-aware `FakeSentenceTransformer` via dependency
injection, so CI can validate semantic chunking and retrieval logic
without downloading any model. Sample PDF/DOCX test fixtures are
generated inline in `conftest.py` (no external scripts directory).

What's covered:
- PDF/DOCX extraction (heading detection, corrupt files, missing files)
- Semantic chunking (topic-boundary splitting, size limits, edge cases)
- Sentence-level highlight extraction (Python-API-only feature — correct
  sentence selection, original-order preservation, short-chunk fallback)
- FAISS build/search/save/load (roundtrip correctness, empty/invalid states)
- Full end-to-end pipeline (multi-file indexing, partial failures,
  persistence, blank query)
- The Streamlit app was smoke-tested headlessly (`streamlit run --server.headless`)
  and confirmed to serve HTTP 200 with no startup exceptions

### End-to-end example (real sample PDF/DOCX)

```
Index built: 7 chunks

Q: "What is entropy?"
[1] score=0.999  physics.docx › Chapter 2: Thermodynamics
    The Second Law of Thermodynamics states that the total entropy of an
    isolated system can never decrease over time. Entropy is a measure of
    disorder in a system.

[2] score=0.995  physics.docx › Chapter 2: Thermodynamics
    Heat naturally flows from hotter objects to colder objects, never the
    reverse, without external work being applied.
```

This is exactly what appears as a chat reply in the UI — no LLM involved.

## 🧠 How semantic chunking works

`chunking/semantic_chunker.py`:
1. Splits each paragraph into sentences (handles both Tamil `।` and
   English `. ! ?` sentence terminators)
2. Computes an embedding for every sentence
3. Computes cosine similarity between consecutive sentences
4. Starts a new chunk wherever similarity **drops sharply** (a topic switch)
5. Respects `min_chunk_chars` / `max_chunk_chars` limits

This keeps a definition/theorem/concept together in a single chunk more
often, improving retrieval accuracy.

## 🔧 Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `COLLEGE_RAG_EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding model name |
| `COLLEGE_RAG_MIN_CHUNK_CHARS` | 200 | Minimum chunk size |
| `COLLEGE_RAG_MAX_CHUNK_CHARS` | 1200 | Maximum chunk size |
| `COLLEGE_RAG_BREAKPOINT_PERCENTILE` | 75 | Topic-switch sensitivity |
| `COLLEGE_RAG_TOP_K` | 5 | Default retrieval count |
| `COLLEGE_RAG_HIGHLIGHT_SENTENCES` | 2 | (Python-API-only) sentences for `SearchResult.highlight` — not shown by the UI |

See `.env.example` for the full list — no LLM API key is needed.

## ☁️ Free deployment (Hugging Face Spaces)

`sentence-transformers` + `torch` + `faiss` still need ~2GB RAM. Hugging
Face Spaces' free CPU Basic tier (2 vCPU, **16GB RAM**, 50GB disk, fully
free) fits comfortably.

1. Create a free account at **[huggingface.co](https://huggingface.co)**.
2. **New Space** → SDK: **Streamlit** → Hardware: "CPU basic · FREE".
3. Push this repo to the Space:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<space-name>
   cd <space-name>
   cp -r /path/to/college_rag/* .
   git add .
   git commit -m "Deploy college RAG chatbot (retrieval-only, no LLM)"
   git push
   ```
   (The YAML frontmatter at the top of this README already points the
   Space at `apps/streamlit_app.py` via the Streamlit SDK — no extra
   config, no API key secrets needed.)
4. The Space builds automatically and goes live within a few minutes at
   `https://huggingface.co/spaces/<user>/<space-name>`.

**Good to know:**
- The free CPU Space sleeps after 48 hours of inactivity (cold start
  ~30-60s on the next visit).
- Storage is ephemeral — re-upload books and rebuild the index after a restart.
- The embedding model (~400MB) auto-downloads on first startup.

## ⚠️ Limitations

- **This is not an LLM answer generator** — it doesn't produce a
  sentence-form "answer"; it returns the most relevant full chunks. You
  read and interpret the passages yourself.
- Scanned/image-only PDFs won't extract text (would require OCR, not included)
- Table/formula-heavy pages may extract with lower quality
- Heading detection is a heuristic — not 100% accurate

## 🔮 Extending

- **Re-add an LLM**: create a new `generation/` module (with any LLM
  provider) and add an optional generation step after `query()` in
  `pipeline.py`. The current `SearchResult` contract is a clean input for
  any LLM layer.
- **OCR support**: add a new extractor in `ingestion/` using
  `pytesseract` + `pdf2image` (extend `BaseExtractor`)
- **A different vector DB**: use `vectorstore/faiss_store.py` as a
  template to write a Chroma/Pinecone/Qdrant backend (implement the same
  public interface as `FaissVectorStore`)
- **New file types**: extend `ingestion/base.py`'s `BaseExtractor` and
  register it in `ingestion/factory.py`

## 📄 License

MIT
