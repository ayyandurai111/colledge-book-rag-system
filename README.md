# RAG Backend — Textbook-Specialized

A production-ready Retrieval-Augmented Generation backend specialized for
**college textbooks**, using:

- **NVIDIA NV-Embed-v1** for embeddings (OpenAI-compatible API)
- **ChromaDB** as the vector store with full section metadata
- **FastAPI** for the REST API
- **PyMuPDF** for rich PDF extraction (font sizes, bold flags, TOC)

> Retrieval-only pipeline — no LLM. Plug in your LLM of choice on top.

---

## Smart Chunking Strategy

### Auto-detect PDF type
| PDF has bookmarks/TOC? | Strategy used |
|---|---|
| ✅ Yes (≥ 3 entries) | **TOC-based** — exact section boundaries from outline |
| ❌ No | **Font-based** — heading hierarchy inferred from font size + bold |

### Chunking rules
- Split by hierarchy: **Chapter → Section → Subsection**
- Target window: **400–600 tokens** with **75-token overlap**
- Never split mid-sentence (sentence-boundary aware)
- Special content detection: `definition`, `formula`, `table`, `figure`

### Chunk metadata (stored in ChromaDB)
```json
{
  "chunk_id":     "physics.pdf_toc_00042",
  "source":       "physics.pdf",
  "chapter":      "Chapter 3 — Dynamics",
  "section":      "Newton's Laws",
  "subsection":   "Second Law",
  "page_number":  42,
  "heading_path": "Chapter 3 — Dynamics > Newton's Laws > Second Law",
  "chunk_type":   "formula",
  "token_count":  134,
  "text":         "Chapter 3 — Dynamics > Newton's Laws > Second Law: F = ma ..."
}
```

### Quality gates
Every chunk must pass before being stored:
- ✅ Minimum 100 tokens
- ✅ Maximum 700 tokens
- ✅ Complete sentences (no broken start/end)
- ✅ Meaningful content (not just a heading)

---

## Context Expansion (full-section retrieval)

Chunks are stored small (400–600 tokens) so **similarity search stays sharp**.
But an LLM answering a question usually needs the **whole section**, not one
window of it. Setting `expand_context: true` on `/query` reconstructs the
full section a hit came from — directly from ChromaDB, no re-parsing the PDF:

1. Similarity search returns the top chunk as usual.
2. Its `heading_path` and `chunk_index` (document position) are read.
3. Every chunk for that `source` is pulled from ChromaDB and sorted by
   `chunk_index`.
4. Starting from the hit, we walk backward and forward through that ordered
   list **while `heading_path` stays identical**. The moment it changes,
   that's the next-topic boundary — detected straight from stored metadata,
   not re-derived.
5. The contiguous run is merged into one block: the duplicate words created
   by the ingest-time sentence overlap (`overlap_tokens`) are stripped, and
   the repeated `heading_path: ` prefix is kept once instead of once per
   window.
6. A `MAX_EXPANDED_TOKENS` (3000) cap prevents a single very long section
   (e.g. a whole UNIT with no subsections) from blowing up LLM context.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Motivation in HRM?", "expand_context": true}'
```

Response shape differs slightly from a raw chunk — `page_number` becomes
`pages` (every page the merged section spans) and `merged_chunk_count`
reports how many stored windows were combined:
```json
{
  "text": "UNIT - IV > What is Motivation: Motivation is defined as ...",
  "source": "Human Resource Management.pdf",
  "heading_path": "UNIT - IV > What is Motivation",
  "pages": [71, 72],
  "merged_chunk_count": 2,
  "score": 0.7441
}
```

Implemented in `src/retrieval/context_expander.py`; wired through
`retriever.retrieve_expanded()` → `rag_pipeline.query(expand_context=True)`.

---

## Project Structure

```
rag-backend/
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py        # Rich PDF extraction (font, bold, TOC)
│   │   ├── pdf_type_detector.py # Auto-detect TOC vs font strategy
│   │   ├── chunker.py           # Smart textbook chunker (both strategies)
│   │   ├── embedder.py          # NVIDIA NV-Embed-v1 embedding
│   │   └── indexer.py           # ChromaDB indexing with metadata
│   ├── retrieval/
│   │   ├── retriever.py         # Query + filter + type-specific retrieval
│   │   └── context_expander.py  # Merge a hit into its full section
│   ├── pipeline/
│   │   ├── ingest_pipeline.py   # End-to-end ingest orchestration
│   │   └── rag_pipeline.py      # End-to-end query orchestration
│   ├── vectorstore/
│   │   ├── base.py              # Abstract base class
│   │   └── chroma_store.py      # ChromaDB with metadata filters
│   └── utils/
│       ├── config.py            # YAML + .env config loader
│       └── logger.py            # Structured logging
├── api/
│   ├── main.py                  # FastAPI app
│   ├── routes/
│   │   ├── ingest.py            # POST /ingest
│   │   └── query.py             # POST /query
│   └── schemas/
│       ├── request.py           # QueryRequest with filters
│       └── response.py          # ChunkResult with full metadata
├── scripts/
│   ├── ingest.py                # CLI ingestion
│   └── query.py                 # CLI querying with filters
├── tests/
│   └── unit/
│       ├── test_chunker.py
│       ├── test_pdf_loader.py
│       ├── test_retriever.py
│       └── test_context_expander.py
├── configs/default.yaml
├── requirements.txt
└── .env.example
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your NVIDIA_API_KEY to .env
uvicorn api.main:app --reload --port 8000
```

---

## API Usage

### Ingest a textbook PDF
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/raw/physics_textbook.pdf"
```

Response:
```json
{
  "message": "Successfully ingested 'physics_textbook.pdf'",
  "pdf_type": "toc",
  "pages_loaded": 512,
  "chunks_created": 843,
  "chunk_types": { "text": 780, "definition": 34, "formula": 22, "figure": 7 },
  "total_in_store": 843
}
```

### Query — general search
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain Newton Second Law", "top_k": 5}'
```

### Query — definitions only
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is entropy?", "content_type": "definition"}'
```

### Query — filter by textbook and chapter
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does photosynthesis work?",
    "source": "biology.pdf",
    "chapter": "Chapter 7"
  }'
```

---

## CLI Usage

```bash
# Ingest
python scripts/ingest.py --file data/raw/calculus.pdf
python scripts/ingest.py --dir data/raw/

# Query
python scripts/query.py --question "What is a derivative?"
python scripts/query.py --question "Define integral" --type definition
python scripts/query.py --question "Force formula" --type formula --source physics.pdf
```

---

## Configuration (`configs/default.yaml`)

```yaml
chunking:
  min_tokens: 400         # Target minimum per chunk
  max_tokens: 600         # Target maximum per chunk
  overlap_tokens: 75      # Overlap between adjacent chunks
  quality_min_tokens: 100 # Hard minimum (chunk dropped if below)
  quality_max_tokens: 700 # Hard maximum (chunk dropped if above)
```

---

## Run Tests

```bash
pytest tests/ -v
```
