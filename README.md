# Maritime Safety RAG

A Retrieval-Augmented Generation application for maritime safety. Ask natural language questions about maritime regulations and accident investigations — get accurate, source-cited answers powered by a local LLM.

The system ingests maritime safety documents (SOLAS/MARPOL conventions, accident investigation reports, text files), stores them as vector embeddings, and retrieves relevant context to ground LLM responses in real source material.

## Why RAG over a General LLM?

General-purpose LLMs (GPT-4, Claude, etc.) know *about* maritime safety but:
- Hallucinate regulation numbers that don't exist
- Can't cite exact source text
- Have knowledge cutoffs — miss recent amendments and reports
- Mix up conventions (SOLAS vs MARPOL vs STCW)

This RAG system answers from **actual documents** and shows you exactly where each answer comes from.

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Llama 3 (8B) via [Ollama](https://ollama.com) (local, private) |
| Embeddings | all-MiniLM-L6-v2 via sentence-transformers (384 dims) |
| Vector DB | ChromaDB (embedded, persistent) |
| Framework | LangChain modular packages |
| UI | Streamlit (chat interface + document upload) |
| Language | Python 3.10+ |

## Project Structure

```
RAG-Maritime-Safety/
├── app.py                      # Streamlit entry point
├── config.py                   # Central configuration (models, chunking, retrieval)
├── requirements.txt
│
├── ingestion/
│   ├── loader.py               # PDF, web, and text document loaders
│   ├── chunker.py              # Text splitting with RecursiveCharacterTextSplitter
│   └── embedder.py             # Embedding wrapper + ChromaDB storage
│
├── retrieval/
│   ├── retriever.py            # Query ChromaDB for relevant chunks
│   └── rag_chain.py            # Prompt template + Ollama LLM call + source formatting
│
├── ui/
│   ├── chat.py                 # Chat interface components
│   └── sidebar.py              # Document upload & management sidebar
│
├── scripts/
│   └── scrape_reports.py       # Download MAIB/NTSB/TSB accident reports
│
├── data/
│   ├── sample/                 # Sample maritime docs
│   ├── maib/                   # UK accident investigation reports
│   ├── ntsb/                   # US accident investigation reports
│   └── tsb/                    # Canadian accident investigation reports
│
├── chroma_db/                  # Vector store (auto-created, gitignored)
│
└── tests/
    ├── test_loader.py
    ├── test_chunker.py
    ├── test_retriever.py
    └── test_rag_chain.py
```

## Prerequisites

1. **Python 3.10+**

2. **Ollama** — install from [ollama.com](https://ollama.com), then pull the model:
   ```bash
   ollama pull llama3
   ```
   Verify Ollama is running:
   ```bash
   ollama list
   ```
   If not running, start it with `ollama serve` (on Windows it typically runs as a background service automatically).

## Setup

```bash
# Clone the repository
git clone https://github.com/your-username/RAG-Maritime-Safety.git
cd RAG-Maritime-Safety

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

## Getting the Data

### Scrape Accident Investigation Reports

The included scraper downloads public accident investigation reports from three sources:

| Source | Description | Reports |
|---|---|---|
| **MAIB** | UK Marine Accident Investigation Branch | ~1,089 |
| **NTSB** | US National Transportation Safety Board | ~100+ |
| **TSB** | Transportation Safety Board of Canada | ~562 |

```bash
# Download from all sources (takes a while)
python scripts/scrape_reports.py

# Download from a single source
python scripts/scrape_reports.py --source maib

# Limit downloads for testing
python scripts/scrape_reports.py --source maib --max 20

# Combine options
python scripts/scrape_reports.py --source tsb --max 50
```

Reports are saved as PDFs (or text fallback) to `data/maib/`, `data/ntsb/`, and `data/tsb/`. The scraper skips already-downloaded files, so you can safely re-run it.

### Additional Documents

For the strongest results, supplement with official regulatory texts:

- **SOLAS** — Safety of Life at Sea (ship construction, fire safety, life-saving)
- **MARPOL** — Pollution prevention (oil, chemicals, sewage, emissions)
- **STCW** — Seafarer training and certification
- **ISM Code** — Safety management systems
- **COLREG** — Collision prevention rules

These are copyrighted IMO publications available from the [IMO bookstore](https://www.imo.org/en/publications). University and maritime academy libraries typically have access.

## Usage

### Run the Application

```bash
streamlit run app.py
```

This opens a web interface where you can:

1. **Upload documents** — Use the sidebar to upload PDF, TXT, or MD files, or enter a web URL to ingest
2. **Ask questions** — Type natural language questions in the chat interface
3. **View sources** — Expand the "Sources" section under each answer to see which documents and pages were used

### Example Questions

- "What are the fire detection requirements under SOLAS Chapter II-2?"
- "What caused the capsize of the tug Biter?"
- "What are the requirements for means of escape on passenger ships?"
- "What does MARPOL Annex VI say about SOx emissions?"

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3` | Ollama model name |
| `LLM_TEMPERATURE` | `0.1` | Low for factual accuracy |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `EMBEDDING_DIMENSION` | `384` | Embedding vector size |
| `CHUNK_SIZE` | `1000` | Characters per chunk (~1 regulation paragraph) |
| `CHUNK_OVERLAP` | `200` | Overlap to preserve cross-references |
| `RETRIEVAL_K` | `5` | Number of chunks retrieved per query |

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Document loading (PDF, text, URL dispatch)
- Text chunking (size limits, metadata preservation)
- Embedding shape and similarity
- Ingestion and retrieval round-trip
- RAG chain context formatting and source extraction

## Architecture

```
User Question
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Streamlit   │────▶│  Retriever   │────▶│  ChromaDB   │
│  Chat UI     │     │  (top-k)     │     │  (vectors)  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    retrieved chunks
                           │
                           ▼
                    ┌──────────────┐     ┌─────────────┐
                    │  RAG Chain   │────▶│   Ollama     │
                    │  (prompt +   │     │  (Llama 3)   │
                    │   context)   │     └─────────────┘
                    └──────┬───────┘
                           │
                    answer + sources
                           │
                           ▼
                      User sees
                   cited response
```

**Ingestion flow:** Documents → Loader (PDF/Web/Text) → Chunker (1000 chars, 200 overlap) → Embedder (all-MiniLM-L6-v2) → ChromaDB (persistent storage with deduplication)

**Query flow:** Question → Embed query → Similarity search (top 5) → Format context with source labels → LLM generates grounded answer → Display with expandable sources

## License

This project is for educational and research purposes. Accident investigation reports are public government documents. IMO convention texts are copyrighted and must be obtained through authorized channels.
