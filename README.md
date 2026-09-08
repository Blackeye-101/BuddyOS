# BuddyOS 🤖 — Runbook & Setup Guide

BuddyOS is a model-agnostic, multi-agent AI orchestrator. It acts as a personal "buddy" that remembers facts about you across sessions, routes requests across multiple LLM providers with automatic fallback, grounds answers in your own local documents (RAG), and delegates specialized work (academic research, financial analysis) to dedicated sub-agents.

This document is a practical runbook: what the system does, how to install it in your own workspace, and how to use it day-to-day.

---

## 1. What BuddyOS Does (Functional Overview)

| Capability                                  | Summary                                                                                                                                                                                                                                                                              |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Model-agnostic routing**                  | Uses [LiteLLM](https://github.com/BerriAI/litellm) to discover which models you can use based on the API keys present in your `.env`, and falls back automatically to the next available model if one fails.                                                                         |
| **Hybrid persistence**                      | SQLite (`aiosqlite`) stores conversations/messages transactionally; DuckDB stores "User Facts" and document embeddings for analytical/vector lookups.                                                                                                                                |
| **Continuous learning (fact memory)**       | Buddy silently extracts facts about you (job, preferences, name, etc.) from conversation in the background and stores them as active facts, later injecting only the top-k most relevant facts (via cosine similarity search, `fastembed` + DuckDB VSS/HNSW) into the system prompt. |
| **Context window management**               | Tracks token usage and, once ~75% of the context window is used, automatically compresses the oldest half of the conversation into a dense summary block instead of just warning.                                                                                                    |
| **Local document grounding (Personal RAG)** | Ingest `.txt`, `.md`, `.csv`, `.pdf`, `.docx` files (via `/ingest` in the CLI or the uploader in the UI). Files are chunked, embedded with `fastembed` (`BAAI/bge-small-en-v1.5`, ONNX, no PyTorch), and retrieved into context when your message references a document.             |
| **Plugin-based tool system**                | Tools live as independent, auto-registering modules under `plugins/` (web search via DuckDuckGo, ArXiv search, finance data, agent delegation) and are decorated with `@plugin(...)` in `core/tools.py`'s `ToolRegistry`.                                                            |
| **Multi-agent delegation**                  | A Buddy agent handles general chat, a Researcher agent handles rigorous academic queries (ArXiv-only, zero-hallucination guardrails), and a 4-stage Finance Workflow (Scraper → Processor → Matcher → Validator) produces institutional-style market reports.                        |
| **Streamlit UI**                            | Full graphical chat interface: streaming responses, a "Stop Generating" kill switch, sidebar chat history, a Facts viewer, a document ingestion panel, and buttons to purge chats/facts/documents.                                                                                   |
| **Interactive CLI**                         | A terminal-based chat client with the same core memory/routing engine, useful for headless or quick usage.                                                                                                                                                                           |

---

## 2. Demo Videos

Recorded walkthroughs of the core functionality are available in [`demo_vids/`](demo_vids):

- [`fact_extraction.mp4`](demo_vids/fact_extraction.mp4) — background fact learning and memory recall across turns.
- [`document_ingestion.mp4`](demo_vids/document_ingestion.mp4) — uploading and querying local documents (Personal RAG).
- [`tools_functionality.mp4`](demo_vids/tools_functionality.mp4) — web search, ArXiv research delegation, and finance tool calls.
- [`final_ui_and_purge_functionality.mp4`](demo_vids/final_ui_and_purge_functionality.mp4) — the Streamlit UI, streaming, and the data-purge controls (clear chats/facts/documents).

Watch these first if you want to see expected behavior before running the app yourself.

---

## 3. Tech Stack

- **Language**: Python 3.12+
- **Agent/Routing frameworks**: Pydantic-AI, LiteLLM
- **Databases**: SQLite (`aiosqlite`) for conversations/messages, DuckDB (+ VSS extension, HNSW index) for facts and document chunks
- **Embeddings**: `fastembed` — `BAAI/bge-small-en-v1.5` (384-dim, ONNX Runtime)
- **Tool integrations**: `ddgs` (DuckDuckGo search), ArXiv API, `yfinance` (live stock data)
- **Document parsing**: `pypdf`, `python-docx`
- **Frontend**: Streamlit
- **Package manager**: [`uv`](https://github.com/astral-sh/uv) (an `uv.lock` is committed; `pip` also works)

---

## 4. Cloning & Setting Up Your Own Workspace

### 4.1 Prerequisites

- Python 3.12 or newer
- Git
- At least one LLM API key (e.g. Gemini, OpenAI, Anthropic, Groq, or OpenRouter) — Gemini has a usable free tier and is the default fallback model.

### 4.2 Clone the repository

```bash
git clone <your-fork-or-repo-url>.git
cd buddy-os
```

> The project root (containing `pyproject.toml`, `main.py`, `.gitignore`) is the `buddy-os/` folder — run all commands from inside it.

### 4.3 Create a virtual environment and install dependencies

Using `uv` (recommended, matches the committed `uv.lock`):

```bash
uv sync
```

Or with plain `pip`:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e .
```

### 4.4 Configure API keys

Create a `.env` file in the `buddy-os/` project root (this file is gitignored — never commit it):

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GROQ_API_KEY=your_groq_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

You only need one key configured for BuddyOS to run — it discovers whichever providers are configured at startup and lists their models. Never share or commit real API key values; rotate any key that is accidentally exposed.

### 4.5 Embedding model (for RAG / fact memory)

BuddyOS uses `BAAI/bge-small-en-v1.5` (~66 MB, ONNX) for semantic search. It downloads automatically on first run into `data/fastembed_cache/` (gitignored). If your network blocks the automatic download, fetch it manually:

```bash
python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='data/fastembed_cache')"
```

### 4.6 Data directories

`data/` (SQLite/DuckDB files, ingested documents, embedding cache) and `logs/` are created automatically on first run and are gitignored — no manual setup required.

---

## 5. Running BuddyOS

### Streamlit UI (recommended)

```bash
uv run streamlit run ui/app.py
```

or, if using a plain venv:

```bash
streamlit run ui/app.py
```

This opens the chat UI with the sidebar (chat history), the main chat panel, and a right-hand panel with tabs for **Memory Facts** and **Local Documents**.

### Interactive CLI

```bash
python main.py
```

The CLI supports the following in-chat commands:

- `/help` — show available commands
- `/facts` — view what Buddy currently knows about you
- `/model` — switch the active AI model
- `/new` — start a fresh conversation
- `/history` — view recent conversations
- `/ingest` — map a local file (`.txt`, `.md`, `.pdf`, `.csv`, `.docx`) for searching
- `/exit` — save state and exit gracefully

---

## 6. Using BuddyOS

- **General chat**: just type — Buddy remembers facts you share and recalls the most relevant ones automatically each turn.
- **Web search / recent info**: ask about current events; Buddy calls the `web_search` tool automatically.
- **Academic research**: ask for peer-reviewed papers or deep academic topics; Buddy delegates to the Researcher agent, which is restricted to ArXiv-sourced, cited answers only.
- **Finance / market analysis**: ask stock or macro questions; the request is routed through the 4-stage Finance Workflow (news scraping → sentiment analysis → live quant data matching → compliance-formatted report with disclaimer).
- **Document grounding**: upload a file via the UI's "Local Documents" tab (or `/ingest` in the CLI), then reference words like "document", "file", or "pdf" in your message to have relevant chunks injected into context.
- **Data control**: in the UI, use "Clear All Chats", "Clear All Facts", or "Clear All Documents" to hard-delete the corresponding data from SQLite/DuckDB.

---

## 7. Project Structure

```
buddy-os/
├── agents/             # Agent entry points (buddy.py re-exports the orchestrator, finance.py = Finance Workflow)
├── core/               # Core engine
│   ├── orchestrator.py   # Main supervisor logic, Buddy/Researcher agents, fact extraction
│   ├── router.py         # LiteLLM-backed router with fallback chain
│   ├── tools.py          # ToolRegistry: plugin discovery + execution with retry/backoff
│   ├── database.py       # SQLite + DuckDB hybrid persistence
│   ├── embeddings.py      # fastembed wrapper for vector generation
│   ├── document_parser.py # PDF/DOCX/TXT/MD/CSV parsing + chunking
│   ├── discovery.py       # API key / model discovery
│   └── fact_utils.py       # Fact normalization/contradiction handling
├── plugins/            # Auto-registered tools (web_search, arxiv_search, finance_tools, delegation tools)
├── ui/                  # Streamlit UI (app.py + components/: chat, sidebar, facts_viewer, document_ingestor)
├── data/                # Generated at runtime: SQLite/DuckDB files, ingested documents, embedding cache (gitignored)
├── logs/                # Rotating log files (gitignored)
├── demo_vids/           # Recorded feature walkthroughs (see section 2)
├── main.py              # CLI entry point
└── pyproject.toml       # Project metadata & dependencies
```

---

## 8. Troubleshooting

- **"No API keys configured" on startup**: ensure `.env` exists in `buddy-os/` with at least one valid, non-placeholder key.
- **Embedding model fails to download**: run the manual `fastembed` download command in section 4.5, or check outbound network access to Hugging Face.
- **DuckDB VSS extension unavailable**: BuddyOS automatically falls back to a full fact scan (slower semantic search) instead of the HNSW index; functionality is unaffected.
- **Logs**: check `logs/buddy-os.log` for detailed DEBUG-level output; the console only shows WARNING+.
