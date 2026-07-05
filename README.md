# BuddyOS 🤖

![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![Version](https://img.shields.io/badge/version-0.1.0-green.svg)

BuddyOS is a highly flexible, model-agnostic AI assistant and orchestrator. It is designed to act as a personal "buddy" that continuously learns about you, maintains context across sessions, and dynamically routes your queries to the best available AI models.

## ✨ Key Features

- **Robust Tool Orchestration**: Designed with `max_steps` loop constraints and explicit null-type handling to gracefully prevent infinite tool-calling loops, especially when using complex models like GPT-5.
- **Model-Agnostic Routing**: Powered by [LiteLLM](https://github.com/BerriAI/litellm), BuddyOS can discover available models dynamically based on your environment keys. It supports automated token counting and a **graceful fallback chain** (if one model goes down, it switches to the next available).
- **Hybrid Persistence Layer**:
  - **SQLite** (`aiosqlite`): Transactional tracking of conversations, message history, timestamp updates, and token usage limits.
  - **DuckDB**: Analytical engine for lightning-fast retrieval of learned "User Facts".
- **RAG-Powered Memory (Semantic Fact Retrieval)**: Buddy embeds every learned fact locally using `fastembed` (ONNX Runtime, no PyTorch required). At each turn, it runs a DuckDB VSS cosine-similarity search to inject only the top-5 most relevant facts into the system prompt — keeping context lean and precise.
- **Continuous Learning (Automated Fact Extraction)**: Buddy constantly evaluates your conversations in the background. It extracts information about you (e.g., job, preferences, name) and stores them as active facts to customize future system prompts.
- **Dynamic Context Window Management**: Constantly monitors context tokens and triggers summarization when the context threshold (~75%) is reached, preventing the LLM from forgetting the start of a long conversation.
- **Real-Time Web Search & Tool Calling**: Buddy is equipped with an integrated web search tool using DuckDuckGo (`ddgs`) and an academic search tool using ArXiv (with advanced XML parsing and robust network timeouts). When asked about recent events or complex academic topics, it dynamically pauses the conversation, searches the web or literature, and integrates the live results into its final answer seamlessly.
- **Advanced Multi-Agent Finance Workflow (Trading Desk)**: A specialized 4-stage analytical pipeline designed for institutional-grade stock and market analysis. It features a secure smart-routing regex/LLM cascade to flawlessly differentiate between standard academic/general inquiries and real-time financial scopes. The pipeline operates sub-agents sequentially:
  - **Scraper**: Gathers live macroeconomic and sector-specific news using integrated search tools.
  - **Processor**: Evaluates the raw data to extract core macroeconomic sentiment (Bullish/Bearish/Neutral).
  - **Matcher**: Fuses live quantitative stock ticker data with the qualitative macro sentiment.
  - **Validator**: Acts as a strict compliance officer to format the fused data into an institutional-grade Markdown report with necessary financial disclaimers.
- **Local Document Grounding (Personal RAG)**: Native support for ingesting and querying your local files! You can parse `.txt`, `.md`, `.csv`, `.pdf`, and `.docx` files directly via the UI's document uploader. The text is chunked (512 tokens with overlap) and embedded directly into DuckDB using `fastembed`. To search it, simply include words like "document", "pdf", or "file" in your chat message, and Buddy will dynamically inject the relevant chunks straight into the context.
- **Interactive Streamlit UI**: A modern, feature-rich chat interface that provides:
  - **Dynamic Chat Management**: Create new chats, see auto-generated concise summaries for previous chats in the sidebar, and switch between them effortlessly.
  - **Scrollable Layouts**: Ergonomic, fixed-height bounded scroll areas ensuring the input bar stays pinned to the viewport bottom (just like your favorite commercial tools).
  - **Data Privacy & Control**: Explicit UI buttons to obliterate and hard-delete all conversations (`Clear All Chats`), facts (`Clear All Facts`), and ingested documents (`Clear All Documents`) directly from SQLite/DuckDB.
  - **Word-by-word Streaming**: Real-time asynchronous text streaming over the LiteLLM router.
  - **Interruption Mechanisms (Kill Switch)**: An inline `🛑 Stop Generating` button allows you to sever the model connection midway, abandoning incomplete queries without polluting your context history or DB.
- **Interactive CLI**: Optionally comes with an interactive terminal interface equipped with commands (`/facts`, `/history`, `/model`, `/new`, `/ingest`) to manage your Buddy context easily headless.

## 🛠 Tech Stack

- **Language**: Python 3.12+
- **Frameworks**: Pydantic-AI (Agent logic), LiteLLM (Routing/Tokenization)
- **Tools Integrations**: `ddgs` (DuckDuckGo Search)
- **Database**: SQLite (`aiosqlite`), DuckDB + VSS extension (HNSW vector index)
- **Embeddings**: [`fastembed`](https://github.com/qdrant/fastembed) — `BAAI/bge-small-en-v1.5` (384-dim, ONNX Runtime, no PyTorch)
- **Frontend**: Streamlit-based graphical user interface (`uv run streamlit run ui/app.py`).

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or newer
- An API key for at least one supported LLM (e.g., `GEMINI_API_KEY`, `OPENAI_API_KEY`).

### Installation

1. Clone the repository and navigate to the project root:

   ```bash
   cd buddy-os
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -e .
   ```

4. **Embedding model** (for RAG memory):

   BuddyOS uses [`BAAI/bge-small-en-v1.5`](https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q) (~66 MB, ONNX format) for local semantic search. The model is **downloaded automatically on first run** via `fastembed` and cached to `data/fastembed_cache/`. No manual step is required.

   If automatic download fails (e.g., restricted network), download it manually:

   ```bash
   python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='data/fastembed_cache')"
   ```

   > **Note:** The model cache is excluded from version control (`.gitignore`). Each developer downloads it once on first run.

5. Create a `.env` file in the root directory and add your API keys:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Running BuddyOS

To launch the graphical web UI using Streamlit:

```bash
uv run streamlit run ui/app.py
```

Or run the interactive CLI application:

```bash
python main.py
```

### CLI Commands

Inside the chat loop, you can use the following commands:

- `/help` - Show available commands
- `/facts` - View what Buddy actively knows about you
- `/model` - Switch the current AI model on the fly
- `/new` - Start a fresh conversation
- `/history` - View a list of your recent conversations- `/ingest` - Map a local file for searching (.txt, .md, .pdf, .csv, .docx)- `/exit` - Save state and gracefully exit

## 📂 Project Structure

```
buddy-os/
├── agents/             # Pydantic-AI orchestrator definitions (Buddy agent)
├── core/               # Core engine (LiteLLM router, Hybrid DB manager, embeddings)
│   ├── orchestrator.py   # Main Pydantic-AI orchestrator logic
│   ├── router.py         # Multi-model router with tool & fallback mechanisms
│   ├── tools.py          # Extensible tool registry & duckduckgo web_search implementation
│   └── database.py       # DuckDB & SQLite persistence logic
├── data/               # Local persistence layer (.db and .duckdb generated here)
│   └── fastembed_cache/  # Auto-downloaded ONNX embedding model (gitignored)
├── ui/                 # Streamlit UI logic (WIP)
├── main.py             # CLI Entry point
└── pyproject.toml      # Project definitions & dependencies
```

## 🔄 Recent Changes & Fixes

- **Thread-safe Analytical Persistence**: Implemented strict `asyncio.Lock()` boundaries around all DuckDB interactions. This securely handles concurrent read/writes between the main thread and background LLM fact extraction threads, eliminating pending query lock crashes.
- **UUID-based Key Generation**: Decoupled DuckDB's unique identifier constraints from non-deterministic LLM generation, ensuring stable collision-free database memory inserts.
- **Hardened Fact Contradiction Logic**: Refined the hybrid RAG background extraction bounds to intelligently update mutually exclusive facts (e.g., correcting "Red Corolla" to "Silver Corolla") whilst ensuring independent historical or future plans are not unwarrantedly wiped by aggressive overriding.
- **Local Document Grounding (Personal RAG)**: Transitioned from roadmap to active feature! Support for parsing `.txt`, `.md`, `.csv`, `.pdf`, and `.docx` via `/ingest`, integrating text chunking, SHA256 hashed deduplication, and fast HNSW vector lookup for secure standalone querying.

## 🗺️ Roadmap & Upcoming Architecture

- **Rolling Summarization (Context Window Management)**: Moving beyond simply warning you when context thresholds hit ~75%, BuddyOS will actively manage token footprints. Once the limit is met, Buddy will automatically compress the oldest 50% of the conversation history into a dense `SYSTEM MEMORY` block, discarding raw verbose text but retaining the core logical flow entirely seamlessly.
- **Plugin-Based Tool Decoupling**: The current tool implementations hardcoded in `core/tools.py` will be transitioned to a modular, decoupled plugin architecture (e.g., a `plugins/` directory). Individual tools and sub-agents will be constructed as independent classes that auto-register tightly with the runtime upon startup, unlocking a vastly expanded multi-agent ecosystem.
