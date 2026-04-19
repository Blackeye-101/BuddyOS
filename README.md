# BuddyOS 🤖

![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![Version](https://img.shields.io/badge/version-0.1.0-green.svg)

BuddyOS is a highly flexible, model-agnostic AI assistant and orchestrator. It is designed to act as a personal "buddy" that continuously learns about you, maintains context across sessions, and dynamically routes your queries to the best available AI models.

## ✨ Key Features

- **Model-Agnostic Routing**: Powered by [LiteLLM](https://github.com/BerriAI/litellm), BuddyOS can discover available models dynamically based on your environment keys. It supports automated token counting and a **graceful fallback chain** (if one model goes down, it switches to the next available).
- **Hybrid Persistence Layer**:
  - **SQLite** (`aiosqlite`): Transactional tracking of conversations, message history, timestamp updates, and token usage limits.
  - **DuckDB**: Analytical engine for lightning-fast retrieval of learned "User Facts".
- **RAG-Powered Memory (Semantic Fact Retrieval)**: Buddy embeds every learned fact locally using `fastembed` (ONNX Runtime, no PyTorch required). At each turn, it runs a DuckDB VSS cosine-similarity search to inject only the top-5 most relevant facts into the system prompt — keeping context lean and precise.
- **Continuous Learning (Automated Fact Extraction)**: Buddy constantly evaluates your conversations in the background. It extracts information about you (e.g., job, preferences, name) and stores them as active facts to customize future system prompts.
- **Dynamic Context Window Management**: Constantly monitors context tokens and triggers summarization when the context threshold (~75%) is reached, preventing the LLM from forgetting the start of a long conversation.
- **Real-Time Web Search & Tool Calling**: Buddy is equipped with an integrated web search tool using DuckDuckGo (`ddgs`). When asked about recent events, current stock prices, or unknown facts, it dynamically pauses the conversation, searches the web, and integrates the live results into its final answer seamlessly.
- **Interactive CLI**: Comes with an interactive terminal interface equipped with commands (`/facts`, `/history`, `/model`, `/new`) to manage your Buddy context easily.

## 🛠 Tech Stack

- **Language**: Python 3.12+
- **Frameworks**: Pydantic-AI (Agent logic), LiteLLM (Routing/Tokenization)
- **Tools Integrations**: `ddgs` (DuckDuckGo Search)
- **Database**: SQLite (`aiosqlite`), DuckDB + VSS extension (HNSW vector index)
- **Embeddings**: [`fastembed`](https://github.com/qdrant/fastembed) — `BAAI/bge-small-en-v1.5` (384-dim, ONNX Runtime, no PyTorch)
- **Frontend (Planned)**: Streamlit

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

Run the interactive CLI application:

```bash
python main.py
```

### CLI Commands

Inside the chat loop, you can use the following commands:

- `/help` - Show available commands
- `/facts` - View what Buddy actively knows about you
- `/model` - Switch the current AI model on the fly
- `/new` - Start a fresh conversation
- `/history` - View a list of your recent conversations
- `/exit` - Save state and gracefully exit

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
