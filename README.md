# BuddyOS 🤖

![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)
![Version](https://img.shields.io/badge/version-0.1.0-green.svg)

BuddyOS is a highly flexible, model-agnostic AI assistant and orchestrator. It is designed to act as a personal "buddy" that continuously learns about you, maintains context across sessions, and dynamically routes your queries to the best available AI models.

## ✨ Key Features

- **Model-Agnostic Routing**: Powered by [LiteLLM](https://github.com/BerriAI/litellm), BuddyOS can discover available models dynamically based on your environment keys. It supports automated token counting and a **graceful fallback chain** (if one model goes down, it switches to the next available).
- **Hybrid Persistence Layer**:
  - **SQLite** (`aiosqlite`): Transactional tracking of conversations, message history, timestamp updates, and token usage limits.
  - **DuckDB**: Analytical engine for lightning-fast retrieval of learned "User Facts".
- **Continuous Learning (Automated Fact Extraction)**: Buddy constantly evaluates your conversations in the background. It extracts information about you (e.g., job, preferences, name) and stores them as active facts to customize future system prompts.
- **Dynamic Context Window Management**: Constantly monitors context tokens and triggers summarization when the context threshold (~75%) is reached, preventing the LLM from forgetting the start of a long conversation.
- **Interactive CLI**: Comes with an interactive terminal interface equipped with commands (`/facts`, `/history`, `/model`, `/new`) to manage your Buddy context easily.

## 🛠 Tech Stack

- **Language**: Python 3.12+
- **Frameworks**: Pydantic-AI (Agent logic), LiteLLM (Routing/Tokenization)
- **Database**: SQLite (`aiosqlite`), DuckDB
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

4. Create a `.env` file in the root directory and add your API keys:
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
├── core/               # Core engine (LiteLLM router, Hybrid DB manager)
├── data/               # Local persistence layer (.db and .duckdb generated here)
├── ui/                 # Streamlit UI logic (WIP)
├── main.py             # CLI Entry point
└── pyproject.toml      # Project definitions & dependencies
```
