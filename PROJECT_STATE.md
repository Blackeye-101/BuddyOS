# BuddyOS Project State

## 📊 Status

**Current Version:** `0.1.0` (Alpha/Prototype phase)
**Primary Environment:** CLI App via `main.py`
**Dependencies Context:** Python 3.12+, LiteLLM, Pydantic-AI, aiosqlite, DuckDB.

---

## 🏛 Architecture Overview

BuddyOS operates around an **Orchestrator** (the "Buddy" agent), acting as the central coordination layer between a diverse set of AI models (via **Router**) and a persistent memory store (via **Hybrid Database**).

### Core Components

1.  **`buddy-os/main.py` (CLI Interface)**
    - **State**: Functional
    - **Responsibilities**: Handles user interaction on the terminal, processes slash commands (`/help`, `/facts`, `/model`, `/new`, `/history`), loops until `/exit`.
    - **Workflow**: Creates components via `initialize()`, initiates an `asyncio` event loop.

2.  **`buddy-os/agents/buddy.py` (`BuddyOrchestrator`)**
    - **State**: Functional
    - **Capabilities**:
      - Manages context flow: loading history, building the dynamic prompt with "User Facts".
      - Fires background async tasks (`_extract_facts_background`) to extract knowledge using a secondary LLM pipeline.
      - Saves token counts and message histories.
    - **Implementation**: Utilizes `Pydantic-AI` constructs wrapping requests to `Router`.

3.  **`buddy-os/core/router.py` (`BuddyRouter`)**
    - **State**: Functional
    - **Capabilities**:
      - Wrapper over `litellm`. It token-counts prompts (`token_counter`) before sending them.
      - Manages a "fallback chain". If the active model hits a rate limit or API failure, it retries down the fallback chain.
      - Raises custom `RouterError` if all models fail.

4.  **`buddy-os/core/database.py` (`BuddyDatabase`)**
    - **State**: Functional (Hybrid Architecture)
    - **Capabilities**:
      - **SQLite** (`aiosqlite`) - Transactional layer for: `conversations` (UUID, timestamp updates) and `messages` (role, content, tokens).
      - **DuckDB** - Analytical layer for fast knowledge lookup (`UserFact`): category, text, confidence, active status.

5.  **`buddy-os/core/discovery.py` & `core/deps.py`**
    - **State**: Functional
    - **Capabilities**: Dynamic model discovery detecting local `_API_KEY` env vars and listing what's available contextually. `BuddyDeps` manages the Dependency Injection container used heavily by Pydantic-AI.

6.  **`buddy-os/ui/`**
    - **State**: Stubbed / Not Implemented
    - **Capabilities**: Intended house for a richer Streamlit frontend. Empty besides `__init__.py`.

---

## 🚀 Working Capabilities

1.  **Conversational Chatting**: Multi-turn dialogue with memory intact across restarts.
2.  **Agent Fact Extraction**: As users share data (e.g., "I'm a scientist"), Buddy silently categorizes and stores it via DuckDB and weaves it into future prompts dynamically.
3.  **Model Availability Scanning**: On launch, the system automatically checks what inference engines are alive/available (OpenAI, Anthropic, Google, etc.).
4.  **Graceful Failover**: Fault-tolerant execution if API providers stutter or fail.
5.  **Slash Commands**: Functional CLI features (`/model`, `/facts`).

---

## 🚧 Areas of Refinement & Missing Features

- **DuckDB Overkill/Opportunity**: DuckDB is fundamentally built for large analytical workloads. Right now, it's used for small facts (`UserFact` table). Vector Embeddings/RAG should likely be integrated into it (DuckDB supports VSS) for proper semantic searches rather than plain text matching.
- **Fact Conflict Resolution**: Over time, extracted facts might conflict (e.g., "I like cats" followed months later by "I prefer dogs entirely"). `update_fact_confidence` and `deactivate_fact` exist but conflict-detection logic needs refinement.
- **Web UI**: The `ui/` directory indicates that a graphical interface is pending. (Streamlit is in `pyproject.toml`).
- **Token Summarization Implementation**: The code comments indicate that context window management (summarizing conversations when approaching 75% limits) is planned/stubbed, but the heavy lifting of summarization logic inside the Orchestrator needs rigorous testing.
- **Testing**: E2E testing is currently manual (Verification Tests in `main.py`). A strong Pytest suite is required.
