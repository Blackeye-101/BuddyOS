import asyncio
import threading
import streamlit as st
from core.discovery import discover_available_models, get_default_model
from core.database import create_database
from core.router import create_router
from agents.buddy import create_orchestrator

# Thread-safe event loop for async operations in Streamlit
_loop = asyncio.new_event_loop()
_loop_thread = threading.Thread(target=_loop.run_forever, daemon=True)
_loop_thread.start()

def run_async(coro):
    """Safely run async coroutines in Streamlit."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result()


def run_async_gen(async_gen):
    """Safely run async generators in Streamlit."""
    import queue
    q = queue.Queue()
    _DONE = object()
    _ERROR = object()

    async def _exhaust():
        try:
            async for item in async_gen:
                q.put(("item", item))
            q.put(("done", _DONE))
        except Exception as e:
            q.put(("error", e))

    asyncio.run_coroutine_threadsafe(_exhaust(), _loop)

    while True:
        msg_type, payload = q.get()
        if msg_type == "item":
            yield payload
        elif msg_type == "done":
            break
        elif msg_type == "error":
            raise payload

@st.cache_resource
def get_system_components():
    """Initialize system components exactly once."""
    # Models
    models = discover_available_models(include_paid=True)
    if not models:
        models = [get_default_model()]
    default_model = next((m for m in models if m.model_id == "gemini/gemini-2.5-flash"), models[0])

    # Database
    db = run_async(create_database(
        sqlite_path="data/buddy.db",
        duckdb_path="data/knowledge.duckdb"
    ))

    # Router
    router = create_router()

    # Orchestrator
    orchestrator = create_orchestrator(router, db)

    return models, default_model, db, router, orchestrator

def init_session_state():
    """Initialize Streamlit session state variables."""
    models, default_model, db, router, orchestrator = get_system_components()
    
    if "models" not in st.session_state:
        st.session_state.models = models
    if "db" not in st.session_state:
        st.session_state.db = db
    if "router" not in st.session_state:
        st.session_state.router = router
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = orchestrator
        
    if "current_model_id" not in st.session_state:
        st.session_state.current_model_id = default_model.model_id
        
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
