import logging
import logging.handlers
import sys
from pathlib import Path

# ── Logging bootstrap ──────────────────────────────────────────────────────────
# Must run before any local imports so every module's logger is captured.
# Mirrors the same setup in main.py (CLI entry-point) so the Streamlit UI
# path writes to the same rotating log file.
_LOGS_DIR = Path("logs")
_LOGS_DIR.mkdir(exist_ok=True)

_root = logging.getLogger()
if not _root.handlers:          # guard: only configure once per process
    _root.setLevel(logging.DEBUG)

    _fh = logging.handlers.RotatingFileHandler(
        _LOGS_DIR / "buddy-os.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    _root.addHandler(_fh)

    _ch = logging.StreamHandler(sys.stderr)
    _ch.setLevel(logging.WARNING)
    _ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    _root.addHandler(_ch)

    # Silence noisy LiteLLM internals
    import litellm
    litellm.suppress_debug_info = True
    litellm.verbose = False
    litellm.drop_params = True
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
# ──────────────────────────────────────────────────────────────────────────────

import streamlit as st

# Configure the page - MUST be the first Streamlit command
st.set_page_config(
    page_title="BuddyOS",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

from ui.state import init_session_state
from ui.components.sidebar import render_sidebar
from ui.components.chat import render_chat
from ui.components.facts_viewer import render_facts_viewer
from ui.components.document_ingestor import render_document_ingestor

def main():
    # 1. Initialize core system safely via singleton and state vars
    init_session_state()

    # 2. Render sidebar navigation
    render_sidebar()

    # 3. CSS:
    #    (a) Extra bottom padding so messages are never hidden behind the pinned bar.
    #    (b) Constrain the pinned chat input bar to the left (main) column only by
    #        eating into the bar's right side by ~35% — the approximate width of
    #        the right context column inside section.main.
    st.markdown(
        """<style>
        section.main > div.block-container { padding-bottom: 6rem; }
        [data-testid="stBottom"] {
            padding-right: 35% !important;
            box-sizing: border-box !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    # 4. Top-level chat_input — Streamlit pins this to the viewport bottom.
    #    Must be outside columns so it gets true viewport-sticky behaviour.
    user_input = st.chat_input("Chat with Buddy...")

    # 5. Render main columns/tabs
    col_main, col_context = st.columns([2, 1], gap="large")

    with col_main:
        st.title("🤖 BuddyOS Chat")
        render_chat(user_input=user_input)

    with col_context:
        # Use tabs for the right-hand panel: Facts vs Ingestion
        tab_facts, tab_docs = st.tabs(["🧠 Memory Facts", "📚 Local Documents"])

        with tab_facts:
            render_facts_viewer()

        with tab_docs:
            render_document_ingestor()

if __name__ == "__main__":
    main()
