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
