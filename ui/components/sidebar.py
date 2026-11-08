import streamlit as st
from core.models import CONTEXT_PRESETS, resolve_context_window
from ui.state import run_async

def render_sidebar():
    with st.sidebar:
        st.title("🤖 BuddyOS")
        st.markdown("Your Model-Agnostic AI Assistant")
        st.divider()

        # Model Selector
        st.subheader("⚙️ Settings")
        if not st.session_state.get("api_keys_configured", True):
            st.warning(
                "No API keys found. Add GEMINI_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY "
                "to your .env file and restart."
            )
        models = st.session_state.models
        
        # Build options mapping
        model_options = {m.model_id: f"{'🆓' if m.tier == 'free' else '💰'} {m.display_name}" for m in models}
        current_index = 0
        for i, m in enumerate(models):
            if m.model_id == st.session_state.current_model_id:
                current_index = i
                break
                
        selected_model_id = st.selectbox(
            "Active Model",
            options=[m.model_id for m in models],
            format_func=lambda x: model_options[x],
            index=current_index
        )
        
        if selected_model_id != st.session_state.current_model_id:
            st.session_state.current_model_id = selected_model_id
            st.rerun()

        # Context size: how much conversation history each message carries
        active_model = next((m for m in models if m.model_id == st.session_state.current_model_id), None)

        def _size_label(size_key: str) -> str:
            window = resolve_context_window(active_model, size_key)
            return f"{CONTEXT_PRESETS[size_key].label} ({window // 1000}K)"

        # Streamlit drops a widget's state when a run ends before the widget renders (the
        # st.rerun() above), so the choice lives in a plain session value that the radio
        # reads as its starting point and writes back to.
        size_keys = list(CONTEXT_PRESETS)
        st.session_state.context_size = st.radio(
            "Context Window",
            options=size_keys,
            index=size_keys.index(st.session_state.context_size),
            format_func=_size_label,
            key="context_size_radio",
            horizontal=True,
            help="Small sends only the most recent turns and costs less per message. Large remembers more of the chat.",
        )
        preset = CONTEXT_PRESETS[st.session_state.context_size]
        summarize_at = int(
            resolve_context_window(active_model, st.session_state.context_size)
            * st.session_state.orchestrator.summarization_threshold
        )
        st.caption(
            f"Sends the last ~{preset.recent_messages // 2} exchanges (up to {preset.history_budget // 1000}K tokens). "
            f"Older messages are summarized past ~{summarize_at // 1000}K tokens."
        )

        st.divider()

        # Conversation Management
        st.subheader("💬 Conversations")
        
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.rerun()

        # Load history
        db = st.session_state.db
        history = run_async(db.list_conversations(limit=10))
        
        if history:
            with st.container(height=350, border=False):
                for conv in history:
                    if isinstance(conv, dict):
                        conv_id = conv.get("id")
                    elif hasattr(conv, "id"):
                        conv_id = conv.id
                    else:
                        conv_id = conv[0]

                    # Priority: LLM summary → first-message title → short UUID
                    summary = getattr(conv, "summary", "") or ""
                    title_fallback = getattr(conv, "title", "") or ""
                    label = summary or title_fallback or f"Chat {str(conv_id)[:6]}"

                    btn_type = "primary" if conv_id == st.session_state.conversation_id else "secondary"
                    if st.button(label, key=f"hist_{conv_id}", use_container_width=True, type=btn_type):
                        st.session_state.conversation_id = conv_id
                        st.session_state.messages = run_async(db.get_messages(conv_id))
                        st.rerun()
        else:
            st.info("No previous conversations")

        if history:
            st.divider()
            if st.button("🗑️ Clear All Chats", use_container_width=True, type="secondary"):
                run_async(db.purge_all_conversations())
                st.session_state.conversation_id = None
                st.session_state.messages = []
                st.rerun()

        st.divider()
        st.markdown("### Help")
        st.markdown("- **Facts:** Check what Buddy knows about you in the Memory tab.\n- **Ingest:** Upload docs to your Local RAG DB.\n- **Model:** Switch models freely, even mid-chat!")
