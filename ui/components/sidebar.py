import streamlit as st
from ui.state import run_async

def render_sidebar():
    with st.sidebar:
        st.title("🤖 BuddyOS")
        st.markdown("Your Model-Agnostic AI Assistant")
        st.divider()

        # Model Selector
        st.subheader("⚙️ Settings")
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

                    # Show LLM-generated summary when available, else short ID
                    summary = getattr(conv, "summary", "") or ""
                    label = summary if summary else f"Chat {str(conv_id)[:6]}"

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
