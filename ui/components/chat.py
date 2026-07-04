import streamlit as st
from ui.state import run_async
from core.orchestrator import OrchestratorRequest

def render_chat(user_input=None):
    db = st.session_state.db
    orchestrator = st.session_state.orchestrator

    # Ensure messages are synced to the active conversation
    if st.session_state.conversation_id and not st.session_state.messages:
        st.session_state.messages = run_async(db.get_messages(st.session_state.conversation_id))

    # Plain container — no fixed height. Messages grow naturally and the page
    # scrolls; the pinned top-level st.chat_input stays at the viewport bottom.
    with st.container():
        if not st.session_state.messages and not user_input:
            st.info("👋 Hello! I am Buddy. Ask me anything or share something about yourself!")
        else:
            for msg in st.session_state.messages:
                role = msg.get("role", "assistant") if isinstance(msg, dict) else msg.role
                content = msg.get("content", "") if isinstance(msg, dict) else msg.content
                with st.chat_message(role):
                    st.markdown(content)

        # Handle new input — rendered inside the same container so it appears
        # inline in the message flow, not below the input bar.
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            if isinstance(st.session_state.messages, list):
                st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                with st.spinner("Buddy is thinking..."):
                    request = OrchestratorRequest(
                        user_message=user_input,
                        conversation_id=st.session_state.conversation_id,
                        model_id=st.session_state.current_model_id,
                    )
                    try:
                        response = run_async(
                            orchestrator.process_message(
                                user_message=request.user_message,
                                conversation_id=request.conversation_id,
                                model_id=request.model_id,
                            )
                        )
                        st.markdown(response.response)
                        st.session_state.conversation_id = response.conversation_id
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response.response}
                        )
                        if response.fallback_occurred:
                            st.toast(
                                f"Fallback triggered! Failed over from {response.fallback_from}",
                                icon="⚠️",
                            )
                        if response.extracted_facts:
                            st.toast(
                                f"Learned {len(response.extracted_facts)} new fact(s) about you!",
                                icon="✨",
                            )
                    except Exception as e:
                        st.error(f"Error processing message: {str(e)}")
                        st.exception(e)
