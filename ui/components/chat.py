import streamlit as st
from ui.state import run_async, run_async_gen
from core.orchestrator import OrchestratorRequest


def _word_stream(text: str):
    """Sync generator that yields the response word-by-word for st.write_stream."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word if i == 0 else " " + word


def render_chat(user_input=None):
    db = st.session_state.db
    orchestrator = st.session_state.orchestrator

    # Initialise stop flag on first run
    if "stop_generation" not in st.session_state:
        st.session_state.stop_generation = False

    # ── Stop-generation handler ────────────────────────────────────────────────
    # Runs at the TOP of the render triggered by the 🛑 button click.
    # Deletes the discarded turn from the DB and removes it from session state.
    if st.session_state.stop_generation:
        last_ids = st.session_state.pop("_last_turn_ids", {})
        ids_to_delete = [v for v in last_ids.values() if v]
        if ids_to_delete:
            run_async(db.delete_messages(ids_to_delete))
        msgs = st.session_state.get("messages", [])
        
        def _get_role(m):
            return m.get("role") if isinstance(m, dict) else m.role
            
        if (
            len(msgs) >= 2
            and _get_role(msgs[-2]) == "user"
            and _get_role(msgs[-1]) == "assistant"
        ):
            st.session_state.messages = msgs[:-2]
        elif msgs and _get_role(msgs[-1]) == "user":
            st.session_state.messages = msgs[:-1]
            
        st.session_state.stop_generation = False
        st.rerun()
        return
    # ────────────────────────────────────────────────────────────────────────

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

        if user_input:
            # 1. Show user message immediately
            with st.chat_message("user"):
                st.markdown(user_input)
            if isinstance(st.session_state.messages, list):
                st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                # 2. Add an explicit 'stop' button above the chat stream.
                # Use a larger portion of the column or no column at all to prevent
                # the button text from wrapping vertically inside tight column constraints.
                st.button(
                    "🛑 Stop Generating",
                    key=f"stop_{id(user_input)}",
                    help="Cancel & discard this response",
                    type="secondary",
                    on_click=lambda: st.session_state.update({"stop_generation": True}),
                )

                # 3. Block while the LLM + tool chain runs
                status_container = st.status("Buddy is thinking...", expanded=True)
                
                try:
                    response = None
                    gen = orchestrator.process_message(
                        user_message=user_input,
                        conversation_id=st.session_state.conversation_id,
                        model_id=st.session_state.current_model_id,
                    )
                    
                    for item in run_async_gen(gen):
                        if isinstance(item, dict) and item.get("type") == "status":
                            status_container.write(item["msg"])
                        else:
                            response = item
                            
                    status_container.update(label="Done thinking!", state="complete", expanded=False)
                except Exception as e:
                    status_container.update(label="An error occurred", state="error", expanded=True)
                    st.error(f"Error processing message: {str(e)}")
                    st.exception(e)
                    # Roll back the pending user message
                    if (
                        st.session_state.messages
                        and st.session_state.messages[-1].get("role") == "user"
                    ):
                        st.session_state.messages.pop()
                    return

                # If Stop was clicked during thinking, discard now without streaming
                if st.session_state.get("stop_generation"):
                    st.session_state._last_turn_ids = {
                        "user_msg_id": response.user_message_id,
                        "asst_msg_id": response.assistant_message_id,
                    }
                    # stop handler will fire on next rerun (already triggered by button click)
                    return

                # 4. Stream the reply word-by-word
                st.write_stream(_word_stream(response.response))

                # 5. Persist in session state
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
