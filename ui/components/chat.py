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

    # Initialise flags on first run
    if "stop_generation" not in st.session_state:
        st.session_state.stop_generation = False
    if "pending_user_message" not in st.session_state:
        st.session_state.pending_user_message = None
    if "is_generating" not in st.session_state:
        st.session_state.is_generating = False

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
        st.session_state.pending_user_message = None
        st.session_state.is_generating = False
        st.rerun()
        return
    # ────────────────────────────────────────────────────────────────────────

    # Ensure messages are synced to the active conversation
    if st.session_state.conversation_id and not st.session_state.messages:
        st.session_state.messages = run_async(db.get_messages(st.session_state.conversation_id))

    # Step 1: a brand-new submission just arrived. Append the user message once,
    # stash it as "pending", and rerun immediately. This guarantees the
    # generation step below only ever runs in a dedicated rerun — never twice
    # for the same submission — and the history loop is the single source of
    # truth for everything that's rendered.
    if user_input and not st.session_state.pending_user_message and not st.session_state.is_generating:
        if isinstance(st.session_state.messages, list):
            st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.pending_user_message = user_input
        st.rerun()
        return

    # Plain container — no fixed height. Messages grow naturally and the page
    # scrolls; the pinned top-level st.chat_input stays at the viewport bottom.
    with st.container():
        if not st.session_state.messages:
            st.info("👋 Hello! I am Buddy. Ask me anything or share something about yourself!")
        else:
            for msg in st.session_state.messages:
                role = msg.get("role", "assistant") if isinstance(msg, dict) else msg.role
                content = msg.get("content", "") if isinstance(msg, dict) else msg.content
                with st.chat_message(role):
                    st.markdown(content)

        # Step 2: generate the reply for the pending user message. Guarded by
        # is_generating so a stray duplicate rerun mid-flight can't trigger a
        # second LLM call / second append for the same turn.
        if st.session_state.pending_user_message and not st.session_state.is_generating:
            pending_message = st.session_state.pending_user_message
            st.session_state.is_generating = True

            with st.chat_message("assistant"):
                # 2. Add an explicit 'stop' button above the chat stream.
                # Use a larger portion of the column or no column at all to prevent
                # the button text from wrapping vertically inside tight column constraints.
                st.button(
                    "🛑 Stop Generating",
                    key=f"stop_{id(pending_message)}",
                    help="Cancel & discard this response",
                    type="secondary",
                    on_click=lambda: st.session_state.update({"stop_generation": True}),
                )

                # 3. Block while the LLM + tool chain runs
                status_container = st.status("Buddy is thinking...", expanded=True)
                
                try:
                    response = None
                    gen = orchestrator.process_message(
                        user_message=pending_message,
                        conversation_id=st.session_state.conversation_id,
                        model_id=st.session_state.current_model_id,
                    )
                    
                    for item in run_async_gen(gen):
                        if isinstance(item, dict) and item.get("type") == "status":
                            status_container.write(item["msg"])
                        elif isinstance(item, dict) and item.get("type") == "content":
                            # Intermediate content marker from the orchestrator — not the
                            # final response object, so don't treat it as one.
                            pass
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
                    st.session_state.pending_user_message = None
                    st.session_state.is_generating = False
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

            st.session_state.pending_user_message = None
            st.session_state.is_generating = False
