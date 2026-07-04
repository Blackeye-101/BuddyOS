import streamlit as st
from ui.state import run_async

def render_facts_viewer():
    st.subheader("🧠 What Buddy Knows About You")
    
    db = st.session_state.db
    facts = run_async(db.get_user_facts(active_only=True))
    
    if not facts:
        st.info("No facts recorded yet. Start chatting about your preferences, background, or goals!")
        return

    if st.button("🗑️ Clear All Facts", use_container_width=True, type="secondary"):
        run_async(db.purge_all_facts())
        st.rerun()

    # Group facts by category
    from collections import defaultdict
    categories = defaultdict(list)
    
    for fact in facts:
        # Pydantic models or named tuples
        cat = getattr(fact, "category", "General")
        categories[cat].append(fact)

    with st.container(height=500, border=False):
        for cat, items in categories.items():
            with st.expander(f"📁 {cat} ({len(items)} facts)", expanded=True):
                for fact in items:
                    fact_text = getattr(fact, "fact_text", "")
                    confidence = getattr(fact, "confidence", 0.0)
                    conf_pct = int(confidence * 100)

                    color = "green" if confidence >= 0.85 else "orange"
                    st.markdown(f"- **{fact_text}** `[{conf_pct}% confidence]`")
