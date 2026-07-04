from pathlib import Path
import streamlit as st
from ui.state import run_async
from core.document_parser import DocumentParser
from core.embeddings import generate_embeddings_batch

DOCS_DIR = Path("data/documents")


def render_document_ingestor():
    st.subheader("📚 Local Document Ingestion (Personal RAG)")
    st.write(
        "Upload documents here. BuddyOS will chunk and embed these files, "
        "allowing you to ask questions about them directly!"
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    db = st.session_state.db

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["txt", "md", "pdf", "csv", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.spinner("Parsing and embedding documents…"):
            for file in uploaded_files:
                file_path = DOCS_DIR / file.name

                # Persist raw file to disk first
                with open(file_path, "wb") as fh:
                    fh.write(file.getbuffer())

                try:
                    metadata = DocumentParser.process_file(str(file_path))

                    # Skip if already in DuckDB (same content hash)
                    if run_async(db.document_hash_exists(metadata["filehash"])):
                        st.toast(f"{file.name} already ingested — skipping", icon="ℹ️")
                        continue

                    # Embed all chunks in a single batch call
                    chunk_texts = metadata["chunks"]  # list[str] from the parser
                    embeddings = run_async(generate_embeddings_batch(chunk_texts))
                    metadata["chunks"] = [
                        {"text": t, "embedding": e}
                        for t, e in zip(chunk_texts, embeddings)
                    ]

                    run_async(db.save_document(metadata))
                    st.toast(
                        f"Ingested {file.name} — {len(chunk_texts)} chunks", icon="✅"
                    )

                except Exception as exc:
                    st.error(f"Failed to ingest {file.name}: {exc}")
                    # Remove orphaned raw file so it doesn’t linger on disk
                    if file_path.exists():
                        file_path.unlink()

        st.rerun()

    st.divider()

    # ── Ingested document list ──────────────────────────────────────────────────
    existing_docs = run_async(db.list_documents())

    col_hdr, col_btn = st.columns([3, 1])
    with col_hdr:
        st.markdown("#### Ingested Documents")
    with col_btn:
        if existing_docs and st.button(
            "🗑️ Clear All", use_container_width=True, type="secondary"
        ):
            # Delete physical files
            for f in DOCS_DIR.glob("*.*"):
                f.unlink(missing_ok=True)
            # Wipe DuckDB rows (chunks + documents; facts are untouched)
            run_async(db.purge_all_documents())
            st.rerun()

    if not existing_docs:
        st.info("No documents ingested yet.")
    else:
        for doc in existing_docs:
            st.markdown(f"- 📄 `{doc['filename']}`")
