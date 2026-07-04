import os
from pathlib import Path
import streamlit as st

DOCS_DIR = Path("data/documents")

def render_document_ingestor():
    st.subheader("📚 Local Document Ingestion (Personal RAG)")
    st.write("Upload documents here. BuddyOS will chunk and embed these files, allowing you to ask questions about them directly!")
    
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    
    uploaded_files = st.file_uploader(
        "Upload files", 
        type=["txt", "md", "pdf", "csv", "docx"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for file in uploaded_files:
            file_path = DOCS_DIR / file.name
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
                
            st.toast(f"Saved {file.name}", icon="💾")
            # TODO: Integrate with backend embedding logic once available
            # e.g., run_async(orchestrator.ingest_document(file_path))
            
        st.success("Files saved and queued for ingestion!")
            
    st.divider()
    st.markdown("#### Embedded Documents")
    
    # Simple list of available documents
    existing_docs = list(DOCS_DIR.glob("*.*"))
    if not existing_docs:
        st.info("No documents uploaded yet.")
    else:
        for doc in existing_docs:
            st.markdown(f"- 📄 `{doc.name}`")
