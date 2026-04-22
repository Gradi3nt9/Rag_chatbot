import streamlit as st
import os
from rag_engine import (
    load_pdfs, split_into_chunks, build_vector_store,
    load_existing_vector_store, build_rag_chain, query_rag
)

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")
st.title("📄 PDF RAG Chatbot")
st.caption("Ask questions about your uploaded PDFs")

# --- Sidebar: PDF Management ---
with st.sidebar:
    st.header("Document Setup")
    
    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True
    )
    
    if uploaded_files:
        os.makedirs("pdfs", exist_ok=True)
        for f in uploaded_files:
            with open(os.path.join("pdfs", f.name), "wb") as out:
                out.write(f.read())
        st.success(f"Uploaded {len(uploaded_files)} file(s)")
    
    if st.button("🔄 Process PDFs", type="primary"):
        with st.spinner("Loading and chunking PDFs..."):
            docs = load_pdfs("pdfs")
            chunks = split_into_chunks(docs)
        st.info(f"Split into {len(chunks)} chunks")
        
        with st.spinner("Embedding chunks (first run downloads model ~90MB)..."):
            vector_store = build_vector_store(chunks)
            st.session_state.vector_store = vector_store
        
        with st.spinner("Building RAG chain..."):
            st.session_state.rag_chain = build_rag_chain(vector_store)
        
        st.success("Ready to chat!")
    
    # Load existing DB if already processed
    if "rag_chain" not in st.session_state and os.path.exists("./chroma_db"):
        if st.button("Load existing index"):
            vs = load_existing_vector_store()
            st.session_state.vector_store = vs
            st.session_state.rag_chain = build_rag_chain(vs)
            st.success("Loaded!")
    
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        if "rag_chain" in st.session_state:
            st.session_state.rag_chain = build_rag_chain(
                st.session_state.vector_store
            )

# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources used"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['file']}** — Page {src['page']}")

# Handle new input
if prompt := st.chat_input("Ask something about your PDFs..."):
    if "rag_chain" not in st.session_state:
        st.error("Please process your PDFs first using the sidebar.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                answer, sources = query_rag(st.session_state.rag_chain, prompt)
            st.markdown(answer)
            if sources:
                with st.expander("📚 Sources used"):
                    for src in sources:
                        st.markdown(f"- **{src['file']}** — Page {src['page']}")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })