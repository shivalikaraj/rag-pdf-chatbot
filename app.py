import streamlit as st
from rag.pipeline import (
    build_index,
    query_rag,
    reset_collection,
    stream_gemini,
    get_indexed_document_count,
    get_available_documents,
)


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="📄",
    layout="wide",
)


# ---------------- HEADER ----------------

st.title("💬 AI Knowledge Assistant")
st.caption("Ask question based on your document")


# ---------------- SESSION STATE ----------------

if "collection" not in st.session_state:
    st.session_state.collection = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []

if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = "All Documents"


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("📄Document Controls")

    uploaded_files = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.caption("Uploaded files:")

        for file in uploaded_files:
            st.write(f"{file.name}")

    # -------- Process --------
    if st.button("Process Document"):
        if uploaded_files is None:
            st.warning("Please upload a PDF first.")
        else:
            try:

                with st.spinner("Processing PDF and building index..."):
                    for file in uploaded_files:
                        collection, _ = build_index(file)

                st.session_state.collection = collection
                st.success(
                    f"{len(uploaded_files)} documets indexed successfully!"
                )

            except Exception as e:
                st.error(f"Error: {e}")

    # -------- Reset --------
    if st.button("Reset DB"):
        try:
            if st.session_state.collection is not None:
                reset_collection(st.session_state.collection)

            st.session_state.collection = None
            st.session_state.messages = []
            st.session_state.last_chunks = []
            st.session_state.selected_doc = "All Documents"

            st.success("Reset complete")

        except Exception as e:
            st.error(f"Reset failed: {e}")

    # ---------------- Collection Info ----------------
    if st.session_state.collection is not None:
        total_docs = get_indexed_document_count(st.session_state.collection)
        st.info(f"Indexed documents: {total_docs}")

        available_documents = get_available_documents(
            st.session_state.collection)
        selected_doc = st.selectbox(
            "Search Scope",
            available_documents,
        )
        st.session_state.selected_doc = selected_doc


# ---------------- ONBOARDING (shown only when no collection is loaded) ----------------

if st.session_state.collection is None:
    st.info("""
📄 No document loaded yet

👉 Step 1: Upload a PDF (sidebar)  
👉 Step 2: Click "Process Document"  
👉 Step 3: Start asking questions  

💡 Your AI assistant will answer based only on your document.
""")


# ---------------- CHAT DISPLAY ----------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------- USER INPUT ----------------

prompt = st.chat_input("Ask a question")


# ---------------- QUERY FLOW ----------------

if prompt:
    # guard check
    if st.session_state.collection is None:
        st.warning("Please upload and process a document first.")
        st.stop()

    # Store and display user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG pipeline
    try:
        with st.spinner("Thinking..."):
            full_prompt, chunks = query_rag(
                st.session_state.collection,
                prompt,
                source_filter=st.session_state.selected_doc
            )
        st.session_state.last_chunks = chunks

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    # Stream and store assistant response
    with st.chat_message("assistant"):
        response = st.write_stream(stream_gemini(full_prompt))

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": chunks,
    })


# ---------------- SOURCES ----------------

if st.session_state.last_chunks:
    st.caption(
        f"Retrieved {len(st.session_state.last_chunks)} chunks"
    )
    st.subheader("📚 Sources")

    for chunk in st.session_state.last_chunks[:3]:
        title = (
            f"{chunk['source']} "
            f"| Page {chunk['page']} "
            f"| Relevance {chunk['rerank_score']:.2f}"
        )
        with st.expander(title):
            st.write(chunk["text"])
