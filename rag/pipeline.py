import os
import re
import uuid
import hashlib

import numpy as np
import pdfplumber
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai


# CONFIG

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL = "gemini-2.5-flash-lite"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_chunks"

RETRIEVAL_K = 20
FINAL_K = 8

API_KEY_GEMINI = os.getenv("GEMINI_API_KEY")
if not API_KEY_GEMINI:
    raise EnvironmentError("Missing GEMINI_API_KEY environment variable")


# CACHED MODELS

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_reranker():
    return CrossEncoder(RERANKER_MODEL)


@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=API_KEY_GEMINI)


# PDF PROCESSING

def load_pdf(file):
    pages = []
    with pdfplumber.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                pages.append({
                    "page": page_num,
                    "text": page_text,
                })
    return pages


def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()


def generate_file_hash(file):
    file.seek(0)
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)
    return file_hash


# CHUNKING

def chunk_pages(pages):
    all_chunks = []
    for page in pages:
        cleaned_text = clean_text(page["text"])
        for chunk in semantic_chunk(cleaned_text):
            all_chunks.append(
                {
                    "text": chunk,
                    "page": page["page"]
                }
            )
    return all_chunks


# SEMANTIC CHUNKING

def semantic_chunk(text, threshold=0.2, max_words=300):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [
        s.strip()
        for s in sentences
        if s.strip()
    ]
    if not sentences:
        return []

    embeddings = create_embeddings(sentences)
    similarities = []
    for i in range(len(embeddings)-1):
        similarity = np.dot(
            embeddings[i],
            embeddings[i+1]
        )
        similarities.append(float(similarity))

    chunks = []
    current_chunk = [sentences[0]]

    for i, similarity in enumerate(similarities):
        next_sentence_words = len(sentences[i+1].split())
        current_chunk_words = len(" ".join(current_chunk).split())

        if similarity > threshold and current_chunk_words + next_sentence_words < max_words:
            current_chunk.append(sentences[i+1])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i+1]]

    chunks.append(" ".join(current_chunk))
    return chunks


# EMBEDDINGS

def create_embeddings(texts):
    return get_embedding_model().encode(texts).tolist()


# VECTOR DB (Chroma)

def get_chroma_collection():
    return get_chroma_client().get_or_create_collection(name=COLLECTION_NAME)


def document_exists(collection, file_hash):
    results = collection.get(
        where={
            "file_hash": file_hash,
        }
    )
    return len(results["ids"]) > 0


def index_document(collection, chunks, filename, file_hash):

    documents = [
        chunk["text"]
        for chunk in chunks
    ]
    embeddings = create_embeddings(documents)

    ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]
    metadatas = [
        {
            "source": filename,
            "file_hash": file_hash,
            "chunk_id": i,
            "page": chunk["page"]
        }
        for i, chunk in enumerate(chunks)
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return collection


def get_indexed_document_count(collection):
    data = collection.get(include=["metadatas"])
    metadatas = data["metadatas"]

    unique_docs = {
        meta["source"]
        for meta in metadatas
    }
    return len(unique_docs)


def get_available_documents(collection):
    data = collection.get(include=["metadatas"])
    metadatas = data["metadatas"]
    sources = {
        meta.get("source")
        for meta in metadatas
        if meta.get("source")
    }
    return ["All Documents"] + sorted(list(sources))


# RETRIEVAL

def retrieve_chunks(collection, query, k, source_filter=None):
    query_embedding = create_embeddings([query])[0]

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": k,
        "include": [
            "documents",
            "distances",
            "metadatas",
        ]
    }

    if source_filter and source_filter != "All Documents":
        query_params["where"] = {"source": source_filter}

    results = collection.query(**query_params)
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    chunks = []
    for doc, dist, meta in zip(documents, distances, metadatas):
        chunks.append({
            "text": doc,
            "score": round(1-dist, 3),
            "source": meta["source"],
            "page": meta["page"],
            "chunk_id": meta["chunk_id"],
        })
    return chunks


def rerank_chunks(question, chunks):
    reranker = get_reranker()
    pairs = [
        (question, chunk["text"])
        for chunk in chunks
    ]
    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    chunks.sort(
        key=lambda chunk: chunk["rerank_score"],
        reverse=True
    )
    return chunks


# PROMPT BUILDING

def get_chat_history(limit=3):
    messages = st.session_state.messages
    history = []

    for message in messages[-limit * 2:]:
        history.append(
            f"{message['role']}: {message['content']}"
        )
    return "\n".join(history)


def build_prompt(context, question, chat_history):
    return f"""
You are a helpful assistant. 

Conversational History:
{chat_history}

Context:
{context}

Question:
{question}

Answer ONLY using the provided context.
If the answer is not in the context, say: "I don't know based on the provided document."
""".strip()


# LLM STREAMING

def stream_gemini(prompt: str):
    client = get_gemini_client()
    chat = client.chats.create(model=GEMINI_MODEL)
    response = chat.send_message_stream(prompt)

    for token in response:
        if token.text:
            yield token.text


# PUBLIC PIPELINE API (used by app.py)

def build_index(file):
    pages = load_pdf(file)
    chunks = chunk_pages(pages)
    collection = get_chroma_collection()
    file_hash = generate_file_hash(file)

    if document_exists(collection, file_hash):
        print(f"{file.name} already indexed")
        return collection, []

    collection = index_document(collection, chunks, file.name, file_hash)
    return collection, chunks


def query_rag(collection, question, source_filter=None):
    retrieved_chunks = retrieve_chunks(
        collection, question, k=RETRIEVAL_K, source_filter=source_filter
    )

    reranked_chunks = rerank_chunks(question, retrieved_chunks)
    top_chunks = reranked_chunks[:FINAL_K]

    texts = [c["text"] for c in top_chunks]
    context = "\n\n".join(texts)
    chat_history = get_chat_history()
    prompt = build_prompt(context, question, chat_history)

    return prompt, top_chunks


def reset_collection(collection):
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)
