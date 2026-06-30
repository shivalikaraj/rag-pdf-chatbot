# AI Knowledge Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-6E56CF)
![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash%20Lite-4285F4?logo=googlegemini&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://rag-pdf-chatbot-mcjrxqwuetsztrgdytqocr.streamlit.app/)

A simple RAG (Retrieval-Augmented Generation) chatbot that answers questions based on your uploaded PDF documents. Built with Streamlit, ChromaDB, and Google Gemini.

## Features

- Upload one or more PDFs and ask questions about their content
- Semantic chunking for more coherent context retrieval
- Cross-encoder reranking for improved relevance
- Scope questions to a specific document or search across all uploaded documents
- Per-session document isolation — each user/browser session only sees their own uploads
- Streaming responses

## Tech Stack

- **UI:** Streamlit
- **PDF parsing:** pdfplumber
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Reranking:** CrossEncoder (`ms-marco-MiniLM-L-6-v2`)
- **Vector store:** ChromaDB
- **LLM:** Google Gemini (`gemini-2.5-flash-lite`)

## Project Structure

```
.
├── app.py              # Streamlit UI
├── rag/
│   ├── __init__.py
│   └── pipeline.py     # PDF processing, chunking, retrieval, and LLM logic
├── requirements.txt
└── .gitignore
```

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/shivalikaraj/rag-pdf-chatbot.git
   cd rag-pdf-chatbot
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate   # Windows (Git Bash)
   # or: source .venv/bin/activate  # macOS/Linux

   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Gemini API key:
   ```
   GEMINI_API_KEY=your-api-key-here
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## How It Works

1. **Upload & Process** — PDFs are parsed page by page, cleaned, and split into semantically coherent chunks (sentences are grouped based on embedding similarity).
2. **Index** — Each chunk is embedded and stored in ChromaDB, tagged with the document name and a session ID.
3. **Query** — When you ask a question, the top-k most similar chunks are retrieved, then reranked by a cross-encoder for relevance.
4. **Generate** — The top reranked chunks are passed as context to Gemini, which streams back an answer grounded in your document.

## Notes

- Documents are scoped per browser session — closing the tab or reloading starts a new session, so previously indexed documents won't carry over.
- On Streamlit Community Cloud, the underlying storage is ephemeral and will reset on app reboot/redeploy.

## Author

Built by [Shivalika Raj](https://github.com/shivalikaraj)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
