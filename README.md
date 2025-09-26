# 📄 Agent + RAG System

This project integrates **appointment booking (ChatBot)** and **document-based Q&A (RAG)** into a single FastAPI backend, with a Streamlit frontend.

---

## 🚀 Features

- **Appointment Booking**  
  Users interact with the `/chat` endpoint, powered by `agent.ChatBot`.

- **Document Upload & Ingestion**  
  Upload files via `/documents/upload`. Each file is passed to `rag_app.RAGApp.add_document(path)`.

- **Ask Questions About Documents**  
  Query the uploaded documents using `/rag/ask`.

- **Vector Store Integration**  
  Uses `vectorStore.QdrantVector` and **Qdrant DB** for semantic search & retrieval.

- **Streamlit Frontend**  
  Provides an easy-to-use interface for uploading documents, chatting with the booking agent, and asking questions about documents.

---



## 📂 Project Structure

```
fastapi_backend/
├── main.py                # FastAPI backend
├── frontend.py            # Streamlit frontend tester
├── agent.py               # Booking chatbot logic
├── rag_app.py             # RAG application logic
├── document_processor.py  # File reading and splitting utilities
├── vectorStore.py         # Qdrant vector DB wrapper
├── requirements.txt       # Backend requirements
├── requirements.tests.txt # Test requirements
├── README.md              # Project documentation
└── tests/                 # Pytest test suite
```

---

## ⚡ Quickstart

1. **Setup environment**
   ```bash
   cd fastapi_backend
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the backend**
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   Open Swagger UI at: [http://localhost:8000/docs](http://localhost:8000/docs)

3. **Run the frontend**
   ```bash
   pip install streamlit requests
   streamlit run frontend.py
   ```

---

## ✅ Endpoints

### `/chat`  
Send booking-related messages to the ChatBot.

### `/documents/upload`  
Upload one or more documents. Each file is processed and ingested.

### `/rag/ask`  
Ask a question about the uploaded documents.


