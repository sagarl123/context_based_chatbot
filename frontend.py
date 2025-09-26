import streamlit as st
import requests
from typing import List, Dict

# === Config ===
st.set_page_config(page_title="Agent + RAG Tester", layout="wide")
st.title("🧪 Agent (Booking) + RAG (Docs) — Tester")

# Sidebar — API URL + health
st.sidebar.subheader("API Server")
API_URL = st.sidebar.text_input("Base URL", value="http://localhost:8000")
health_status = "Unknown"
try:
    r = requests.get(f"{API_URL}/health", timeout=5)
    health_status = "✅ Live" if r.ok else f"⚠️ {r.status_code}"
except Exception as e:
    health_status = f"❌ {e}"
st.sidebar.write("Health:", health_status)

# Session state for chat history (Booking)
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, str]] = []

# Tabs for flows
tab_booking, tab_docs = st.tabs(["📅 Appointment Booking (ChatBot)", "📚 Document Tools (RAG)"])

# -------------------------------
# TAB 1 — Appointment Booking
# -------------------------------
with tab_booking:
    st.subheader("Chat with Booking Agent")
    st.caption("Type natural language like: *“Book an appointment for Friday 3 PM for John Doe.”*")

    # Render chat history
    for msg in st.session_state.chat_history:
        role = msg["role"]
        text = msg["text"]
        if role == "user":
            st.chat_message("user").markdown(text)
        else:
            st.chat_message("assistant").markdown(text)

    # Input
    user_input = st.chat_input("Message the booking assistant…")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        try:
            resp = requests.post(f"{API_URL}/chat", json={"message": user_input}, timeout=30)
            if resp.ok:
                data = resp.json()
                reply = data.get("reply", "")
                st.session_state.chat_history.append({"role": "assistant", "text": reply})
                st.chat_message("assistant").markdown(reply)
            else:
                err = f"Error {resp.status_code}: {resp.text}"
                st.session_state.chat_history.append({"role": "assistant", "text": err})
                st.chat_message("assistant").markdown(err)
        except Exception as e:
            err = f"❌ Request failed: {e}"
            st.session_state.chat_history.append({"role": "assistant", "text": err})
            st.chat_message("assistant").markdown(err)

    cols = st.columns(2)
    if cols[0].button("🧹 Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

# -------------------------------
# TAB 2 — RAG Document Tools
# -------------------------------
with tab_docs:
    st.subheader("Add Documents")
    st.caption("Upload files. Your backend calls `RAGApp.add_document(file_path)` per file.")
    files = st.file_uploader("Choose files", accept_multiple_files=True)
    if st.button("Upload"):
        if files:
            try:
                payload = [("files", (f.name, f.getvalue(), getattr(f, "type", "application/octet-stream"))) for f in files]
                resp = requests.post(f"{API_URL}/documents/upload", files=payload, timeout=120)
                if resp.ok:
                    st.success("Uploaded.")
                    st.json(resp.json())
                else:
                    st.error(f"Upload failed: {resp.status_code}")
                    st.code(resp.text)
            except Exception as e:
                st.error(f"Upload error: {e}")
        else:
            st.info("No files selected.")

    st.divider()

    st.subheader("Ask Questions (RAG)")
    question = st.text_area("Question", placeholder="e.g., What is the refund policy?")
    k_rag = st.number_input("Top K (RAG)", min_value=1, max_value=20, value=3)
    if st.button("Ask RAG"):
        if question.strip():
            try:
                resp = requests.post(f"{API_URL}/rag/ask", json={"question": question, "k": int(k_rag)}, timeout=60)
                if resp.ok:
                    st.json(resp.json())
                else:
                    st.error(f"RAG failed: {resp.status_code}")
                    st.code(resp.text)
            except Exception as e:
                st.error(f"RAG request error: {e}")
        else:
            st.info("Enter a question.")

    st.divider()

    st.subheader("Search Vector Store (Optional)")
    st.caption("Directly call `QdrantVector.find_similar_texts(query, k)`.")
    query = st.text_input("Search query", placeholder="e.g., cancellation terms")
    k_vs = st.number_input("Top K (Search)", min_value=1, max_value=20, value=3, key="k_vs")
    if st.button("Search"):
        if query.strip():
            try:
                resp = requests.post(f"{API_URL}/documents/search", json={"query": query, "k": int(k_vs)}, timeout=60)
                if resp.ok:
                    st.json(resp.json())
                else:
                    st.error(f"Search failed: {resp.status_code}")
                    st.code(resp.text)
            except Exception as e:
                st.error(f"Search request error: {e}")
        else:
            st.info("Enter a query.")
