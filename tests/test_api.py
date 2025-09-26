
import io
import json

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "routes" in body
    assert "/chat" in body["routes"]

def test_chat_echo(client):
    payload = {"message": "hello bot"}
    r = client.post("/chat", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "reply" in data
    assert data["reply"] == "echo: hello bot"

def test_rag_ask_default_k(client):
    payload = {"question": "What is policy?"}
    r = client.post("/rag/ask", json=payload)
    assert r.status_code == 200
    data = r.json()
    # Default k=3 based on schema
    assert data["answer"] == "answer(3): What is policy?"

def test_rag_ask_custom_k(client):
    payload = {"question": "Explain refunds", "k": 7}
    r = client.post("/rag/ask", json=payload)
    assert r.status_code == 200
    assert r.json()["answer"] == "answer(7): Explain refunds"

def test_documents_search_default_k(client):
    payload = {"query": "refund policy"}
    r = client.post("/documents/search", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert isinstance(data["results"], list)
    assert data["results"][0]["text"] == "match: refund policy"
    assert data["results"][0]["k"] == 3

def test_documents_search_custom_k(client):
    payload = {"query": "cancellations", "k": 10}
    r = client.post("/documents/search", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["results"][0]["k"] == 10

def test_documents_upload_single_file(client):
    # Build a small in-memory text file
    f = io.BytesIO(b"hello world")
    f.name = "doc.txt"
    files = [("files", ("doc.txt", f, "text/plain"))]
    r = client.post("/documents/upload", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "uploaded_files" in body
    assert "added_count" in body
    assert body["added_count"] == 1
    assert len(body["uploaded_files"]) == 1

def test_documents_upload_multiple_files(client):
    f1 = io.BytesIO(b"alpha")
    f1.name = "a.txt"
    f2 = io.BytesIO(b"beta")
    f2.name = "b.txt"
    files = [
        ("files", ("a.txt", f1, "text/plain")),
        ("files", ("b.txt", f2, "text/plain")),
    ]
    r = client.post("/documents/upload", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["added_count"] == 2
    assert len(body["uploaded_files"]) == 2
