
import sys
import types
import importlib
import pathlib
import pytest
from fastapi.testclient import TestClient

# Ensure backend path is importable
BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# ---- Dummy implementations strictly matching your method signatures ----
# agent.ChatBot.chat(self, message: str) -> str
class _DummyChatBot:
    def __init__(self):
        self.calls = []

    def chat(self, message: str) -> str:
        self.calls.append(message)
        return f"echo: {message}"

# rag_app.RAGApp.setup(), ask(question: str, k: int = 3), add_document(new_file_path: str)
class _DummyRAG:
    def __init__(self):
        self.documents = []
        self.setup_called = False

    def setup(self):
        self.setup_called = True

    def ask(self, question: str, k: int = 3):
        return f"answer({k}): {question}"

    def add_document(self, new_file_path: str):
        self.documents.append(new_file_path)
        return True

# vectorStore.QdrantVector.find_similar_texts(self, query: str, k: int = 3)
class _DummyVector:
    def __init__(self):
        pass

    def find_similar_texts(self, query: str, k: int = 3):
        return [{"text": f"match: {query}", "k": k}]

@pytest.fixture(autouse=True)
def inject_dummy_modules(monkeypatch):
    # Build dummy modules and inject into sys.modules BEFORE importing main.py
    m_agent = types.ModuleType("agent")
    m_agent.ChatBot = _DummyChatBot

    m_rag = types.ModuleType("rag_app")
    m_rag.RAGApp = _DummyRAG

    m_vs = types.ModuleType("vectorStore")
    m_vs.QdrantVector = _DummyVector

    monkeypatch.setitem(sys.modules, "agent", m_agent)
    monkeypatch.setitem(sys.modules, "rag_app", m_rag)
    monkeypatch.setitem(sys.modules, "vectorStore", m_vs)

    yield

@pytest.fixture()
def app():
    # Import (or reload) the FastAPI app after dummies are in place
    main = importlib.import_module("main")
    # If already imported, reload to ensure fixtures are used
    main = importlib.reload(main)
    return main.app

@pytest.fixture()
def client(app):
    return TestClient(app)
