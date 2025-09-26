from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
import docx


def split_documents(text: str, chunk_size: int = 1000, chunk_overlap: int = 20):
    if not text:
        return []
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_text(text)
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return []


def read_file(file_path: str):
    file_type = Path(file_path).suffix.lower()
    if file_type == ".txt":
        return read_text(file_path)
    elif file_type == ".pdf":
        return read_pdf(file_path)
    elif file_type == ".docx":
        return read_doc(file_path)
    else:
        print(f"Unsupported file format: {file_type}")
        return None


def read_text(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading text file: {e}")
        return None


def read_pdf(file_path: str):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        if not documents:
            return ""
        return "\n".join(doc.page_content or "" for doc in documents)
    except Exception as e:
        print(f"Error loading pdf: {e}")
        return None


def read_doc(file_path: str):
    try:
        d = docx.Document(file_path)
        return "\n".join(p.text for p in d.paragraphs)
    except Exception as e:
        print(f"Error reading docx file: {e}")
        return None


if __name__ == "__main__":
    file_path = "NepaliBert.pdf"
    text = read_file(file_path)
    chunks = split_documents(text)
    print(chunks)
