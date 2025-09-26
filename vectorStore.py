from uuid import uuid4

from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

from document_processor import read_file, split_documents


class QdrantVector:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "metacloud",
        embedding_model: str = "llama3.2:3b",
        file_path: str = "NepaliBert.pdf",
    ):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding = OllamaEmbeddings(model=embedding_model)
        self.file_path = file_path
        self.client: QdrantClient | None = None

    def connect_client(self):
        try:
            self.client = QdrantClient(url=self.qdrant_url)
            return self.client
        except Exception as e:
            print(f"Error occurred while connecting to Qdrant client: {e}")
            self.client = None
            return None

    def _ensure_connected(self):
        if self.client is None:
            raise RuntimeError("Qdrant client is not connected. Call connect_client() first.")

    def _embedding_dim(self) -> int:
        test_vec = self.embedding.embed_query("to check dimension")
        return len(test_vec)

    def create_collection(self):
        try:
            self._ensure_connected()
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            if self.collection_name not in collection_names:
                size = self._embedding_dim()
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=size, distance=Distance.COSINE),
                )
                print(f"Collection '{self.collection_name}' created with size={size}.")
            else:
                print("Collection already exists.")
        except Exception as e:
            print(f"Error occurred while creating collection: {e}")

    def add_texts_to_collection(self):
        try:
            self._ensure_connected()

            texts = self._get_texts()
            if not texts:
                print("No texts to add (empty or failed to read).")
                return

            # Ensure collection exists (idempotent)
            self.create_collection()

            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding,
            )

            ids = [str(uuid4()) for _ in texts]
            vector_store.add_texts(texts=texts, ids=ids)
            print(f"Added {len(texts)} chunks successfully.")
        except Exception as e:
            print(f"Error occurred while adding texts to collection: {e}")

    def find_similar_texts(self, query: str, k: int = 3):
        try:
            self._ensure_connected()
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding,
            )
            results = vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error occurred while finding similar texts: {e}")
            return None

    def _get_texts(self):
        text = read_file(self.file_path)
        if text is None:
            return []
        return split_documents(text)


if __name__ == "__main__":
    qd = QdrantVector()
    qd.connect_client()
    qd.create_collection()
    qd.add_texts_to_collection()

    results = qd.find_similar_texts("What are the processes used to generate embeddings?", k=3)
    if results:
        for res in results:
            print(res.page_content, end="\n\n")
    else:
        print("No results.")
