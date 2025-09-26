from langchain_ollama import ChatOllama
from vectorStore import QdrantVector
from document_processor import read_file
import sys


class RAGApp:
    def __init__(
        self,
        file_path: str = "NepaliBert.pdf",
        collection_name: str = "metacloud",
        embedding_model: str = "llama3.2:3b",
        chat_model: str = "llama3.2:3b",
        qdrant_url: str = "http://localhost:6333"
    ):
        self.file_path = file_path
        self.vector_store = QdrantVector(
            qdrant_url=qdrant_url,
            collection_name=collection_name,
            embedding_model=embedding_model,
            file_path=file_path
        )
        self.llm = ChatOllama(model=chat_model, temperature=0.7)
        self.setup_complete = False

    def setup(self):
        """Initialize the RAG system by connecting to Qdrant and setting up the vector store."""
        print("Setting up RAG system...")
        
        # Connect to Qdrant
        if not self.vector_store.connect_client():
            print("Failed to connect to Qdrant. Make sure Qdrant is running.")
            return False
        
        # Create collection and add documents
        self.vector_store.create_collection()
        self.vector_store.add_texts_to_collection()
        
        self.setup_complete = True
        print("RAG system setup complete!")
        return True

    def ask(self, question: str, k: int = 3) -> str:
        """Ask a question and get an answer based on the document content."""
        if not self.setup_complete:
            return "Error: RAG system not set up. Call setup() first."
        
        try:
            # Retrieve relevant documents
            print(f"Searching for relevant information...")
            similar_docs = self.vector_store.find_similar_texts(question, k=k)
            
            if not similar_docs:
                return "No relevant information found in the document."
            
            # Combine retrieved content
            context = "\n\n".join([doc.page_content for doc in similar_docs])
            
            # Create prompt
            prompt = f"""Based on the following context from the document, please answer the question.

                        Context:
                        {context}

                        Question: {question}

                        Answer: Please provide a comprehensive answer based only on the information provided in the context above. 
                        If the context doesn't contain enough information to answer the question, please say so.
                        """

            # Generate response
            print("Generating answer...")
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Error processing question: {e}"

    def chat_loop(self):
        """Start an interactive chat session."""
        if not self.setup_complete:
            if not self.setup():
                return
        
        print("\n" + "="*50)
        print("RAG Chat Application")
        print("Type 'quit', 'exit','bye' ,or 'q' to stop")
        print("="*50 + "\n")
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q', '','bye']:
                    print("Goodbye!")
                    break
                
                print("\nThinking...")
                answer = self.ask(question)
                print(f"\nAnswer: {answer}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    def add_document(self, new_file_path: str):
        """Add a new document to the vector store."""
        if not self.setup_complete:
            print("Setup RAG system first.")
            return False
        
        try:
            # Temporarily change file path
            original_path = self.vector_store.file_path
            self.vector_store.file_path = new_file_path
            
            # Add new document
            self.vector_store.add_texts_to_collection()
            
            # Restore original path
            self.vector_store.file_path = original_path
            
            print(f"Successfully added document: {new_file_path}")
            return True
            
        except Exception as e:
            print(f"Error adding document: {e}")
            return False


def main():
    # Configuration
    config = {
        "file_path": "NepaliBert.pdf", 
        "collection_name": "metacloud",
        "embedding_model": "llama3.2:3b",
        "chat_model": "llama3.2:3b",
        "qdrant_url": "http://localhost:6333"
    }
    
    # Check if file path is provided as command line argument
    if len(sys.argv) > 1:
        config["file_path"] = sys.argv[1]
    
    # Initializing RAG app
    rag_app = RAGApp(**config)
    
    # Setup and start chat
    if rag_app.setup():
        rag_app.chat_loop()
    else:
        print("Failed to setup RAG system. Please check your configuration.")


if __name__ == "__main__":
    main()