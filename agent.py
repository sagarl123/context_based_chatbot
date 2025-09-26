from typing import Dict, Any, List, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from datetime import datetime, timedelta
import re
import dateparser
from vectorStore import QdrantVector
import operator

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    context: Dict[str, Any]

class SessionStore:
    def __init__(self):
        self.data = {
            "booking": {"name": None, "email": None, "phone": None, "date": None},
            "booking_active": False,
            "history": []
        }
    
    def update_booking(self, field: str, value: str):
        self.data["booking"][field] = value
        print(f"Saved: {field} = {value}")
    
    def is_booking_complete(self) -> bool:
        return all(self.data["booking"].values())
    
    def get_next_field(self) -> str:
        for field in ["name", "email", "phone", "date"]:
            if not self.data["booking"][field]:
                return field
        return None

session = SessionStore()

@tool
def search_documents(query: str) -> str:
    """Search documents for relevant information."""
    try:
        vector_store = QdrantVector(
            qdrant_url="http://localhost:6333",
            collection_name="metacloud",
            embedding_model="llama3.2:3b"
        )
        if not vector_store.connect_client():
            return "Error: Could not connect to document database"
        results = vector_store.find_similar_texts(query, k=3)
        if not results:
            return "No relevant information found."
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def extract_info(message: str, field_type: str) -> str:
    """Extract specific information from user message using LLM."""
    llm = ChatOllama(model="llama3.2:3b", temperature=0.1)
    
    prompts = {
        "name": f"""Extract the person's full name from this message: "{message}"
        Return ONLY the extracted name or "NOT_FOUND".""",
        
        "email": f"""Extract the email address from this message: "{message}"
        Return ONLY the email or "NOT_FOUND".""",
        
        "phone": f"""Extract the phone number from this message: "{message}"
        Return ONLY the phone number or "NOT_FOUND".""",
        
        "date": f"""Extract any date/time reference from this message: "{message}"
        Return EXACT date/time phrase or "NOT_FOUND"."""
    }
    
    response = llm.invoke([HumanMessage(content=prompts[field_type])])
    result = response.content.strip()
    
    if field_type == "email":
        match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', result)
        result = match.group(0) if match else None
    elif field_type == "phone":
        match = re.search(r'[\+\d\-\(\)\s]{10,20}', result)
        result = match.group(0) if match else None
    elif result == "NOT_FOUND":
        result = None
    
    return result.strip() if result else None

@tool
def parse_date(date_text: str) -> str:
    """Convert natural language date to YYYY-MM-DD format using dateparser."""
    if not date_text:
        return None
    parsed = dateparser.parse(date_text)
    if parsed:
        return parsed.strftime('%Y-%m-%d')
    return None

def validate_phone(phone: str) -> bool:
    phone_digits = re.sub(r'\D', '', phone)
    return 10 <= len(phone_digits) <= 15

def validate_input(field: str, value: str) -> str:
    if not value:
        return f"Invalid {field} format"
    
    if field == "phone":
        return "valid" if validate_phone(value) else f"Invalid {field} format"
    
    if field == "email":
        return "valid" if re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', value) else f"Invalid {field} format"
    
    if field == "name":
        return "valid" if re.match(r"^[A-Za-z\s'-]{2,}$", value) else f"Invalid {field} format"
    
    if field == "date":
        return "valid" if value else f"Invalid {field} format"
    
    return "valid"

class ChatBot:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2:3b", temperature=0.1)
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(ChatState)
        
        workflow.add_node("router", self._router)
        workflow.add_node("document_handler", self._handle_documents)
        workflow.add_node("booking_handler", self._handle_booking)
        workflow.add_node("general_handler", self._handle_general)
        
        workflow.set_entry_point("router")
        
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "documents": "document_handler",
                "booking": "booking_handler",
                "general": "general_handler"
            }
        )
        
        workflow.add_edge("document_handler", END)
        workflow.add_edge("booking_handler", END)
        workflow.add_edge("general_handler", END)
        
        return workflow.compile()
    
    def _router(self, state: ChatState) -> ChatState:
        message = state["messages"][-1].content.lower()
        if session.data["booking_active"] or any(word in message for word in ["book", "appointment", "schedule"]):
            state["context"]["intent"] = "booking"
        elif any(word in message for word in ["document", "information", "about", "what", "how"]):
            state["context"]["intent"] = "documents"
        else:
            state["context"]["intent"] = "general"
        return state
    
    def _route_decision(self, state: ChatState) -> str:
        return state["context"]["intent"]
    
    def _handle_documents(self, state: ChatState) -> ChatState:
        query = state["messages"][-1].content
        doc_results = search_documents.invoke({"query": query})
        if "Error" in doc_results:
            response = "Sorry, I couldn't access the documents right now."
        else:
            prompt = f"Based on this information: {doc_results}\nAnswer: {query}"
            ai_response = self.llm.invoke([HumanMessage(content=prompt)])
            response = ai_response.content
        state["messages"].append(AIMessage(content=response))
        return state
    
    def _handle_booking(self, state: ChatState) -> ChatState:
        message = state["messages"][-1].content
        if not session.data["booking_active"]:
            session.data["booking_active"] = True
            response = "I'll help you book an appointment. Let's start with your full name."
        else:
            next_field = session.get_next_field()
            if next_field:
                # Special handling for date: try parsing the user's full message as date
                if next_field == "date":
                    extracted = parse_date(message)  # directly parse message
                else:
                    extracted = extract_info.invoke({"message": message, "field_type": next_field})
                
                if extracted:
                    validation = validate_input(next_field, extracted)
                    if validation == "valid":
                        session.update_booking(next_field, extracted)
                        if session.is_booking_complete():
                            booking = session.data["booking"]
                            response = (f"Booking Complete!\nName: {booking['name']}\nEmail: {booking['email']}"
                                        f"\nPhone: {booking['phone']}\nDate: {booking['date']}\nYour appointment is confirmed!")
                            session.data["booking_active"] = False
                        else:
                            next_field = session.get_next_field()
                            prompts = {
                                "email": "Great! Now I need your email address.",
                                "phone": "Perfect! What's your phone number?",
                                "date": "Excellent! When would you like your appointment? (e.g., 'tomorrow', 'next Monday')"
                            }
                            response = prompts.get(next_field, f"Now I need your {next_field}.")
                    else:
                        response = f"{validation}. Please provide a valid {next_field}."
                else:
                    field_prompts = {
                        "name": "I need your full name to proceed.",
                        "email": "Please provide your email address.",
                        "phone": "Please provide your phone number.",
                        "date": "When would you like your appointment?"
                    }
                    response = field_prompts.get(next_field, f"Please provide your {next_field}.")
            else:
                response = "Your booking information is already complete!"
        state["messages"].append(AIMessage(content=response))
        return state
    
    def _handle_general(self, state: ChatState) -> ChatState:
        message = state["messages"][-1].content
        prompt = f"""You are a helpful assistant. You can help with:
                    1. Answering questions about documents
                    2. Booking appointments

                    User message: {message}

                    Respond helpfully and guide them to available services if appropriate."""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        state["messages"].append(AIMessage(content=response.content))
        return state
    
    def chat(self, message: str) -> str:
        session.data["history"].append({"role": "user", "content": message})
        state = ChatState(messages=[HumanMessage(content=message)], context={})
        result = self.graph.invoke(state)
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        response = ai_messages[-1].content if ai_messages else "I didn't understand that."
        session.data["history"].append({"role": "assistant", "content": response})
        return response

if __name__ == "__main__":
    bot = ChatBot()
    print("Chatbot Ready! Ask about documents or book appointments.")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
        if user_input:
            response = bot.chat(user_input)
            print(f"Bot: {response}\n")
