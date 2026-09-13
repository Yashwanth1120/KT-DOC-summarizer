import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from functools import lru_cache
import faiss

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

llm = ChatGoogleGenerativeAI(
    model="models/gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY
)

# --- Helper: Build FAISS Vector Store ---
def build_vector_store(content: str, embeddings_model: str) -> FAISS:
    documents = [Document(page_content=content)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=embeddings_model,
        google_api_key=GOOGLE_API_KEY
    )
    embedding_dim = len(embeddings.embed_query("hello world"))
    index = faiss.IndexFlatL2(embedding_dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(chunks)
    return vector_store

# --- Internet RAG Setup ---
internet_content = """The Internet is a global system ... (same text as before)"""
vector_store_internet = build_vector_store(
    internet_content,
    embeddings_model="models/gemini-embedding-001"
)

@tool(response_format="content_and_artifact")
@lru_cache(maxsize=128)
def retrieve_internet_context(query: str):
    """Retrieve information regarding history of internet to help answer a query."""
    docs = vector_store_internet.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in docs
    )
    return serialized, docs

# --- KT Guide RAG Setup ---
kt_guide_content = """Welcome to InnovateCorp! This Knowledge Transfer (KT) guide ... (same text as before)"""
vector_store_kt = build_vector_store(
    kt_guide_content,
    embeddings_model="models/gemini-embedding-001"
)

@tool(response_format="content_and_artifact")
@lru_cache(maxsize=128)
def retrieve_kt_context(query: str):
    """Retrieve information from the InnovateCorp KT Guide to help answer a query."""
    docs = vector_store_kt.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in docs
    )
    return serialized, docs

# --- Multi-tool Agent Setup ---
all_tools = [retrieve_internet_context, retrieve_kt_context]

multi_tool_prompt = (
    "You are a helpful assistant with access to two tools:\n"
    "1. `retrieve_internet_context`: Use this tool to answer questions about the history and general information of the internet.\n"
    "2. `retrieve_kt_context`: Use this tool to answer questions specifically about InnovateCorp's Knowledge Transfer (KT) Guide.\n\n"
    "Choose the appropriate tool based on the user's query. If the query is unrelated or context is missing, politely state that you don't have the information."
)

multi_tool_agent = create_agent(llm, all_tools, system_prompt=multi_tool_prompt)

class AgentInput(BaseModel):
    input: str = Field(description="Your message to the agent")

def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {"messages": [("user", user_input)]}

def extract_text_response(agent_output: dict) -> str:
    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")
    if not messages:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        last = messages[-1]
        return getattr(last, "content", str(last))

    return str(agent_output)

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | multi_tool_agent
    | RunnableLambda(extract_text_response)   # ✅ ensures only final text is returned
).with_types(input_type=AgentInput, output_type=str)

# --- FastAPI Setup ---
app = FastAPI(title="Multi-tool RAG Agent")
# Route setup
add_routes(app, formatted_agent_chain, path="/agent", playground_type="default")
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
