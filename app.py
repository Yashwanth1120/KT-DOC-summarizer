import os
import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage



# --- Configuration and Initialization ---
GOOGLE_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

llm = ChatGoogleGenerativeAI(model="models/gemma-4-31b-it", google_api_key=GOOGLE_API_KEY)

# --- Setup for Internet RAG ---
big_paragraph = (
    "The Internet is a global system of interconnected computer networks that uses the Internet protocol suite (TCP/IP) to communicate between networks and devices. It is a network of networks that consists of private, public, academic, business, and government networks of local to global scope, linked by a broad array of electronic, wireless, and optical networking technologies. The Internet carries a vast range of information resources and services, such as the inter-linked hypertext documents and applications of the World Wide Web (WWW), electronic mail, telephony, and file sharing. \n\n" +
    "The origins of the Internet date back to the development of packet switching and research commissioned by the United States Department of Defense in the 1960s to enable time-sharing of computers. The primary precursor network, the ARPANET, initially served as a backbone for interconnection of academic and research networks. The funding of the National Science Foundation Network (NSFNET) in the 1980s, as well as private commercial Internet service providers, led to the worldwide participation in the development of new networking technologies and the merger of many networks. The commercialization of the Internet in the mid-1990s marked a turning point in its expansion, as it began to permeate almost every aspect of modern human life.\n\n" +
    "Today, the Internet is a pervasive global information medium. Users communicate with one another by electronic mail and can share information and data. It supports various applications, including cloud computing, video conferencing, online gaming, and social media. The impact of the Internet on society has been profound, influencing commerce, education, government, healthcare, and daily communication. While it offers unprecedented access to information and facilitates global connectivity, it also presents challenges related to privacy, security, and the spread of misinformation. Continuous innovation in its underlying technologies and applications continues to shape its future trajectory."
)
internet_documents = [Document(page_content=big_paragraph)]
internet_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
internet_chunks = internet_text_splitter.split_documents(internet_documents)

embeddings_internet = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
embedding_dim_internet = len(embeddings_internet.embed_query("hello world"))
index_internet = faiss.IndexFlatL2(embedding_dim_internet)
vector_store_internet = FAISS(
    embedding_function=embeddings_internet,
    index=index_internet,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)
vector_store_internet.add_documents(documents=internet_chunks)

@tool(response_format="content_and_artifact")
def retrieve_internet_context(query: str):
    """Retrieve information regarding history of internet to help answer a query."""
    retrieved_docs = vector_store_internet.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# --- Setup for KT Guide RAG ---
kt_guide_content = """
Welcome to InnovateCorp! This Knowledge Transfer (KT) guide is designed to help new employees navigate their initial weeks and understand key aspects of our operations. Our core values are Innovation, Collaboration, and Customer Focus.

**Team Structure:** You will be joining the 'Project Alpha' team, reporting to Sarah Chen, the Senior Project Manager. Your direct teammates include David Lee (Lead Developer), Maria Rodriguez (UI/UX Designer), and Tom Jackson (QA Engineer). Our team meetings are held every Monday at 10 AM in Conference Room 3, and daily stand-ups are at 9:30 AM via Google Meet.

**Key Tools & Software:** For project management, we use Jira for task tracking and Confluence for documentation. Our primary communication tool is Slack for instant messaging and Google Workspace for email and calendars. Development work is primarily done using Python and JavaScript, with code hosted on GitHub. Access to these tools will be granted within your first three days.

**Onboarding Process:** Your first week will focus on setup and introductions. You'll receive your laptop and login credentials on day one. HR will conduct an orientation session on Tuesday covering company policies, benefits, and payroll. You'll have one-on-one meetings with your team members throughout the week. By the end of your second week, you should have access to all necessary systems and have completed mandatory compliance training modules.

**Important Resources:** The company's internal knowledge base can be found at `internal.innovatecorp.com/kb`. This includes FAQs, best practices, and troubleshooting guides. For IT support, please submit a ticket via `support.innovatecorp.com` or call extension 5555. Health and wellness benefits information is available on the HR portal.

**Culture & Expectations:** InnovateCorp encourages a proactive and collaborative environment. We value open communication and continuous learning. Don't hesitate to ask questions; your team is here to support your growth. Performance reviews are conducted quarterly, and professional development courses are available through our 'InnovateLearn' platform.
"""
kt_documents = [Document(page_content=kt_guide_content)]
kt_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
kt_chunks = kt_text_splitter.split_documents(kt_documents)

embeddings_kt_guide = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)
embedding_dim_kt = len(embeddings_kt_guide.embed_query("hello world"))
index_kt = faiss.IndexFlatL2(embedding_dim_kt)
vector_store_kt_guide = FAISS(
    embedding_function=embeddings_kt_guide,
    index=index_kt,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)
vector_store_kt_guide.add_documents(documents=kt_chunks)

@tool(response_format="content_and_artifact")
def retrieve_kt_context(query: str):
    """Retrieve information from the InnovateCorp KT Guide to help answer a query."""
    retrieved_docs = vector_store_kt_guide.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# --- Multi-tool Agent Setup ---
all_tools = [retrieve_internet_context, retrieve_kt_context]

multi_tool_prompt = (
    "You are a helpful assistant with access to two tools:\n" +
    "1. `retrieve_internet_context`: Use this tool to answer questions about the history and general information of the internet.\n" +
    "2. `retrieve_kt_context`: Use this tool to answer questions specifically about InnovateCorp's Knowledge Transfer (KT) Guide, company policies, team structure, tools, and onboarding process.\n\n" +
    "Choose the appropriate tool based on the user's query. If the query is related to neither the internet nor the KT guide, or if the retrieved context does not contain relevant information, politely state that you don't have the information or that the query is outside your scope."
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

    if messages is None:
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
    | RunnableLambda(extract_text_response)
).with_types(input_type=AgentInput, output_type=str)

# --- FastAPI Routes ---
app = FastAPI(title="Multi-tool RAG Agent")
add_routes(app, formatted_agent_chain, path="/agent", playground_type="default")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
