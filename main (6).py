import os
from typing import TypedDict, Literal

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

# ── Config ──────────────────────────────────────────────────────────────────
os.environ["GROQ_API_KEY"] = "GROQ_API_KEY"   # paste your key here

PDF_PATH = "data/knowledge_base.pdf"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
# ── 1. Load & chunk PDF ─────────────────────────────────────────────────────
print("📄 Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"   → {len(chunks)} chunks created")

# ── 2. Embeddings + Vector DB ───────────────────────────────────────────────
print("🔢 Building vector store...")
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=CHROMA_DIR
)
retriever = db.as_retriever(search_kwargs={"k": 3})
print("   → ChromaDB ready")

# ── 3. LLM (Groq — free) ────────────────────────────────────────────────────
llm = ChatGroq(model=GROQ_MODEL, temperature=0.2)

# ── 4. LangGraph State ──────────────────────────────────────────────────────
class GraphState(TypedDict):
    query: str
    context: str
    answer: str
    route: Literal["answer", "escalate"]
    confidence: float

# ── 5. Graph Nodes ───────────────────────────────────────────────────────────

def retrieve_node(state: GraphState) -> GraphState:
    """Retrieve relevant chunks from vector DB."""
    docs = retriever.invoke(state["query"])
    context = "\n\n".join([doc.page_content for doc in docs])
    state["context"] = context
    # Simple confidence heuristic: if very little context retrieved, low confidence
    state["confidence"] = 1.0 if len(context) > 200 else 0.3
    return state


def generate_node(state: GraphState) -> GraphState:
    """Generate an answer using the LLM."""
    prompt = f"""You are a helpful customer support assistant.
Use ONLY the context below to answer. If you cannot find the answer, say "I don't know."

Context:
{state['context']}

Question:
{state['query']}

Answer:"""
    response = llm.invoke(prompt)
    state["answer"] = response.content
    return state


def escalate_node(state: GraphState) -> GraphState:
    """Handle low-confidence queries — HITL escalation."""
    state["answer"] = (
        "⚠️  I couldn't find enough information to answer confidently.\n"
        "🧑 Escalating to a human agent...\n\n"
        "[HUMAN-IN-THE-LOOP]: A support agent will follow up shortly.\n"
        f"   Your question was: \"{state['query']}\""
    )
    return state


# ── 6. Routing Logic ────────────────────────────────────────────────────────

def route_decision(state: GraphState) -> Literal["generate", "escalate"]:
    """Route to generate or escalate based on confidence."""
    low_confidence_phrases = [
        "i don't know", "not sure", "cannot find", "no information"
    ]
    # Escalate if context is thin OR query signals complexity
    if state["confidence"] < 0.5:
        return "escalate"
    if any(p in state["query"].lower() for p in ["urgent", "lawsuit", "legal", "refund escalate"]):
        return "escalate"
    return "generate"


# ── 7. Build LangGraph ───────────────────────────────────────────────────────
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("escalate", escalate_node)

workflow.set_entry_point("retrieve")

workflow.add_conditional_edges(
    "retrieve",
    route_decision,
    {
        "generate": "generate",
        "escalate": "escalate",
    }
)

workflow.add_edge("generate", END)
workflow.add_edge("escalate", END)

app = workflow.compile()
print("✅ LangGraph workflow compiled\n")

# ── 8. Chat Loop ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  🤖 RAG Customer Support Assistant (with HITL)")
print("=" * 55)
print("  Type your question, or 'exit' to quit.\n")

while True:
    query = input("You: ").strip()
    if not query:
        continue
    if query.lower() == "exit":
        print("👋 Goodbye!")
        break

    initial_state: GraphState = {
        "query": query,
        "context": "",
        "answer": "",
        "route": "answer",
        "confidence": 1.0,
    }

    result = app.invoke(initial_state)
    print(f"\n🤖 Answer:\n{result['answer']}\n")
    print("-" * 55)