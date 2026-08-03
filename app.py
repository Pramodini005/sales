import os
import tempfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables or .env file.")
    st.stop()

# ==========================================================
# Streamlit Page Config
# ==========================================================

st.set_page_config(
    page_title="PragyanAI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 PragyanAI Conversational AI Assistant")
st.caption("Powered by Groq + LangChain + FAISS")

# ==========================================================
# Create Default FAQ Excel (If missing)
# ==========================================================

if not os.path.exists("pragyan_faq_prices.xlsx"):
    faq_data = {
        "Category": [
            "Program Overview", "Program Structure", "Program Structure",
            "Pricing", "Pricing", "Curriculum", "Curriculum",
            "Evaluation", "Career", "Leadership"
        ],
        "Question": [
            "What is PragyanAI?", "What happens in Phase 1?", "What happens in Phase 2?",
            "What is the fee?", "Expected salary?", "Months 1-3 syllabus?",
            "Months 4-6 syllabus?", "Evaluation process?", "Career tracks?", "Who leads PragyanAI?"
        ],
        "Answer": [
            "18 month AI Program.", "Offline intensive training.", "Internship and placement.",
            "₹50,000 + ₹50,000 Success Fee.", "₹6LPA to ₹25LPA.", "Python, Data Science, ML.",
            "DL, NLP, GenAI, Agentic AI.", "Seminars and Hackathons.", "7 AI Career Tracks.",
            "Sateesh Ambesange."
        ]
    }
    pd.DataFrame(faq_data).to_excel("pragyan_faq_prices.xlsx", index=False)

# ==========================================================
# Cache Embedding Model
# ==========================================================

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# ==========================================================
# Vector Store Loader
# ==========================================================

def load_documents_into_vectorstore(uploaded_files=None):
    documents = []

    # Load Default PDF
    if os.path.exists("PragyanAI_Presentation.pdf"):
        loader = PyPDFLoader("PragyanAI_Presentation.pdf")
        documents.extend(loader.load())

    # Load FAQ Excel
    if os.path.exists("pragyan_faq_prices.xlsx"):
        df = pd.read_excel("pragyan_faq_prices.xlsx")
        for _, row in df.iterrows():
            documents.append(
                Document(
                    page_content=f"Category: {row['Category']}\nQuestion: {row['Question']}\nAnswer: {row['Answer']}"
                )
            )

    # Process User Uploads
    if uploaded_files:
        for uploaded_file in uploaded_files:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name

            try:
                if suffix.lower() == ".pdf":
                    loader = PyPDFLoader(temp_path)
                    documents.extend(loader.load())
                elif suffix.lower() in [".xlsx", ".xls"]:
                    df = pd.read_excel(temp_path)
                    for _, row in df.iterrows():
                        text = " | ".join([str(i) for i in row.values])
                        documents.append(Document(page_content=text))
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    if not documents:
        return None

    return FAISS.from_documents(documents, embedding_model)

# Initialize Session Vector Store
if "vectorstore" not in st.session_state:
    with st.spinner("Initializing Knowledge Base..."):
        st.session_state.vectorstore = load_documents_into_vectorstore()

# ==========================================================
# Persona Prompts
# ==========================================================

SALES_PROMPTS = {
    "PragyanAI Student Counselor": """You are PragyanAI's official AI Student Counselor.

Use ONLY the provided context below to answer questions:
{context}

Rules:
1. Answer strictly using only the provided context.
2. If information is unavailable, politely state: "I couldn't find that information in the provided documents."
3. Keep answers professional, concise, and helpful.
4. Encourage students to ask follow-up questions.""",

    "Technical Mentor": """You are an AI Technical Mentor for PragyanAI.

Context:
{context}

Rules:
- Explain technical concepts clearly using practical examples.
- If the required details are not found in the context, clearly state that you don't know based on the provided documents.""",

    "Placement Advisor": """You are PragyanAI's Placement Advisor.

Context:
{context}

Rules:
- Answer placement-related queries regarding career tracks, salaries, and internships.
- Never fabricate information not present in the context."""
}

# ==========================================================
# LLM & Chat History Memory
# ==========================================================

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# ==========================================================
# RAG Chain Construction
# ==========================================================

def create_rag_chain(persona_name, retrieved_context):
    base_instructions = SALES_PROMPTS.get(
        persona_name,
        SALES_PROMPTS["PragyanAI Student Counselor"]
    )
    
    # Safely inject context to prevent prompt template parsing errors on retrieved text
    system_prompt = base_instructions.replace("{context}", retrieved_context)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    return prompt | llm | StrOutputParser()

def respond(message, persona_name):
    if st.session_state.vectorstore is None:
        return "Knowledge base is empty. Please check your document sources."

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(message)
    context = "\n\n".join(doc.page_content for doc in docs)

    chain = create_rag_chain(persona_name, context)

    conversational_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    session_id = f"session_{persona_name}"

    return conversational_chain.invoke(
        {"input": message},
        config={"configurable": {"session_id": session_id}}
    )

def clear_chat_history(persona):
    session_id = f"session_{persona}"
    if session_id in st.session_state.store:
        st.session_state.store[session_id].clear()

# ==========================================================
# Sidebar Settings
# ==========================================================

st.sidebar.title("⚙️ Settings")

persona = st.sidebar.selectbox(
    "Choose Persona",
    list(SALES_PROMPTS.keys())
)

uploaded_files = st.sidebar.file_uploader(
    "Upload Additional PDFs / Excel",
    type=["pdf", "xlsx", "xls"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button("Update Knowledge Base"):
        with st.spinner("Rebuilding Vector Database..."):
            updated_vs = load_documents_into_vectorstore(uploaded_files)
            if updated_vs is not None:
                st.session_state.vectorstore = updated_vs
                st.sidebar.success("Knowledge Base Updated!")

if st.sidebar.button("🗑️ Clear Chat"):
    clear_chat_history(persona)
    st.session_state.messages = []
    st.rerun()

# ==========================================================
# UI Chat State & Rendering
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================================
# User Input Processing
# ==========================================================

question = st.chat_input("Ask anything about PragyanAI...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = respond(question, persona)
            except Exception as e:
                answer = f"❌ Error: {str(e)}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ==========================================================
# Footer
# ==========================================================

st.markdown("---")
st.caption("🚀 PragyanAI Conversational Assistant | Powered by Streamlit + LangChain + Groq")
