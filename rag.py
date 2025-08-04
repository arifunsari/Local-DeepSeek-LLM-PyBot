import streamlit as st  # Web UI framework for building interactive apps
from langchain_community.document_loaders import PDFPlumberLoader  # Loads structured PDF content like tables and columns
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Splits large texts into manageable chunks
from langchain_core.vectorstores import InMemoryVectorStore  # Stores document embeddings for similarity search
from langchain_ollama import OllamaEmbeddings  # Embedding generator using Ollama backend
from langchain_core.prompts import ChatPromptTemplate  # Template for LLM prompt formatting
from langchain_ollama.llms import OllamaLLM  # Loads LLM model using Ollama (local model inference)

# Custom CSS styling for dark mode and chat formatting
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stChatInput input { background-color: #1E1E1E !important; color: #FFFFFF !important; border: 1px solid #3A3A3A !important; }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important; border: 1px solid #3A3A3A !important; color: #E0E0E0 !important; border-radius: 10px; padding: 15px; margin: 10px 0;
    }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important; border: 1px solid #404040 !important; color: #F0F0F0 !important; border-radius: 10px; padding: 15px; margin: 10px 0;
    }
    .stChatMessage .avatar { background-color: #00FFAA !important; color: #000000 !important; }
    .stChatMessage p, .stChatMessage div { color: #FFFFFF !important; }
    .stFileUploader { background-color: #1E1E1E; border: 1px solid #3A3A3A; border-radius: 5px; padding: 15px; }
    h1, h2, h3 { color: #00FFAA !important; }
    </style>
    """, unsafe_allow_html=True)

# Prompt structure to instruct the LLM
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context}  
Answer:
"""

PDF_STORAGE_PATH = 'Z:\End To End RAG Agent With DeepSeek-R1 And Ollama\Gen-AI-With-Deep-Seek-R1\Document_store'  # Path to store uploaded PDFs
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")  # Load DeepSeek R1 embeddings model
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)  # Initialize in-memory vector DB for storing embeddings
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")  # Load DeepSeek R1 LLM for generating responses

# Save uploaded PDF file to disk
def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

# Load and parse PDF using PDFPlumber
def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

# Split raw documents into overlapping text chunks
def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

# Add chunked documents to the in-memory vector DB
def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

# Search for document chunks semantically similar to the query
def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

# Generate an LLM answer from the context documents and user query
def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])  # Concatenate all chunk texts
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)  # Format prompt
    response_chain = conversation_prompt | LANGUAGE_MODEL  # Chain the prompt and model
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})  # Get LLM response

# Streamlit UI section
st.title("📘 DocuMind AI")  # Main app title
st.markdown("### Your Intelligent Document Assistant")  # Subtitle
st.markdown("---")  # Divider

# File upload UI
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",  # Label
    type="pdf",  # Accept only PDFs
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

# If a PDF is uploaded, process it
if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)  # Save PDF
    raw_docs = load_pdf_documents(saved_path)  # Load content
    processed_chunks = chunk_documents(raw_docs)  # Chunk content
    index_documents(processed_chunks)  # Add chunks to vector DB
    
    st.success("✅ Document processed successfully! Ask your questions below.")  # Success message
    
    user_input = st.chat_input("Enter your question about the document...")  # Chat input field
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)  # Show user message
        
        with st.spinner("Analyzing document..."):  # Show loading spinner
            relevant_docs = find_related_documents(user_input)  # Search for relevant chunks
            ai_response = generate_answer(user_input, relevant_docs)  # Get LLM answer
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)  # Show assistant reply
