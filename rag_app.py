import streamlit as st
import os
import httpx
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from pdfminer.high_level import extract_text
import tempfile
import tiktoken

# Disable SSL warnings (for dev environments only)
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
_old_get = requests.get
def _patched_get(*args, **kwargs):
    kwargs['verify'] = False
    return _old_get(*args, **kwargs)
requests.get = _patched_get

# Token caching for tiktoken
tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir

# Disable SSL verification (for dev environments only)
client = httpx.Client(verify=False)

# LLM and Embedding setup
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-c1D1_Ku4y9sY3y1EVYvtIg",
    http_client=client
)

embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-c1D1_Ku4y9sY3y1EVYvtIg",
    http_client=client
)

# Initialize Chroma DB with defects_analysis.csv
def initialize_chroma_db():
    csv_file_path = "defects_analysis.csv"
    if not os.path.exists(csv_file_path):
        st.error(f"defects_analysis.csv not found at {csv_file_path}. Please ensure the file exists.")
        st.stop()
    
    try:
        loader = CSVLoader(file_path=csv_file_path)
        documents = loader.load()
        if not documents:
            st.error("The defects_analysis.csv file is empty or unreadable.")
            st.stop()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Create or load Chroma DB
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="./chroma_index"
        )
        vectordb.persist()
        return vectordb
    except Exception as e:
        st.error(f"Failed to initialize Chroma DB with defects_analysis.csv: {e}")
        st.stop()

# Load Chroma DB at startup
with st.spinner("Initializing Chroma DB with defects_analysis.csv..."):
    vectordb = initialize_chroma_db()

# Streamlit UI
st.set_page_config(page_title="RAG Defect Analysis")
st.title("RAG-powered Defect Analysis Report Generator")

# File uploader for PDF or CSV
upload_file = st.file_uploader("Upload a PDF or CSV file with defect data", type=["pdf", "csv"])

if upload_file:
    # Determine file type and save to temporary file
    file_extension = os.path.splitext(upload_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(upload_file.read())
        temp_file_path = temp_file.name

    # Step 1: Load and process uploaded file
    try:
        if file_extension == ".pdf":
            # Extract text from PDF
            raw_text = extract_text(temp_file_path)
            if not raw_text.strip():
                st.error("The PDF appears to be empty or unreadable.")
                st.stop()
            # Create a document-like structure for PDF text
            from langchain.docstore.document import Document
            uploaded_documents = [Document(page_content=raw_text, metadata={"source": upload_file.name})]
        elif file_extension == ".csv":
            # Load CSV using CSVLoader
            loader = CSVLoader(file_path=temp_file_path)
            uploaded_documents = loader.load()
            if not uploaded_documents:
                st.error("The uploaded CSV appears to be empty or unreadable.")
                st.stop()
        else:
            st.error("Unsupported file type. Please upload a PDF or CSV file.")
            st.stop()
    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")
        st.stop()

    # Step 2: Chunking uploaded data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    uploaded_chunks = text_splitter.split_documents(uploaded_documents)

    # Step 3: Add uploaded chunks to Chroma DB
    with st.spinner("Indexing uploaded data..."):
        try:
            # Add new chunks to existing Chroma DB
            vectordb.add_documents(
                documents=uploaded_chunks,
                embedding=embedding_model
            )
            vectordb.persist()
        except Exception as e:
            st.error(f"Failed to add uploaded data to vector store: {e}")
            st.stop()

    # Step 4: Set up RAG QA Chain
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # Step 5: Define prompt for defect analysis
    analysis_prompt = """
You are an expert quality control analyst. You have access to a reference dataset (defects_analysis.csv) containing historical defect data and a newly uploaded defect dataset (from a PDF or CSV file). Analyze the uploaded dataset by comparing it with the reference dataset (defects_analysis.csv) to generate a detailed defect analysis report. The report should include:

1. **Summary of Defects**: Summarize the types of defects, their frequency, and distribution across machines, production lines, or other relevant categories in the uploaded dataset. Highlight similarities or differences with the reference dataset (defects_analysis.csv).
2. **Root Cause Analysis (RCA)**: Identify potential root causes for the most frequent defects in the uploaded dataset, considering patterns in machines, production lines, dates, or other factors. Cross-reference with historical patterns from defects_analysis.csv to validate causes.
3. **Action Plan**: Provide specific, actionable recommendations for engineers to reduce or eliminate these defects, leveraging insights from both the uploaded dataset and defects_analysis.csv.

**Context**:
{context}

Please provide a concise and professional report based on the defect data.
"""

    # Step 6: Run RAG chain for analysis
    with st.spinner("Generating defect analysis report..."):
        try:
            result = rag_chain.invoke(analysis_prompt)
            report = result['result']
            source_documents = result['source_documents']
        except Exception as e:
            st.error(f"Failed to generate report: {e}")
            st.stop()

    # Step 7: Display results
    st.subheader("Defect Analysis Report")
    st.write(report)

    # Optional: Display source documents
    with st.expander("Show Source Data"):
        st.write("Retrieved data used for analysis (from defects_analysis.csv and uploaded file):")
        for doc in source_documents:
            st.write(doc.page_content)

    # Clean up temporary file
    os.remove(temp_file_path)

# Notes:
# - The report quality depends on the LLM (model: azure_ai/genailab-maas-DeepSeek-V3-0324) and the data in defects_analysis.csv and the uploaded file.
# - Ensure defects_analysis.csv is in the same directory as this script.
# - The Chroma index is stored in ./chroma_index for reuse.
# - For PDFs, the report depends on the quality of text extraction; for CSVs, it assumes structured data like DefectID, DefectType, Machine, ProductionLine, and DefectDate.