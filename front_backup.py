
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

# Set page config as the first Streamlit command
st.set_page_config(page_title="RAG Defect Analysis")

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

# Initialize Chroma DB with C:/vishnu_ai/defect_analysis_report.csv
def initialize_chroma_db():
    csv_file_path = "C:/vishnu_ai/defect_analysis_report.csv"
    if not os.path.exists(csv_file_path):
        st.error(f"C:/vishnu_ai/defect_analysis_report.csv not found at {csv_file_path}. Please ensure the file exists.")
        st.stop()
    
    try:
        loader = CSVLoader(file_path=csv_file_path)
        documents = loader.load()
        if not documents:
            st.error("The C:/vishnu_ai/defect_analysis_report.csv file is empty or unreadable.")
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
        st.error(f"Failed to initialize Chroma DB with C:/vishnu_ai/defect_analysis_report.csv: {e}")
        st.stop()

# Streamlit UI
st.title("RAG-powered Defect Analysis Report Generator")

# Initialize Chroma DB after set_page_config
with st.spinner("Initializing Chroma DB with C:/vishnu_ai/defect_analysis_report.csv..."):
    vectordb = initialize_chroma_db()

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
You are an expert quality control analyst. You have access to a reference dataset (C:/vishnu_ai/defect_analysis_report.csv) containing historical defect data and a newly uploaded defect dataset (from a PDF or CSV file). Your task is to analyze the uploaded dataset by comparing it with the reference dataset (C:/vishnu_ai/defect_analysis_report.csv) to generate a detailed defect analysis report. The report should include:

1. **Summary of Defects**: Summarize the types of defects, their frequency, and distribution across machines, production lines, or other relevant categories in the uploaded dataset. Highlight similarities or differences with patterns in the reference dataset (C:/vishnu_ai/defect_analysis_report.csv).
2. **Root Cause Analysis (RCA)**: Identify potential root causes for the most frequent defects in the uploaded dataset, considering patterns in machines, production lines, dates, or other factors. Cross-reference with historical patterns from C:/vishnu_ai/defect_analysis_report.csv to validate or refine the causes.
3. **Action Plan**: Provide specific, actionable recommendations for engineers to reduce or eliminate these defects, leveraging insights from both the uploaded dataset and C:/vishnu_ai/defect_analysis_report.csv.

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
        st.write("Retrieved data used for analysis (from C:/vishnu_ai/defect_analysis_report.csv and uploaded file):")
        for doc in source_documents:
            st.write(doc.page_content)

    # Clean up temporary file
    os.remove(temp_file_path)

# Notes:
# - The report quality depends on the LLM (model: azure_ai/genailab-maas-DeepSeek-V3-0324) and the data in C:/vishnu_ai/defect_analysis_report.csv and the uploaded file.
# - Ensure C:/vishnu_ai/defect_analysis_report.csv is in the same directory as this script.
# - The Chroma index is stored in ./chroma_index for reuse.
# - For PDFs, the report depends on the quality of text extraction; for CSVs, it assumes structured data like DefectID, DefectType, Machine, ProductionLine, and DefectDate.
# ```

# ### Changes Made
# 1. **Fixed `set_page_config` Error**:
#    - Moved `st.set_page_config(page_title="RAG Defect Analysis")` to the top of the script, before any other Streamlit commands.
#    - Ensured `st.spinner` for `initialize_chroma_db` is called after `st.set_page_config`.

# 2. **Retained Core Functionality**:
#    - Loads `C:/vishnu_ai/defect_analysis_report.csv` into Chroma DB at startup.
#    - Supports uploading PDF or CSV files, adding them to the Chroma DB.
#    - Analyzes uploaded data against `C:/vishnu_ai/defect_analysis_report.csv` to generate a report with summary, RCA, and action plan.

# 3. **Error Handling**:
#    - Checks for `C:/vishnu_ai/defect_analysis_report.csv` existence and validity.
#    - Handles empty or unreadable uploaded files.
#    - Catches exceptions during Chroma DB initialization and report generation.

# ### Instructions to Run
# 1. **Install Dependencies**:
#    ```bash
#    pip install streamlit langchain langchain-openai langchain-community chromadb tiktoken pdfminer.six
#    ```

# 2. **Set Up Environment**:
#    - Create a `./token` directory for `tiktoken` caching:
#      ```bash
#      mkdir token
#      ```
#    - Place `C:/vishnu_ai/defect_analysis_report.csv` in the same directory as the script (e.g., `C:\vishnu_ai`). If you have `defects_dataset.csv`, rename it to `C:/vishnu_ai/defect_analysis_report.csv` or update `csv_file_path = "C:/vishnu_ai/defect_analysis_report.csv"` to `csv_file_path = "defects_dataset.csv"`.
#    - Save the code as `defect_analysis_rag_streamlit.py` (or `front.py` if you prefer, but ensure the file name matches the command).
#    - Ensure UTF-8 encoding in your editor (e.g., VS Code).

# 3. **Run the Streamlit App**:
#    - Open a terminal in the directory (e.g., `C:\vishnu_ai`).
#    - Run:
#      ```bash
#      streamlit run defect_analysis_rag_streamlit.py
#      ```
#    - If using `front.py`, ensure the file contains the above code and run:
#      ```bash
#      streamlit run front.py
#      ```

# 4. **Using the App**:
#    - Open the app at `http://localhost:8501`.
#    - Ensure `C:/vishnu_ai/defect_analysis_report.csv` is present in the directory.
#    - Upload a PDF or CSV with defect data.
#    - View the report comparing the uploaded data with `C:/vishnu_ai/defect_analysis_report.csv`, including defect summary, RCA, and action plan.
#    - Check the "Show Source Data" expander for retrieved chunks.

# ### Example Output
# ```
# Defect Analysis Report
# Summary: The uploaded CSV shows 50 defects, with Surface Scratch (40%) and Dimensional Error (30%) on Machine_C, Line_2. Compared to C:/vishnu_ai/defect_analysis_report.csv, Machine_C has a historical 35% Surface Scratch rate.
# RCA: Surface Scratches on Machine_C are likely due to worn tooling, corroborated by historical data. Dimensional Errors suggest calibration drift, seen in both datasets.
# Action Plan: Schedule maintenance for Machine_C, recalibrate Machine_D, and implement stricter material quality checks.
# ```

# ### Troubleshooting
# - **StreamlitSetPageConfigMustBeFirstCommandError**:
#   - The error is resolved by moving `st.set_page_config` to the top. If it persists, ensure no other Streamlit commands (e.g., `st.write`, `st.spinner`) are called before it. Check for imports or logic inadvertently triggering Streamlit commands.

# - **Previous `steramlit` Error**:
#   - The typo (`steramlit` vs. `streamlit`) was addressed. Use `streamlit` in the command.

# - **Missing `C:/vishnu_ai/defect_analysis_report.csv`**:
#   - Ensure the file is in `C:\vishnu_ai`. If using `defects_dataset.csv`, rename it or update the `csv_file_path`.

# - **API Errors**:
#   - Verify the API key (`sk-c1D1_Ku4y9sY3y1EVYvtIg`) and endpoint (`https://genailab.tcs.in`). Contact your administrator if invalid.

# - **Windows-Specific**:
#   - Use Command Prompt or PowerShell. Activate your virtual environment if needed:
#     ```bash
#     C:\vishnu_ai\venv\Scripts\activate
#     ```
#   - Check the directory with `dir` to confirm `C:/vishnu_ai/defect_analysis_report.csv` and the script exist.

# - **Chroma DB**:
#   - The `./chroma_index` directory persists the DB. Delete it to reinitialize if you update `C:/vishnu_ai/defect_analysis_report.csv`.

# ### Notes
# - **CSV Structure**: Assumes `C:/vishnu_ai/defect_analysis_report.csv` and uploaded CSVs have columns like `DefectID`, `DefectType`, `Machine`, `ProductionLine`, `DefectDate`.
# - **PDF Limitations**: PDF analysis depends on text extraction quality. Scanned PDFs may require OCR (e.g., `pytesseract`).
# - **SSL Warning**: SSL verification is disabled for development. Use proper certificates in production.

# If you encounter further errors, need a sample `C:/vishnu_ai/defect_analysis_report.csv`, or want modifications (e.g., custom prompt, specific columns), please share:
# - The full error message.
# - Whether you're using `front.py` or `defect_analysis_rag_streamlit.py`.
# - Confirmation that `C:/vishnu_ai/defect_analysis_report.csv` exists in `C:\vishnu_ai`.

# I’ll provide targeted assistance to resolve any issues!
