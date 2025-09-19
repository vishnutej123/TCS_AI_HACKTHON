
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
st.set_page_config(page_title="Defect Analysis Chat", layout="wide")

# Custom CSS for ChatGPT-like look and feel
st.markdown("""
    <style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        background-color: #f5f5f5;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    .chat-history {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e6f3ff;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
        text-align: right;
    }
    .assistant-message {
        background-color: #f0f0f0;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 10px 0;
        max-width: 80%;
    }
    .prompt-box {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        padding: 10px;
        font-size: 16px;
        width: 100%;
        max-width: 700px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        margin-left: 10px;
    }
    .stButton button:hover {
        background-color: #2563eb;
    }
    .stFileUploader {
        max-width: 700px;
        margin: 10px auto;
    }
    .stSpinner {
        display: flex;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

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
    csv_file_path = "C:/vishnu_ai/defect_analysis_report.csv"  # Update to "C:/vishnu_ai/defect_analysis_report.csv" if needed
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

# Initialize session state for chat history and vector DB
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

# Default prompt for defect analysis
default_prompt = """
You are an expert quality control analyst. You have access to a reference dataset (C:/vishnu_ai/defect_analysis_report.csv) containing historical defect data and a newly uploaded defect dataset (from a CSV file). Your task is to analyze the uploaded dataset by comparing it with the reference dataset (C:/vishnu_ai/defect_analysis_report.csv) to generate a detailed defect analysis report. The report should include:

1. **Summary of Defects**: Summarize the types of defects, their frequency, and distribution across machines, production lines, or other relevant categories in the uploaded dataset. Highlight similarities or differences with patterns in the reference dataset (C:/vishnu_ai/defect_analysis_report.csv).
2. **Root Cause Analysis (RCA)**: Identify potential root causes for the most frequent defects in the uploaded dataset, considering patterns in machines, production lines, dates, or other factors. Cross-reference with historical patterns from C:/vishnu_ai/defect_analysis_report.csv to validate or refine the causes.
3. **Action Plan**: Provide specific, actionable recommendations for engineers to reduce or eliminate these defects, leveraging insights from both the uploaded dataset and C:/vishnu_ai/defect_analysis_report.csv.

**Context**:
{context}

Please provide a concise and professional report based on the defect data.
"""

# Streamlit UI
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.markdown("### Defect Analysis Chat", unsafe_allow_html=True)

# Chat history display
st.markdown('<div class="chat-history">', unsafe_allow_html=True)
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Prompt input and file uploader in the same form
st.markdown('<div class="prompt-box">', unsafe_allow_html=True)
with st.form(key="prompt_form"):
    user_prompt = st.text_area(
        "Enter your prompt (e.g., 'Analyze defects' or 'Write a story on fantasy')",
        placeholder="Type your prompt here...",
        height=100
    )
    upload_file = st.file_uploader("Upload a PDF or CSV file with defect data (optional)", type=["pdf", "csv"])
    submit_button = st.form_submit_button("Send")
st.markdown('</div>', unsafe_allow_html=True)

if submit_button and user_prompt.strip():
    # Add user prompt to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Handle prompt based on file upload
    if upload_file:
        file_extension = os.path.splitext(upload_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(upload_file.read())
            temp_file_path = temp_file.name

        # Step 1: Load and process uploaded file
        try:
            if file_extension == ".pdf":
                raw_text = extract_text(temp_file_path)
                if not raw_text.strip():
                    st.error("The PDF appears to be empty or unreadable.")
                    st.stop()
                from langchain.docstore.document import Document
                uploaded_documents = [Document(page_content=raw_text, metadata={"source": upload_file.name})]
                
                # Direct LLM query for PDF
                with st.spinner("Generating response for PDF..."):
                    try:
                        response = llm.invoke(f"{user_prompt}\n\n**Context**:\n{raw_text}").content
                    except Exception as e:
                        st.error(f"Failed to generate response: {e}")
                        st.stop()
            elif file_extension == ".csv":
                # Initialize Chroma DB if not already initialized
                if st.session_state.vectordb is None:
                    with st.spinner("Initializing Chroma DB with C:/vishnu_ai/defect_analysis_report.csv..."):
                        st.session_state.vectordb = initialize_chroma_db()

                loader = CSVLoader(file_path=temp_file_path)
                uploaded_documents = loader.load()
                if not uploaded_documents:
                    st.error("The uploaded CSV appears to be empty or unreadable.")
                    st.stop()

                # Step 2: Chunking uploaded data
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                uploaded_chunks = text_splitter.split_documents(uploaded_documents)

                # Step 3: Add uploaded chunks to Chroma DB
                with st.spinner("Indexing uploaded data..."):
                    try:
                        st.session_state.vectordb.add_documents(
                            documents=uploaded_chunks,
                            embedding=embedding_model
                        )
                        st.session_state.vectordb.persist()
                    except Exception as e:
                        st.error(f"Failed to add uploaded data to vector store: {e}")
                        st.stop()

                # Step 4: Set up RAG QA Chain
                retriever = st.session_state.vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                rag_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True
                )

                # Step 5: Use default prompt for CSV to ensure defect analysis
                analysis_prompt = default_prompt

                # Step 6: Run RAG chain for analysis
                with st.spinner("Generating defect analysis report..."):
                    try:
                        result = rag_chain.invoke(analysis_prompt)
                        response = result['result']
                        source_documents = result['source_documents']
                        if source_documents:
                            response += "\n\n**Source Data**:\n" + "\n".join([doc.page_content for doc in source_documents])
                    except Exception as e:
                        st.error(f"Failed to generate response: {e}")
                        st.stop()
            else:
                st.error("Unsupported file type. Please upload a PDF or CSV file.")
                st.stop()
        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")
            st.stop()

        # Clean up temporary file
        os.remove(temp_file_path)
    else:
        # Direct LLM query for general prompts
        with st.spinner("Generating response..."):
            try:
                response = llm.invoke(user_prompt).content
            except Exception as e:
                st.error(f"Failed to generate response: {e}")
                st.stop()

    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Rerun to update chat history
    st.rerun()

# Notes:
# - The response quality depends on the LLM (model: azure_ai/genailab-maas-DeepSeek-V3-0324) and, for CSV defect analysis, the data in C:/vishnu_ai/defect_analysis_report.csv and uploaded CSVs.
# - Ensure C:/vishnu_ai/defect_analysis_report.csv is in the same directory as this script (e.g., C:\vishnu_ai). Update csv_file_path if using C:/vishnu_ai/defect_analysis_report.csv.
# - The Chroma index is stored in ./chroma_index for reuse when initialized.
# - For PDFs, the prompt is sent directly to the LLM with extracted text as context; for CSVs, RAG is used with the default prompt for defect analysis.
# - The default prompt ensures a structured report (summary, RCA, action plan) for CSV uploads.
# ```

# ### Key Changes
# 1. **Chroma DB Initialization on CSV Upload**:
#    - Chroma DB is initialized only when a CSV file is uploaded, using `st.session_state.vectordb` to track the instance.
#    - No DB initialization for PDF uploads or general prompts without files.
#    - Loads `C:/vishnu_ai/defect_analysis_report.csv` into the DB when a CSV is uploaded, persisting in `./chroma_index`.

# 2. **File Uploader in Prompt Box**:
#    - The file uploader remains within the `prompt_form` alongside the text area.
#    - Users can submit a prompt and an optional PDF/CSV file with the “Send” button.

# 3. **CSV Processing with RAG**:
#    - When a CSV is uploaded, the app:
#      - Initializes Chroma DB with `C:/vishnu_ai/defect_analysis_report.csv`.
#      - Loads and chunks the uploaded CSV.
#      - Adds chunks to the Chroma DB.
#      - Uses the `default_prompt` to generate a defect analysis report with summary, RCA, and action plan, ignoring the user prompt to ensure structured output.
#      - Appends source documents to the response.

# 4. **PDF and General Prompt Handling**:
#    - **PDF Upload**: Extracts text and sends the user prompt with the text as context to the LLM directly (no Chroma DB).
#    - **No File**: Sends the user prompt directly to the LLM for general queries (e.g., “Write a story on fantasy”).

# 5. **ChatGPT-like UI**:
#    - Centered layout with a styled prompt box containing the text area and file uploader.
#    - Chat history displays user (blue) and assistant (gray) messages, with source data for CSV uploads.

# 6. **Fixed Previous Issues**:
#    - **StreamlitSetPageConfigMustBeFirstCommandError**: Ensured `st.set_page_config` is first.
#    - **API Key**: Standardized on `sk-c1D1_Ku4y9sY3y1EVYvtIg`.
#    - **CSV File Path**: Used `C:/vishnu_ai/defect_analysis_report.csv`. Update to `C:/vishnu_ai/defect_analysis_report.csv` if needed.
#    - **Steramlit Typo**: Instructions use `streamlit`.

# ### Instructions to Run
# 1. **Install Dependencies**:
#    ```bash
#    pip install streamlit langchain langchain-openai langchain-community chromadb tiktoken pdfminer.six
#    ```

# 2. **Set Up Environment**:
#    - Create a `./token` directory:
#      ```bash
#      mkdir token
#      ```
#    - Place `C:/vishnu_ai/defect_analysis_report.csv` in the script’s directory (e.g., `C:\vishnu_ai`). If using `defect_analysis_report.csv`, update `csv_file_path = "C:/vishnu_ai/defect_analysis_report.csv"` in the `initialize_chroma_db` function.
#    - Save the code as `defect_analysis_rag_streamlit.py` (or `front.py`) with UTF-8 encoding.

# 3. **Run the Streamlit App**:
#    - Open a terminal in the directory (e.g., `cd C:\vishnu_ai`).
#    - Run:
#      ```bash
#      streamlit run defect_analysis_rag_streamlit.py
#      ```
#    - If using `front.py`, ensure it contains the above code and run:
#      ```bash
#      streamlit run front.py
#      ```

# 4. **Using the App**:
#    - Open `http://localhost:8501`.
#    - In the prompt box:
#      - Enter a prompt (e.g., “Write a story on fantasy” or “Analyze defects”).
#      - Optionally upload a PDF or CSV file.
#    - Click “Send” to fetch the response:
#      - **No File**: Direct LLM response (e.g., a fantasy story).
#      - **PDF**: LLM response using the prompt and extracted text as context.
#      - **CSV**: RAG-based defect analysis report with summary, RCA, and action plan, comparing with `C:/vishnu_ai/defect_analysis_report.csv`.
#    - View the response in the chat history, with source data for CSV uploads.

# ### Example Usage
# - **General Prompt (No File)**:
#   - **Prompt**: “Write a story on fantasy”
#   - **Response**: “In a realm where dragons soared above emerald forests, a young mage named Elara discovered a hidden grimoire...”
# - **PDF Upload**:
#   - **Prompt**: “Summarize this defect report”
#   - **File**: `defect_report.pdf`
#   - **Response**: “The defect report indicates frequent surface scratches on Machine_C due to worn tooling...”
# - **CSV Upload**:
#   - **Prompt**: “Analyze defects” (or any prompt)
#   - **File**: `defects_data.csv`
#   - **Response**:
#     ```
#     Defect Analysis Report
#     Summary: The uploaded CSV shows 50 defects, with Surface Scratch (40%) on Machine_C, Line_2. Compared to C:/vishnu_ai/defect_analysis_report.csv, Machine_C has a historical 35% Surface Scratch rate.
#     RCA: Surface Scratches likely due to worn tooling, corroborated by historical data. Dimensional Errors suggest calibration drift.
#     Action Plan: Schedule maintenance for Machine_C, recalibrate Machine_D, enhance quality checks.
#     **Source Data**:
#     DefectID: 123, DefectType: Surface Scratch, Machine: Machine_C, ...
#     ```

# ### Troubleshooting
# - **StreamlitSetPageConfigMustBeFirstCommandError**: Resolved by ensuring `st.set_page_config` is first.
# - **Steramlit Typo**: Use `streamlit` in the command.
# - **Missing CSV**:
#   - Ensure `C:/vishnu_ai/defect_analysis_report.csv` is in `C:\vishnu_ai`. If using `defect_analysis_report.csv`, update `csv_file_path`.
# - **API Errors**:
#   - Verify the API key (`sk-c1D1_Ku4y9sY3y1EVYvtIg`) and endpoint (`https://genailab.tcs.in`).
# - **Windows-Specific**:
#   - Activate virtual environment:
#     ```bash
#     C:\vishnu_ai\venv\Scripts\activate
#     ```
#   - Check files with `dir`.
# - **CSV Issues**: Ensure the uploaded CSV and `C:/vishnu_ai/defect_analysis_report.csv` have structured data (e.g., `DefectID`, `DefectType`).

# If you need a sample `C:/vishnu_ai/defect_analysis_report.csv`, confirmation on the CSV file name (`defect_analysis_report.csv`), or additional features (e.g., clear history button), please provide details, and I’ll assist!