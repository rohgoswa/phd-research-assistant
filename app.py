import streamlit as st
import os
import platform # This is the new "Detective" tool 🕵️‍♂️
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pypdf import PdfReader

# --- PAGE CONFIG ---
st.set_page_config(page_title="PhD Research Assistant", page_icon="🎓", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stSpinner {text-align: center; color: #4CAF50;}
    div[data-testid="stSidebar"] {background-color: #f7f9fb;}
    [data-testid="stSidebar"] img {
        border-radius: 50%;
        object-fit: cover;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIG ---
with st.sidebar:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        # Note: If 'profile.png' isn't found on Cloud, this might show a broken image. 
        # Ensure you upload the image to GitHub too!
        if os.path.exists("profile.png"):
            st.image("profile.png", width=100) 
    
    st.title("Settings")
    st.write("Built by **Rohit Goswami**") 
    
    # --- INTELLIGENT MODE DETECTION ---
    os_name = platform.system()
    
    # Default options
    mode_options = ["☁️ Cloud (Speed)"]
    
    # If we are on Mac ("Darwin"), we add the Local option.
    if os_name == "Darwin":
        mode_options.insert(0, "🔒 Local (Privacy)")
    
    mode = st.radio("Processing Mode:", mode_options)
    
    if mode == "☁️ Cloud (Speed)":
        api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        st.caption("[Get Key](https://console.groq.com/keys)")
        model_id = "llama-3.3-70b-versatile" 
    else:
        # This part only runs on your Mac
        model_id = st.selectbox("Local Model:", ["llama3.2", "deepseek-r1"], index=0)
        st.info("Mode: 100% Private (Runs on your Mac)")
    
    st.divider()

# --- APP LOGIC ---
st.title("🎓 Private PhD Assistant")
st.markdown("##### Upload your thesis, manuals, or papers and chat with them.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 1. File Upload
uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")

if uploaded_file and st.session_state.vector_store is None:
    with st.spinner("🧠 Reading & Indexing Document..."):
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # OLD: chunk_overlap=200
        # NEW: chunk_overlap=500 (Preserves more context between cuts)    
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=500, length_function=len)
        chunks = text_splitter.split_text(text)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
        st.toast("Document Indexed Successfully!", icon="✅")

# 2. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle User Input
if st.session_state.vector_store:
    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                docs = st.session_state.vector_store.similarity_search(prompt, k=10)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # ... inside the "if st.session_state.vector_store:" block ...
                
                # NEW STRICT PROMPT
                rag_prompt = f"""You are a strict Technical Auditor. 
                Your job is to answer the QUESTION based ONLY on the provided CONTEXT.
                
                RULES:
                1. Do not hallucinate or make up information.
                2. Do not merge separate topics.
                3. If the answer is not in the context, say "I cannot find this information."
                4. Answer in bullet points.
                5. DO NOT expand acronyms (e.g., do not change "DA" to "Died in Service") unless the definition is explicitly written in the text. Keep them as acronyms if unsure.
                
                CONTEXT:
                {context}
                
                QUESTION: 
                {prompt}
                """
                # ... rest of the code ...
                
                try:
                    if mode == "☁️ Cloud (Speed)":
                        if not api_key:
                            st.error("Please enter API Key in sidebar!")
                            st.stop()
                        llm = ChatGroq(groq_api_key=api_key, model_name=model_id)
                        response = llm.invoke(rag_prompt)
                        result = response.content
                    else:
                        llm = ChatOllama(model=model_id)
                        response = llm.invoke(rag_prompt)
                        result = response.content
                    
                    st.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                    with st.expander("📚 View Sources"):
                        st.caption(context)
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    st.info("👆 Please upload a PDF document to start chatting.")