# 🎓 Private PhD Research Assistant

A secure, hallucination-resistant RAG (Retrieval Augmented Generation) tool designed for academic research and sensitive corporate documents.

**Live Demo:** [https://rohit-phd-assistant.streamlit.app](https://rohit-phd-assistant.streamlit.app)

## 🚀 Key Features
* **Hybrid Architecture:** Switches between **Local Privacy Mode** (Ollama/Llama 3) for sensitive data and **Cloud Mode** (Groq/Llama 3.3) for speed.
* **Multi-Document Analysis:** Upload and cross-reference multiple PDFs (SOPs, Thesis chapters) simultaneously.
* **Hallucination Guardrails:** Custom prompt engineering prevents "Context Bleeding" and incorrect acronym expansion (e.g., accurately distinguishing "DA" as Daily Allowance).
* **Smart Overlap:** Uses a 500-character chunk overlap to ensure rules and definitions aren't cut off mid-sentence.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **LLM Orchestration:** LangChain
* **Models:** Llama 3.3 (via Groq), DeepSeek-R1 (Local)
* **Vector DB:** FAISS (Facebook AI Similarity Search)
* **Processing:** PyPDF

## 📦 How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

---
*Built by [Rohit Goswami]*