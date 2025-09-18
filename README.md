# RAG Chatbot — Chat with Your Documents

A **Retrieval‑Augmented Generation (RAG)** app built with **Streamlit** and **LangChain**.  
Upload your documents, index them locally, and ask questions with **context‑aware answers and citations**.

---

## Features
- **Multi‑provider LLMs**: OpenAI (GPT‑3.5/4), Google Gemini, Hugging Face (Mistral)
- **Document types**: PDF, TXT, CSV, DOCX
- **Local vector DB**: Chroma (persistent storage)
- **Retrieval strategies**:
  - Vectorstore retriever
  - Contextual Compression (split → dedupe → semantic filter → reorder)
  - Cohere Reranker (optional)
- **Conversational memory** and **multilingual answers**
- **Transparent citations** for every response

---

## Quickstart

### 1) Clone & Setup
```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Prepare folders
```bash
mkdir -p data/tmp data/vector_stores
```

### 3) Run the app
```bash
streamlit run rag_app.py
```

## How it works
-  Upload documents → split into chunks → embed with OpenAI/Google/HF
-  Store embeddings in Chroma (local persistent DB)
-  Retrieve relevant chunks using your chosen strategy
-  Generate answers with context + conversation history

## Requirements
-  Python 3.12+
-  Streamlit, LangChain, Chroma
-  API keys for OpenAI / Google / Hugging Face (and Cohere if using reranker)