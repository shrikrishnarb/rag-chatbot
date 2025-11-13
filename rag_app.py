#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG Chatbot — Latest Models + Free Demo Mode
--------------------------------------------
Adds:
- OpenAI GPT-5 family (gpt-5, gpt-5-mini, gpt-5-nano)
- Google Gemini 2.5 family (gemini-2.5-pro / flash / flash-lite)
- Hugging Face Serverless via HuggingFaceHub
- Ollama (Local, FREE) for LLM + Embeddings (llama3 + nomic-embed-text)
- Demo Mode (no API keys): auto-switch to Ollama

Notes:
- GPT-5 availability/pricing: OpenAI docs.  (See citations in chat)
- Gemini 2.5 models + naming: Google AI Studio & Vertex AI docs. (See citations)
- Hugging Face serverless providers + free tier: HF docs/news. (See citations)
"""

from __future__ import annotations
import os, glob, json
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Tuple

import streamlit as st

# Must be the first Streamlit call:
st.set_page_config(page_title="Chat with your documents", page_icon="📄", layout="wide")

# ---- LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Hugging Face (cloud/serverless) and local embeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import (
    HuggingFaceInferenceAPIEmbeddings,
    HuggingFaceEmbeddings,      # local sentence-transformers
    OllamaEmbeddings            # local (FREE)
)
# Ollama (local, FREE) LLM
from langchain_community.chat_models import ChatOllama

from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory

from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, DirectoryLoader, CSVLoader, Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline, EmbeddingsFilter, CohereRerank
)
from langchain_community.document_transformers import EmbeddingsRedundantFilter, LongContextReorder

# Optional keyword retriever for Hybrid search
try:
    from langchain_community.retrievers import BM25Retriever
    HAS_BM25 = True
except Exception:
    HAS_BM25 = False

APP_TITLE = "🤖 RAG Chatbot — Latest & Free Options"
TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
VSTORE_ROOT = Path(__file__).resolve().parent.joinpath("data", "vector_stores")

LANG_WELCOME = {
    "english": "How can I help you today?",
    "japanese": "今日はどのようにお手伝いできますか？",
    "french": "Comment puis-je vous aider aujourd’hui ?",
    "spanish": "¿Cómo puedo ayudarle hoy?",
    "german": "Wie kann ich Ihnen heute helfen?",
    "russian": "Чем я могу помочь вам сегодня?",
    "chinese": "我今天能帮你什么？",
    "arabic": "كيف يمكنني مساعدتك اليوم؟",
    "portuguese": "Como posso ajudá-lo hoje?",
    "italian": "Come posso assisterti اليوم؟",
}

# Providers and model menus
LLM_PROVIDERS = [
    "OpenAI (GPT‑5)",
    "Google Gemini (2.5)",
    "Hugging Face (Serverless)",
    "Ollama (Local, FREE)",
]

OPENAI_MODELS = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-4o", "gpt-3.5-turbo"]
GEMINI_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
HF_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "deepseek-ai/DeepSeek-R1",  # reasoning model (may be slower)
]
OLLAMA_MODELS = ["llama3:latest", "llama3.1:latest", "mistral:latest"]

RETRIEVER_TYPES = ["Vectorstore", "Contextual compression", "Cohere reranker"]

def _ensure_dirs() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    VSTORE_ROOT.mkdir(parents=True, exist_ok=True)

def _delete_tmp() -> None:
    for f in glob.glob(str(TMP_DIR / "*")):
        try: os.remove(f)
        except Exception: pass

def _badge(text: str, color: str = "#4c78ff") -> str:
    return f"<span style='background:{color};color:white;padding:2px 6px;border-radius:6px;font-size:0.8rem'>{text}</span>"

@dataclass
class BuildConfig:
    provider: str
    model: str
    temperature: float
    top_p: float
    retriever_type: str
    use_hybrid: bool
    chunk_size: int
    chunk_overlap: int
    cohere_api_key: str = ""
    demo_mode: bool = False   # NEW

# -----------------------------
# Document I/O & filters
# -----------------------------
def save_uploads(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> None:
    if not files: return
    for f in files:
        out = TMP_DIR / f.name
        with open(out, "wb") as w:
            w.write(f.read())

def _filter_nonempty_docs(docs: List) -> List:
    good = []
    for d in docs:
        txt = (getattr(d, "page_content", "") or "").strip()
        if txt:
            good.append(d)
    return good

def load_documents() -> List:
    docs = []
    if any(TMP_DIR.glob("**/*.txt")):
        docs.extend(DirectoryLoader(TMP_DIR.as_posix(), glob="**/*.txt", loader_cls=TextLoader, show_progress=True).load())
    if any(TMP_DIR.glob("**/*.pdf")):
        docs.extend(DirectoryLoader(TMP_DIR.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True).load())
    if any(TMP_DIR.glob("**/*.csv")):
        docs.extend(DirectoryLoader(TMP_DIR.as_posix(), glob="**/*.csv", loader_cls=CSVLoader, show_progress=True, loader_kwargs={"encoding": "utf8"}).load())
    if any(TMP_DIR.glob("**/*.docx")):
        docs.extend(DirectoryLoader(TMP_DIR.as_posix(), glob="**/*.docx", loader_cls=Docx2txtLoader, show_progress=True).load())
    return _filter_nonempty_docs(docs)

def split_docs(docs: List, chunk_size: int = 1600, chunk_overlap: int = 200) -> Tuple[List, List]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if (c.page_content or "").strip()]
    return chunks, docs

# -----------------------------
# Embeddings & LLMs builders
# -----------------------------
def _embedding_healthcheck(embeddings) -> bool:
    try:
        _ = embeddings.embed_documents(["healthcheck"])
        return True
    except Exception as e:
        st.error(f"Embeddings failed: {e}")
        return False

def build_embeddings(cfg: BuildConfig, openai_key: str, google_key: str, hf_key: str):
    # Demo Mode: use local embeddings (Ollama or sentence-transformers) without keys
    if cfg.demo_mode or cfg.provider == "Ollama (Local, FREE)":
        try:
            return OllamaEmbeddings(model="nomic-embed-text")  # ollama pull nomic-embed-text
        except Exception:
            # Fallback to local sentence-transformers (requires `sentence-transformers` installed)
            try:
                return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            except Exception as e:
                raise RuntimeError(
                    "No local embeddings available. Install Ollama and pull 'nomic-embed-text', "
                    "or `pip install sentence-transformers`."
                ) from e

    # Cloud providers
    if cfg.provider.startswith("OpenAI"):
        return OpenAIEmbeddings(api_key=openai_key)
    if cfg.provider.startswith("Google"):
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_key)
    if cfg.provider.startswith("Hugging Face"):
        # Serverless (HF Inference API) requires HF token; otherwise prefer demo mode
        if not hf_key:
            raise RuntimeError("Hugging Face serverless requires HF token. Use Demo Mode or Ollama for free local testing.")
        return HuggingFaceInferenceAPIEmbeddings(api_key=hf_key, model_name="thenlper/gte-large")
    raise ValueError("Unsupported provider for embeddings")

def build_llms(cfg: BuildConfig, openai_key: str, google_key: str, hf_key: str):
    # Demo Mode / Local Ollama
    if cfg.demo_mode or cfg.provider == "Ollama (Local, FREE)":
        # Ensure a local model exists (e.g., `ollama pull llama3`)
        return ChatOllama(model=cfg.model or "llama3:latest", temperature=cfg.temperature), ChatOllama(model=cfg.model or "llama3:latest", temperature=cfg.temperature)

    # Cloud providers
    if cfg.provider.startswith("OpenAI"):
        q_llm = ChatOpenAI(api_key=openai_key, model=cfg.model, temperature=0.1)
        a_llm = ChatOpenAI(api_key=openai_key, model=cfg.model, temperature=cfg.temperature, model_kwargs={"top_p": cfg.top_p})
        return q_llm, a_llm
    if cfg.provider.startswith("Google"):
        q_llm = ChatGoogleGenerativeAI(google_api_key=google_key, model=cfg.model, temperature=0.1, convert_system_message_to_human=True)
        a_llm = ChatGoogleGenerativeAI(google_api_key=google_key, model=cfg.model, temperature=cfg.temperature, top_p=cfg.top_p, convert_system_message_to_human=True)
        return q_llm, a_llm
    if cfg.provider.startswith("Hugging Face"):
        if not hf_key:
            raise RuntimeError("Hugging Face Serverless requires HF token. Use Demo Mode for local testing without keys.")
        common = {"huggingfacehub_api_token": hf_key}
        q_llm = HuggingFaceHub(repo_id=cfg.model, model_kwargs={"temperature": 0.1, "top_p": 0.95, "do_sample": True, "max_new_tokens": 1024}, **common)
        a_llm = HuggingFaceHub(repo_id=cfg.model, model_kwargs={"temperature": cfg.temperature, "top_p": cfg.top_p, "do_sample": True, "max_new_tokens": 1024}, **common)
        return q_llm, a_llm
    raise ValueError("Unsupported provider for LLMs")

# -----------------------------
# Retrievers
# -----------------------------
def base_vector_retriever(vs: Chroma, search_type: str = "similarity",
                          k: int = 16, score_threshold: Optional[float] = None):
    kwargs = {}
    if k is not None: kwargs["k"] = k
    if score_threshold is not None: kwargs["score_threshold"] = score_threshold
    return vs.as_retriever(search_type=search_type, search_kwargs=kwargs)

def build_compression_retriever(emb, base, k: int = 20, char_chunk_size: int = 500,
                                similarity_threshold: Optional[float] = None):
    splitter = CharacterTextSplitter(chunk_size=char_chunk_size, chunk_overlap=0, separator=". ")
    redundant = EmbeddingsRedundantFilter(embeddings=emb)
    relevant = EmbeddingsFilter(embeddings=emb, k=k, similarity_threshold=similarity_threshold)
    reorder = LongContextReorder()
    compressor = DocumentCompressorPipeline(transformers=[splitter, redundant, relevant, reorder])
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)

def build_cohere_retriever(base, cohere_api_key: str, model: str = "rerank-multilingual-v2.0", top_n: int = 10):
    compressor = CohereRerank(cohere_api_key=cohere_api_key, model=model, top_n=top_n)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)

def maybe_hybrid_retriever(vector_retriever, chunks: Optional[List] = None):
    if not HAS_BM25 or not chunks:
        return vector_retriever
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 8
    return EnsembleRetriever(retrievers=[bm25, vector_retriever], weights=[0.5, 0.5])

# -----------------------------
# Chains & Memory
# -----------------------------
def _answer_template(language: str = "english") -> str:
    return dedent(f"""
    You are a precise, helpful assistant. Use ONLY the <context> to answer.
    If context is insufficient, say you don't know. Answer in the specified language.

    <context>
    {{chat_history}}
    {{context}}
    </context>

    Question: {{question}}
    Language: {language}
    """)

def build_memory(model_name: str, openai_key: str) -> ConversationBufferMemory:
    if model_name == "gpt-3.5-turbo":
        mem = ConversationSummaryBufferMemory(
            max_token_limit=1024,
            llm=ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_key, temperature=0.1),
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question",
        )
    else:
        mem = ConversationBufferMemory(return_messages=True, memory_key="chat_history", output_key="answer", input_key="question")
    return mem

def build_chain(retriever, cfg: BuildConfig, openai_key: str, google_key: str, hf_key: str, language: str):
    condense = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=dedent("""
        Given the conversation so far and a follow-up question, rephrase it as a standalone question.
        Preserve the original language.

        Chat history:\n{chat_history}
        Follow-up: {question}
        Standalone question:
        """),
    )
    answer_prompt = ChatPromptTemplate.from_template(_answer_template(language=language))
    memory = build_memory(cfg.model, openai_key)
    q_llm, a_llm = build_llms(cfg, openai_key, google_key, hf_key)
    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        condense_question_llm=q_llm,
        llm=a_llm,
        memory=memory,
        retriever=retriever,
        chain_type="stuff",
        verbose=False,
        return_source_documents=True,
    )
    return chain, memory

# -----------------------------
# Vector store lifecycle
# -----------------------------
def create_vectorstore(chunks: List, embeddings, persist_dir: str) -> Chroma:
    if not chunks:
        raise ValueError("No non-empty chunks to embed. Check your uploaded files or chunker settings.")
    if not _embedding_healthcheck(embeddings):
        raise ValueError("Embeddings health check failed. Verify provider or local setup.")
    vs = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
    return vs

def open_vectorstore(persist_dir: str, embeddings) -> Chroma:
    return Chroma(embedding_function=embeddings, persist_directory=persist_dir)

# -----------------------------
# UI Blocks
# -----------------------------
def ui_sidebar() -> BuildConfig:
    st.sidebar.caption("Latest models, plus FREE local demo mode with Ollama.")
    provider = st.sidebar.selectbox("LLM Provider", LLM_PROVIDERS, index=0)
    st.session_state["PROVIDER"] = provider

    # Demo Mode toggle (forces no-key local usage via Ollama)
    demo_mode = st.sidebar.checkbox("Demo Mode (no API keys, use Ollama local models)", value=False)

    # API keys (only relevant when Demo Mode = False and provider != Ollama)
    openai_key = ""
    google_key = ""
    hf_key     = ""
    cohere_key = st.sidebar.text_input("Cohere API Key (only if using reranker)", type="password", value="")

    # Models menu
    if provider.startswith("OpenAI"):
        model = st.sidebar.selectbox("Model", OPENAI_MODELS, index=0)
        if not demo_mode:
            openai_key = st.sidebar.text_input("OpenAI API Key", type="password", value="")
    elif provider.startswith("Google"):
        model = st.sidebar.selectbox("Model", GEMINI_MODELS, index=0)
        if not demo_mode:
            google_key = st.sidebar.text_input("Google API Key", type="password", value="")
    elif provider.startswith("Hugging Face"):
        model = st.sidebar.selectbox("Model", HF_MODELS, index=0)
        if not demo_mode:
            hf_key = st.sidebar.text_input("Hugging Face Token", type="password", value="")
    else:  # Ollama local
        model = st.sidebar.selectbox("Model (local)", OLLAMA_MODELS, index=0)
        st.sidebar.info("No API keys required. Make sure Ollama is installed and you have pulled the models:\n\n"
                        "• `ollama pull llama3`\n• `ollama pull nomic-embed-text`")

    st.sidebar.markdown("---")
    temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.4, 0.05)
    top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.95, 0.05)

    st.sidebar.markdown("---")
    retriever_type = st.sidebar.radio("Retriever", RETRIEVER_TYPES, index=1)
    use_hybrid = st.sidebar.checkbox("Hybrid search (BM25 + dense)", value=True,
                                     help="Enabled when creating a new store in-session and BM25 is available.")

    st.sidebar.markdown("---")
    chunk_size = st.sidebar.slider("Chunk size", 256, 2400, 1600, 64)
    chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 400, 200, 10)

    # Store keys
    st.session_state["OPENAI_API_KEY"] = openai_key
    st.session_state["GOOGLE_API_KEY"] = google_key
    st.session_state["HF_API_TOKEN"]  = hf_key

    return BuildConfig(
        provider=provider,
        model=model,
        temperature=temperature,
        top_p=top_p,
        retriever_type=retriever_type,
        use_hybrid=use_hybrid,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        cohere_api_key=cohere_key,
        demo_mode=demo_mode,
    )

def ui_status_panel():
    with st.sidebar.expander("Status", expanded=False):
        st.write("Vectorstore:", st.session_state.get("VSTORE_NAME", "—"))
        st.write("Retriever:", st.session_state.get("RETRIEVER_KIND", "—"))
        msgs = len(st.session_state.get("messages", []))
        st.write("Chat turns:", msgs)

def ui_header(language: str):
    st.title(APP_TITLE)
    st.caption("Upload docs → build a vector store → chat with context-aware answers.")
    c1, c2, c3 = st.columns([6, 2, 2])
    with c1:
        st.subheader("Chat with your data")
    with c2:
        if st.button("Clear Chat"):
            clear_history(language)
    with c3:
        if st.session_state.get("messages"):
            st.download_button(
                "⬇️ Export chat",
                data=json.dumps(st.session_state.get("messages"), ensure_ascii=False, indent=2),
                file_name="chat_export.json",
                mime="application/json"
            )

# -----------------------------
# Chat helpers
# -----------------------------
def clear_history(language: str):
    st.session_state["messages"] = [{
        "role": "assistant",
        "content": LANG_WELCOME.get(language, LANG_WELCOME["english"])
    }]
    try: st.session_state.get("MEMORY").clear()
    except Exception: pass

def show_sources(resp):
    with st.expander("**Source documents**"):
        docs_txt = []
        for d in resp.get("source_documents", []):
            page = f" (Page: {d.metadata.get('page')})" if "page" in d.metadata else ""
            docs_txt.append(f"**Source:** {d.metadata.get('source','?')}{page}\n\n{d.page_content}\n")
        st.markdown("\n\n".join(docs_txt) if docs_txt else "No source documents returned.")

# -----------------------------
# Main App
# -----------------------------
def main():
    _ensure_dirs()
    with st.sidebar:
        st.markdown(_badge("Latest + Free"), unsafe_allow_html=True)

    cfg = ui_sidebar()
    ui_status_panel()

    language = st.selectbox("Assistant language", list(LANG_WELCOME.keys()), index=0)
    tab_build, tab_open, tab_chat = st.tabs(["Create a new Vectorstore", "Open an existing Vectorstore", "Chat"])

    # --- Build vector store tab ---
    with tab_build:
        st.markdown("### 1) Upload documents")
        uploads = st.file_uploader("Select PDFs, TXT, DOCX, or CSV", accept_multiple_files=True,
                                   type=["pdf", "txt", "docx", "csv"])
        if uploads:
            _delete_tmp()
            save_uploads(uploads)
            st.success(f"Uploaded {len(uploads)} file(s).")

        st.markdown("### 2) Name your vector store")
        vname = st.text_input("Vectorstore name", placeholder="my_knowledge_base")

        st.markdown("### 3) Build")
        if st.button("Create Vectorstore", type="primary"):
            # Key requirements only if not in Demo Mode and not using Ollama
            if not cfg.demo_mode and not cfg.provider.startswith("Ollama"):
                if cfg.provider.startswith("OpenAI") and not st.session_state.get("OPENAI_API_KEY"):
                    st.warning("Please provide your OpenAI API key.")
                    st.stop()
                if cfg.provider.startswith("Google") and not st.session_state.get("GOOGLE_API_KEY"):
                    st.warning("Please provide your Google API key.")
                    st.stop()
                if cfg.provider.startswith("Hugging Face") and not st.session_state.get("HF_API_TOKEN"):
                    st.warning("Please provide your Hugging Face token or enable Demo Mode.")
                    st.stop()

            if not uploads:
                st.warning("Please upload at least one document.")
                st.stop()
            if not vname:
                st.warning("Please provide a vectorstore name.")
                st.stop()

            with st.spinner("Loading, chunking, embedding, and persisting…"):
                docs = load_documents()
                if not docs:
                    st.error("All loaded documents had empty text. If PDFs are scanned, apply OCR first.")
                    st.stop()

                st.info(f"Loaded {len(docs)} document(s) after filtering empties.")
                chunks, _ = split_docs(docs, cfg.chunk_size, cfg.chunk_overlap)
                if not chunks:
                    st.error("No non-empty chunks produced. Adjust chunk size/overlap or verify your docs.")
                    st.stop()
                st.info(f"Produced {len(chunks)} chunks.")

                emb = build_embeddings(cfg,
                                       st.session_state.get("OPENAI_API_KEY",""),
                                       st.session_state.get("GOOGLE_API_KEY",""),
                                       st.session_state.get("HF_API_TOKEN",""))
                if not _embedding_healthcheck(emb):
                    st.stop()

                persist_dir = (VSTORE_ROOT / vname).as_posix()
                vs = create_vectorstore(chunks, emb, persist_dir)

                base_ret = base_vector_retriever(vs, search_type="similarity", k=16)
                if cfg.retriever_type == "Vectorstore":
                    ret = base_ret
                elif cfg.retriever_type == "Contextual compression":
                    ret = build_compression_retriever(emb, base_ret, k=20)
                else:
                    if not cfg.cohere_api_key:
                        st.warning("Cohere key missing. Falling back to Vectorstore retriever.")
                        ret = base_ret
                    else:
                        ret = build_cohere_retriever(base_ret, cfg.cohere_api_key, model="rerank-multilingual-v2.0", top_n=10)

                if cfg.use_hybrid:
                    ret = maybe_hybrid_retriever(ret, chunks)

                q_llm, a_llm = build_llms(cfg,
                                          st.session_state.get("OPENAI_API_KEY",""),
                                          st.session_state.get("GOOGLE_API_KEY",""),
                                          st.session_state.get("HF_API_TOKEN",""))
                chain, memory = build_chain(ret, cfg,
                                            st.session_state.get("OPENAI_API_KEY",""),
                                            st.session_state.get("GOOGLE_API_KEY",""),
                                            st.session_state.get("HF_API_TOKEN",""),
                                            language)

                st.session_state["VSTORE_NAME"] = vname
                st.session_state["VSTORE_PATH"] = persist_dir
                st.session_state["VECTORSTORE"] = vs
                st.session_state["RETRIEVER"] = ret
                st.session_state["CHAIN"] = chain
                st.session_state["MEMORY"] = memory
                st.session_state["RETRIEVER_KIND"] = cfg.retriever_type + (" + Hybrid" if (cfg.use_hybrid and HAS_BM25) else "")
                st.session_state["PROVIDER"] = cfg.provider

                clear_history(language)
                st.success(f"Vectorstore **{vname}** created and ready!")

    # --- Open existing vector store tab ---
    with tab_open:
        st.markdown("### Open a saved vector store")
        root = st.text_input("Vectorstore root", value=VSTORE_ROOT.as_posix())
        root_path = Path(root)
        if not root_path.exists():
            st.info(f"Root folder not found: {root_path}")
        else:
            names = sorted([p.name for p in root_path.iterdir() if p.is_dir()])
            sel = st.selectbox("Available vectorstores", options=names)
            if st.button("Load selected vectorstore"):
                # Key requirements only if not in Demo Mode and not using Ollama
                if not cfg.demo_mode and not cfg.provider.startswith("Ollama"):
                    if cfg.provider.startswith("OpenAI") and not st.session_state.get("OPENAI_API_KEY"):
                        st.warning("Please provide your OpenAI API key.")
                        st.stop()
                    if cfg.provider.startswith("Google") and not st.session_state.get("GOOGLE_API_KEY"):
                        st.warning("Please provide your Google API key.")
                        st.stop()
                    if cfg.provider.startswith("Hugging Face") and not st.session_state.get("HF_API_TOKEN"):
                        st.warning("Please provide your Hugging Face token or enable Demo Mode.")
                        st.stop()

                with st.spinner("Loading vectorstore…"):
                    emb = build_embeddings(cfg,
                                           st.session_state.get("OPENAI_API_KEY",""),
                                           st.session_state.get("GOOGLE_API_KEY",""),
                                           st.session_state.get("HF_API_TOKEN",""))
                    if not _embedding_healthcheck(emb):
                        st.stop()
                    vs = open_vectorstore((root_path / sel).as_posix(), emb)
                    base_ret = base_vector_retriever(vs, search_type="similarity", k=16)
                    if cfg.retriever_type == "Vectorstore":
                        ret = base_ret
                    elif cfg.retriever_type == "Contextual compression":
                        ret = build_compression_retriever(emb, base_ret, k=20)
                    else:
                        if not cfg.cohere_api_key:
                            st.warning("Cohere key missing. Falling back to Vectorstore retriever.")
                            ret = base_ret
                        else:
                            ret = build_cohere_retriever(base_ret, cfg.cohere_api_key, model="rerank-multilingual-v2.0", top_n=10)

                    q_llm, a_llm = build_llms(cfg,
                                              st.session_state.get("OPENAI_API_KEY",""),
                                              st.session_state.get("GOOGLE_API_KEY",""),
                                              st.session_state.get("HF_API_TOKEN",""))
                    chain, memory = build_chain(ret, cfg,
                                                st.session_state.get("OPENAI_API_KEY",""),
                                                st.session_state.get("GOOGLE_API_KEY",""),
                                                st.session_state.get("HF_API_TOKEN",""),
                                                language)

                    st.session_state["VSTORE_NAME"] = sel
                    st.session_state["VSTORE_PATH"] = (root_path / sel).as_posix()
                    st.session_state["VECTORSTORE"] = vs
                    st.session_state["RETRIEVER"] = ret
                    st.session_state["CHAIN"] = chain
                    st.session_state["MEMORY"] = memory
                    st.session_state["RETRIEVER_KIND"] = cfg.retriever_type
                    st.session_state["PROVIDER"] = cfg.provider

                    clear_history(language)
                    st.success(f"**{sel}** loaded.")

    # --- Chat tab (display only; input is at root level) ---
    with tab_chat:
        ui_header(language)
        if "messages" not in st.session_state:
            clear_history(language)
        for msg in st.session_state.get("messages", []):
            st.chat_message(msg["role"]).write(msg["content"])

    # Root-level chat input (required by Streamlit)
    prompt = None
    prompt = st.chat_input("Ask a question about your documents…")
    if prompt is not None:
        if "CHAIN" not in st.session_state:
            st.info("Please create or load a vector store first.")
        else:
            with st.spinner("Thinking…"):
                try:
                    resp = st.session_state["CHAIN"].invoke({"question": prompt})
                    answer = resp.get("answer", "")
                    provider = st.session_state.get("PROVIDER")
                    if provider and provider.startswith("Hugging Face") and "\nAnswer: " in answer:
                        answer = answer.split("\nAnswer: ", 1)[-1]

                    st.session_state.setdefault("messages", [])
                    st.session_state["messages"].append({"role": "user", "content": prompt})
                    st.session_state["messages"].append({"role": "assistant", "content": answer})

                    st.chat_message("user").write(prompt)
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        show_sources(resp)
                except Exception as e:
                    st.warning(str(e))

if __name__ == "__main__":
    main()