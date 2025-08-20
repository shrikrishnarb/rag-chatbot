# src/rag/retriever.py
import os
import pickle
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Load FAISS vectorstore
def load_vectorstore(persist_dir: str) -> FAISS:
    with open(os.path.join(persist_dir, "vectorstore.pkl"), "rb") as f:
        vectorstore = pickle.load(f)
    return vectorstore

# Build embedding object depending on environment
def get_embeddings(use_openai=False):
    if use_openai:
        return OpenAIEmbeddings()
    else:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Query the vectorstore
def retrieve(query: str, vectorstore: FAISS, k: int = 5, use_openai=False) -> List[Tuple[str, float]]:
    embeddings = get_embeddings(use_openai)
    # FAISS already has embeddings stored, so we just do similarity search
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results
