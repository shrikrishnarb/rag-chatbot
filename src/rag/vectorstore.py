import os
import pickle
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_faiss_from_chunks(chunks: List[str], persist_dir: str, use_openai=False):
    """Builds a FAISS vectorstore from text chunks and saves it to disk."""
    if use_openai:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_texts(chunks, embeddings)
    
    os.makedirs(persist_dir, exist_ok=True)
    with open(os.path.join(persist_dir, "vectorstore.pkl"), "wb") as f:
        pickle.dump(vectorstore, f)
    
    print(f"✅ FAISS vectorstore saved to {persist_dir}/vectorstore.pkl")
    return vectorstore

def load_vectorstore(persist_dir: str) -> FAISS:
    """Loads a FAISS vectorstore from disk."""
    with open(os.path.join(persist_dir, "vectorstore.pkl"), "rb") as f:
        vectorstore = pickle.load(f)
    return vectorstore

def search(vectorstore: FAISS, query: str, k: int = 5) -> List[Tuple[str, float]]:
    """Search the FAISS vectorstore and return top-k chunks with scores."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results
