import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Load extracted text from day2
with open("data/samples/sample.json", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.split_text(raw_text)
print(f"✅ Split into {len(chunks)} chunks")

# Generate embeddings using free HuggingFace model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store embeddings in FAISS (vector database)
vectorstore = FAISS.from_texts(chunks, embeddings)

# Save vectorstore locally
with open("data/vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)

print("✅ Embeddings created and vectorstore saved to data/vectorstore.pkl")
