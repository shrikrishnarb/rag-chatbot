import sys
from pathlib import Path
from .vectorstore import build_faiss_from_chunks, load_vectorstore, search

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python -m src.rag.vectorstore_cli build <text_file> <persist_dir>")
        print("  python -m src.rag.vectorstore_cli search <persist_dir> <query> [k]")
        sys.exit(1)

    command = sys.argv[1]
    
    if command == "build":
        text_file = Path(sys.argv[2])
        persist_dir = Path(sys.argv[3])
        with open(text_file, "r", encoding="utf-8") as f:
            raw_text = f.read()
        # Optional: use same chunking logic as Day 3
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(raw_text)
        build_faiss_from_chunks(chunks, persist_dir)
    
    elif command == "search":
        persist_dir = Path(sys.argv[2])
        query = sys.argv[3]
        k = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        vectorstore = load_vectorstore(persist_dir)
        results = search(vectorstore, query, k)
        for chunk, score in results:
            print(f"[{score:.4f}] {chunk[:200]}...")  # show first 200 chars

if __name__ == "__main__":
    main()
