# src/console_qa.py
import os
from .rag.retriever import load_vectorstore, retrieve
from .rag.llm import get_llm
from .rag.chains import get_conversational_chain

def main():
    # --- Load LLM ---
    llm = get_llm()  # automatically uses env LLM_PROVIDER

    # --- Load FAISS vectorstore ---
    persist_dir = os.path.join("data", "vectorstore")  # your repo path
    vectorstore = load_vectorstore(persist_dir)

    # --- Wrap retriever ---
    def retriever(query, k=5):
        return retrieve(query, vectorstore, k=k)

    # --- Build LangChain conversational chain ---
    conv_chain = get_conversational_chain(retriever, llm)

    print("Console RAG Q&A. Type 'exit' to quit.")
    chat_history = []

    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break

        result = conv_chain({"question": question, "chat_history": chat_history})
        answer = result["answer"]
        sources = [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]

        print(f"Answer: {answer}")
        print(f"Sources: {sources}\n")
        chat_history.append((question, answer))

if __name__ == "__main__":
    main()
