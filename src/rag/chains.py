from langchain.chains import RetrievalQA, ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

def get_retrieval_qa_chain(retriever, llm, prompt_template=None):
    """
    Returns a LangChain RetrievalQA chain that returns source documents.
    """
    if prompt_template is None:
        prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template="Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
        )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # or "map_reduce" later
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
    return qa_chain

def get_conversational_chain(retriever, llm):
    """
    Returns a ConversationalRetrievalChain with buffer memory for multi-turn Q&A
    """
    combine_docs_chain = load_qa_chain(llm, chain_type="stuff")

    conv_chain = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=llm,        # now explicitly
        combine_docs_chain=combine_docs_chain,
        memory=None,
        return_source_documents=True
    )
    return conv_chain

def get_summarizer_chain(llm):
    """
    Returns a simple map-reduce summarization chain
    """
    from langchain.chains.summarize import load_summarize_chain
    from langchain.docstore.document import Document

    def summarize_docs(docs: list[Document]):
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        return chain.run(docs)

    return summarize_docs
