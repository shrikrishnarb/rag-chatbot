import os
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "hf")  # default to HF

def get_llm():
    if LLM_PROVIDER == "openai":
        # Optional: OpenAI usage if key is provided
        return ChatOpenAI(temperature=0)
    elif LLM_PROVIDER == "hf":
        # Free HuggingFace model
        pipe = pipeline(
            "text-generation",
            model="gpt2",      # free small model
            max_length=200,
            do_sample=True,
            temperature=0.7
        )
        return HuggingFacePipeline(pipeline=pipe)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER={LLM_PROVIDER}")
