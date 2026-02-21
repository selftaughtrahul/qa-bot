import os
from dotenv import load_dotenv

load_dotenv()


def get_llm(provider: str = "groq"):
    """
    Returns the configured LLM.
    Supports: 'groq', 'huggingface', 'gemini'
    """
    if provider == "huggingface":
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch

        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        return HuggingFacePipeline(pipeline=pipe)
    

    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model_name="gemini-2.5-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")