import os
from dotenv import load_dotenv

load_dotenv()


def get_llm(provider: str = "groq"):
    """
    Returns the configured LLM.
    Supports: 'groq', 'huggingface', 'gemini'
    """
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

        hf_token = os.getenv("HUGGINGFACE_API_KEY", os.getenv("HF_TOKEN"))

        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.2-1B-Instruct",
            huggingfacehub_api_token=hf_token,
            temperature=0.5,
            max_new_tokens=512,
        )

        return ChatHuggingFace(llm=llm)  # wraps it as a chat model (conversational)

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
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")