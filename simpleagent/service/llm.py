import os
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3.5-0.8B")
DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "dummy")

def get_llm(model_name: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL):
    """Initialize and return the LLM client.
    Model and base URL can be overridden by LLM_MODEL and LLM_BASE_URL env vars.
    """
    return ChatOpenAI(
        model_name=model_name,
        openai_api_base=base_url,
        openai_api_key=DEFAULT_API_KEY,
        temperature=0,
    )
