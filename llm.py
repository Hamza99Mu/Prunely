from crewai import LLM

from config import MODEL_NAME, get_groq_api_key


def build_llm() -> LLM:
    return LLM(
        model=MODEL_NAME,
        custom_openai=True,
        base_url="https://api.groq.com/openai/v1",
        api_key=get_groq_api_key(),
        temperature=0.2,
        max_tokens=5000,
    )
