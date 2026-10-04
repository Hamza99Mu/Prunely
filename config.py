import os
from pathlib import Path


APP_NAME = "Prunely"

MODEL_NAME = "openai/gpt-oss-120b"

SOURCE_SHEET_ID = (
    "1qNU_LzN4mV9hZbEY9rxKzZbpm2QmPWf9"
)

SOURCE_XLSX_URL = (
    f"https://docs.google.com/spreadsheets/d/"
    f"{SOURCE_SHEET_ID}/export?format=xlsx"
)


EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

CHROMA_PATH = str(
    Path("/tmp") / "prunely_chroma"
)

COLLECTION_NAME = (
    "prunely_plant_knowledge"
)

TOP_K = 6

CHUNK_SIZE = 1800

CHUNK_OVERLAP = 250


def get_groq_api_key() -> str:

    try:

        import streamlit as st

        key = st.secrets.get(
            "GROQ_API_KEY",
            "",
        )

    except Exception:

        key = ""

    key = key or os.getenv(
        "GROQ_API_KEY",
        "",
    )

    if not key:

        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add it to Streamlit Cloud Secrets."
        )

    return key
