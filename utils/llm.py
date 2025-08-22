import os

# ---- For Development
from dotenv import load_dotenv

load_dotenv()
# ----

from langchain.chat_models import init_chat_model


def get_llm(model_name: str, model_provider: str):
    """Initialize and return a chat model based on the provided model name."""

    return init_chat_model(model_name, model_provider=model_provider)


def get_llm_with_structured_output(llm, structured_output_class):
    return llm.with_structured_output(structured_output_class)
