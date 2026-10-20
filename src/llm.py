import json
import os
from functools import lru_cache
from typing import Any

from fastembed import TextEmbedding
from langchain_groq import ChatGroq

from src.config import EMBEDDING_CACHE_DIR, EMBEDDING_MODEL, LLM_MODEL


@lru_cache(maxsize=1)
def _embedding_model() -> TextEmbedding:
    return TextEmbedding(
        model_name=EMBEDDING_MODEL,
        cache_dir=str(EMBEDDING_CACHE_DIR),
    )


def complete_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required. Set it in the environment or .env file.")
    llm = ChatGroq(
        api_key=api_key,
        model=LLM_MODEL,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    response = llm.invoke(
        [("system", system_prompt), ("human", user_prompt)]
    )
    try:
        value = json.loads(response.content)
    except (AttributeError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("The LLM returned an invalid JSON response.") from error
    if not isinstance(value, dict):
        raise RuntimeError("The LLM response must be a JSON object.")
    return value


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = [
        [float(value) for value in vector]
        for vector in _embedding_model().embed(texts)
    ]
    if len(vectors) != len(texts):
        raise RuntimeError("The embedding model returned a different number of vectors than inputs.")
    return vectors
