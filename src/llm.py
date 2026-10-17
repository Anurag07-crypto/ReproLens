import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.config import EMBEDDING_MODEL, LLM_BASE_URL, LLM_MODEL


def _post(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is required. Set it in the environment or .env file.")
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{LLM_BASE_URL}/{endpoint}",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach the LLM API: {error.reason}") from error


def complete_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    response = _post(
        "chat/completions",
        {
            "model": LLM_MODEL,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
    )
    try:
        content = response["choices"][0]["message"]["content"]
        value = json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("The LLM returned an invalid JSON response.") from error
    if not isinstance(value, dict):
        raise RuntimeError("The LLM response must be a JSON object.")
    return value


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = _post("embeddings", {"model": EMBEDDING_MODEL, "input": texts})
    try:
        rows = sorted(response["data"], key=lambda row: row["index"])
        vectors = [row["embedding"] for row in rows]
    except (KeyError, TypeError) as error:
        raise RuntimeError("The embedding API returned an invalid response.") from error
    if len(vectors) != len(texts):
        raise RuntimeError("The embedding API returned a different number of vectors than inputs.")
    return vectors
