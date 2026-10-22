from typing import Any
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.llm import complete_json

CLAIM_TYPES = {
    "hyperparameter",
    "architecture",
    "dataset",
    "training",
    "preprocessing",
    "other",
}
MAX_PAPER_TEXT_CHARS = 20_000


def extract_claims(paper_text: str, count: int = 6) -> list[dict[str, str]]:
    result = complete_json(
        "Extract factual, independently checkable paper claims. "
        "Return only a JSON object with a claims array. Never invent details.",
        (
            f"Extract exactly {count} concrete claims from this experiment/appendix text. "
            "Prefer explicit hyperparameters, architecture, datasets, training setup, and "
            "preprocessing details. Each claim must include `claim` and `claim_type`, where "
            f"claim_type is one of: {', '.join(sorted(CLAIM_TYPES))}. If the source contains "
            "fewer than the requested number of checkable facts, return only the facts present; "
            "do not infer missing values.\n\nPAPER TEXT:\n" + paper_text[:MAX_PAPER_TEXT_CHARS]
        ),
    )
    claims: Any = result.get("claims")
    if not isinstance(claims, list):
        raise RuntimeError("Claim extraction response is missing the claims array.")
    cleaned: list[dict[str, str]] = []
    for item in claims:
        if not isinstance(item, dict) or not isinstance(item.get("claim"), str):
            continue
        claim = item["claim"].strip()
        if not claim:
            continue
        claim_type = str(item.get("claim_type", "other")).strip().lower()
        if claim_type not in CLAIM_TYPES:
            claim_type = "other"
        cleaned.append({"claim": claim, "claim_type": claim_type})
    if not cleaned:
        raise RuntimeError("No valid checkable claims were returned by the LLM.")
    return cleaned[:count]
