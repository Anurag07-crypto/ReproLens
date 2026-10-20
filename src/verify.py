from typing import Any

from src.llm import complete_json

VERDICTS = {"supported", "contradicted", "not_found"}

def verify_claim(claim: str, claim_type: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_text = "\n\n".join(
        f"Evidence {index} | {item['path']}:{item['start_line']}-{item['end_line']}\n"
        f"```text\n{item['text']}\n```"
        for index, item in enumerate(evidence, start=1)
    )
    result = complete_json(
        "You verify whether a research paper claim is supported by repository code. "
        "Use only the supplied evidence. A claim is supported only when code directly "
        "implements it; contradicted only when code clearly specifies a conflicting fact; "
        "otherwise use not_found. Do not treat absence from retrieved snippets as proof of "
        "contradiction. Return JSON with verdict, reasoning, and evidence (an array of objects "
        "with file, start_line, end_line). Cite only supplied evidence and never invent lines.",
        (
            f"CLAIM TYPE: {claim_type}\nCLAIM: {claim}\n\n"
            f"EVIDENCE:\n{evidence_text or 'No evidence retrieved.'}"
        ),
    )
    verdict = str(result.get("verdict", "not_found")).strip().lower()
    
    if verdict not in VERDICTS:
        raise RuntimeError(f"Verifier returned an invalid verdict: {verdict}")
    reasoning = result.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise RuntimeError("Verifier response is missing a reasoning string.")
    citations: list[dict[str, Any]] = []
    requested = result.get("evidence", [])
    if isinstance(requested, list):
        for citation in requested:
            if not isinstance(citation, dict):
                continue
            try:
                key = (
                    str(citation["file"]),
                    int(citation["start_line"]),
                    int(citation["end_line"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            path, start_line, end_line = key
            source = next(
                (
                    item
                    for item in evidence
                    if str(item["path"]) == path
                    and int(item["start_line"]) <= start_line <= end_line <= int(item["end_line"])
                ),
                None,
            )
            if source is None:
                continue
            source_lines = str(source["text"]).splitlines()
            start_offset = start_line - int(source["start_line"])
            end_offset = end_line - int(source["start_line"]) + 1
            citations.append(
                {
                    "file": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "text": "\n".join(source_lines[start_offset:end_offset]),
                }
            )
    if verdict != "not_found" and not citations:
        verdict = "not_found"
        reasoning = (
            "The verifier did not provide a citation matching the retrieved repository lines; "
            "this claim cannot be confirmed from checkable evidence."
        )
    return {"verdict": verdict, "reasoning": reasoning.strip(), "evidence": citations}
