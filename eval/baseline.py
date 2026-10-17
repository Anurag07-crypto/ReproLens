import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import REPORTS_DIR
from src.llm import complete_json


def run_baseline(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = []
    for claim in report.get("claims", []):
        response = complete_json(
            "Classify the paper claim against your general knowledge, without consulting "
            "repository code. Return JSON with verdict (supported, contradicted, or not_found) "
            "and reasoning. Be conservative: without repository evidence use not_found.",
            f"Claim type: {claim['claim_type']}\nClaim: {claim['claim']}",
        )
        verdict = str(response.get("verdict", "not_found")).strip().lower()
        if verdict not in {"supported", "contradicted", "not_found"}:
            raise RuntimeError(f"Baseline returned an invalid verdict for claim '{claim['claim']}': {verdict}")
        reasoning = response.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise RuntimeError(f"Baseline response is missing reasoning for claim '{claim['claim']}'.")
        results.append(
            {
                "claim": claim["claim"],
                "claim_type": claim["claim_type"],
                "verdict": verdict,
                "reasoning": reasoning.strip(),
                "evidence": [],
            }
        )
    return {
        "repo": report["repo"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claims": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the no-retrieval baseline for existing reports.")
    parser.add_argument("--repo", help="Report slug; by default process all reports.")
    args = parser.parse_args()
    paths = [REPORTS_DIR / f"{args.repo}.json"] if args.repo else sorted(REPORTS_DIR.glob("*.json"))
    if not paths or any(not path.is_file() for path in paths):
        parser.error("No matching report found.")
    output_dir = REPORTS_DIR / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        baseline = run_baseline(path)
        target = output_dir / path.name
        target.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved no-retrieval baseline to {target}")


if __name__ == "__main__":
    main()
