import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PAPERS_DIR, REPORTS_DIR
from src.download_repos import download_repo, export_repo_docs, read_repos
from src.extract_claims import extract_claims
from src.index_repo import index_repo
from src.parse_paper import extract_experiment_text
from src.retrieve import retrieve_evidence
from src.verify import verify_claim


def run_repo(repo: str, paper_path: Path | None = None, force: bool = False) -> dict[str, Any]:
    repo_name = repo.rsplit("/", 1)[-1]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{repo_name}.json"
    if report_path.is_file() and not force:
        return json.loads(report_path.read_text(encoding="utf-8"))

    paper_path = paper_path or (PAPERS_DIR / f"{repo_name}.pdf")
    paper_text = extract_experiment_text(paper_path)
    claims = extract_claims(paper_text)

    source_jsonl = Path(__file__).resolve().parents[1] / "data" / "repo_docs" / f"{repo_name}.jsonl"
    if not source_jsonl.is_file():
        export_repo_docs(repo, download_repo(repo))
    index_repo(repo, repo_name)

    checked_claims: list[dict[str, Any]] = []
    for claim in claims:
        evidence = retrieve_evidence(claim["claim"], repo, repo_name)
        verification = verify_claim(claim["claim"], claim["claim_type"], evidence)
        checked_claims.append({**claim, **verification})

    report: dict[str, Any] = {
        "repo": repo,
        "paper": paper_path.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claims": checked_claims,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check paper experiment claims against a GitHub repository.")
    parser.add_argument("--repo", help="GitHub repository in owner/name form; defaults to data/repos.txt.")
    parser.add_argument("--paper", type=Path, help="PDF path; defaults to data/papers/<repo-name>.pdf.")
    parser.add_argument("--force", action="store_true", help="Rebuild an existing report.")
    args = parser.parse_args()
    repos = [args.repo] if args.repo else read_repos()
    if not repos:
        parser.error("Specify --repo or add repositories to data/repos.txt.")
    for repo in repos:
        report = run_repo(repo, args.paper, args.force)
        print(f"{repo}: checked {len(report['claims'])} claims; report: reports/{repo.rsplit('/', 1)[-1]}.json")


if __name__ == "__main__":
    main()
