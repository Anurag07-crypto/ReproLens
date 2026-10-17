import json
import subprocess
from pathlib import Path

from src.config import DATA_DIR, REPO_DOCS_DIR

IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build"}
ALLOWED_SUFFIXES = {
    ".py", ".ipynb", ".yaml", ".yml", ".toml", ".json", ".md", ".txt", ".sh",
    ".bash", ".c", ".cc", ".cpp", ".h", ".hpp", ".java", ".js", ".ts", ".tsx",
    ".jsx", ".go", ".rs", ".R", ".r", ".ini", ".cfg",
}
MAX_FILE_BYTES = 1_000_000


def read_repos() -> list[str]:
    path = DATA_DIR / "repos.txt"
    if not path.is_file():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _safe_name(repo: str) -> str:
    parts = repo.split("/")
    if (
        len(parts) != 2
        or any(part in {"", ".", ".."} for part in parts)
        or any(not all(character.isalnum() or character in "_.-" for character in part) for part in parts)
    ):
        raise ValueError(f"Invalid GitHub repository identifier: {repo}")
    return parts[1]


def download_repo(repo: str, destination: Path | None = None) -> Path:
    name = _safe_name(repo)
    destination = destination or (DATA_DIR / "repos" / name)
    if destination.exists():
        raise FileExistsError(f"Repository destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["git", "clone", "--depth", "1", f"https://github.com/{repo}.git", str(destination)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"Could not clone {repo}: {completed.stderr.strip()}")
    return destination


def export_repo_docs(repo: str, repo_path: Path) -> Path:
    name = _safe_name(repo)
    REPO_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPO_DOCS_DIR / f"{name}.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        for path in sorted(repo_path.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            relative = path.relative_to(repo_path)
            if any(part in IGNORED_DIRS for part in relative.parts) or path.stat().st_size > MAX_FILE_BYTES:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            stream.write(json.dumps({"path": relative.as_posix(), "content": content}, ensure_ascii=False) + "\n")
    return output


def main() -> None:
    for repo in read_repos():
        export = export_repo_docs(repo, download_repo(repo))
        print(f"Saved {repo} source files to {export}")


if __name__ == "__main__":
    main()
