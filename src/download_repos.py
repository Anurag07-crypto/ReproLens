import json
from collections.abc import Iterator
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from langchain_community.document_loaders import DirectoryLoader, GitLoader, TextLoader
from langchain_core.documents import Document

from src.config import DATA_DIR, REPO_DOCS_DIR

IGNORED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build"}
ALLOWED_SUFFIXES = {
    ".py", ".ipynb", ".yaml", ".yml", ".toml", ".json", ".md", ".txt", ".sh",
    ".bash", ".c", ".cc", ".cpp", ".h", ".hpp", ".java", ".js", ".ts", ".tsx",
    ".jsx", ".go", ".rs", ".R", ".r", ".ini", ".cfg",
}
MAX_FILE_BYTES = 1_000_000


def _is_loadable_file(file_path: str | Path, repo_path: Path) -> bool:
    path = Path(file_path)
    try:
        relative = path.resolve().relative_to(repo_path.resolve())
    except ValueError:
        return False
    if (
        not path.is_file()
        or path.suffix.lower() not in ALLOWED_SUFFIXES
        or any(part in IGNORED_DIRS for part in relative.parts)
    ):
        return False
    return path.stat().st_size <= MAX_FILE_BYTES


class _RepositoryTextLoader(TextLoader):
    def __init__(self, file_path: str | Path, repo_path: Path):
        super().__init__(file_path, encoding="utf-8")
        self.repo_path = repo_path

    def lazy_load(self) -> Iterator[Document]:
        path = Path(self.file_path)
        if not _is_loadable_file(path, self.repo_path):
            return
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return
        relative = path.resolve().relative_to(self.repo_path.resolve())
        yield Document(page_content=content, metadata={"source": relative.as_posix()})


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
    loader = GitLoader(
        repo_path=str(destination),
        clone_url=f"https://github.com/{repo}.git",
        branch="HEAD",
        file_filter=lambda file_path: _is_loadable_file(file_path, destination),
    )
    for _ in loader.lazy_load():
        pass
    return destination


def export_repo_docs(repo: str, repo_path: Path) -> Path:
    name = _safe_name(repo)
    REPO_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPO_DOCS_DIR / f"{name}.jsonl"
    loader = DirectoryLoader(
        str(repo_path),
        glob="**/*",
        recursive=True,
        loader_cls=_RepositoryTextLoader,
        loader_kwargs={"repo_path": repo_path},
        exclude=[f"**/{directory}/**" for directory in IGNORED_DIRS],
    )
    documents = sorted(loader.load(), key=lambda document: document.metadata["source"])
    with output.open("w", encoding="utf-8") as stream:
        for document in documents:
            stream.write(
                json.dumps(
                    {
                        "path": document.metadata["source"],
                        "content": document.page_content,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return output


def main() -> None:
    for repo in read_repos():
        export = export_repo_docs(repo, download_repo(repo))
        print(f"Saved {repo} source files to {export}")


if __name__ == "__main__":
    main()
