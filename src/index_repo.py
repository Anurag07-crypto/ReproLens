import json
from pathlib import Path

import chromadb
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CHROMA_DIR, CHUNK_LINES, CHUNK_OVERLAP, REPO_DOCS_DIR
from src.llm import embed_texts


def chunk_lines(path: str, content: str) -> list[dict[str, object]]:
    lines = content.splitlines()
    if not lines:
        return []
    chunks: list[dict[str, object]] = []
    start = 0
    while start < len(lines):
        end = min(start + CHUNK_LINES, len(lines))
        body = "\n".join(lines[start:end])
        chunks.append(
            {
                "id": f"{path}:{start + 1}-{end}",
                "path": path,
                "start_line": start + 1,
                "end_line": end,
                "text": body,
            }
        )
        if end == len(lines):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def load_repo_chunks(repo_name: str) -> list[dict[str, object]]:
    path = REPO_DOCS_DIR / f"{repo_name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Repository source index not found: {path}")
    chunks: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(record.get("path"), str) or not isinstance(record.get("content"), str):
                raise ValueError(f"Invalid source record at {path}:{line_number}")
            chunks.extend(chunk_lines(record["path"], record["content"]))
    if not chunks:
        raise ValueError(f"No indexable source files found in {path}")
    return chunks


def collection_name() -> str:
    return "reprolens-source-bge-small-en-v1-5"


def index_repo(repo: str, repo_name: str | None = None) -> int:
    repo_name = repo_name or repo.rsplit("/", 1)[-1]
    chunks = load_repo_chunks(repo_name)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        collection_name(),
        metadata={"hnsw:space": "cosine"},
    )
    embeddings: list[list[float]] = []
    for offset in range(0, len(chunks), 64):
        batch = chunks[offset:offset + 64]
        embeddings.extend(embed_texts([str(chunk["text"]) for chunk in batch]))
    collection.delete(where={"repo": repo})

    for offset in range(0, len(chunks), 64):
        batch = chunks[offset:offset + 64]
        collection.add(
            ids=[f"{repo}:{chunk['id']}" for chunk in batch],
            embeddings=embeddings[offset:offset + len(batch)],
            documents=[str(chunk["text"]) for chunk in batch],
            metadatas=[
                {
                    "repo": repo,
                    "path": chunk["path"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                }
                for chunk in batch
            ],
        )
    return len(chunks)


def indexed_repos() -> set[str]:
    if not Path(CHROMA_DIR).exists():
        return set()
    collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_or_create_collection(
        collection_name(),
        metadata={"hnsw:space": "cosine"},
    )
    repos: set[str] = set()
    for offset in range(0, collection.count(), 1000):
        rows = collection.get(include=["metadatas"], limit=1000, offset=offset)
        repos.update(
            item["repo"]
            for item in rows["metadatas"]
            if isinstance(item, dict) and isinstance(item.get("repo"), str)
        )
    return repos
