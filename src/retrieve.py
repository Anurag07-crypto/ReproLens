import re
from collections import Counter
from pathlib import Path

import chromadb

from src.config import CHROMA_DIR, TOP_K
from src.index_repo import collection_name, load_repo_chunks
from src.llm import embed_texts

TOKEN = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def _terms(text: str) -> list[str]:
    return [word.lower() for word in TOKEN.findall(text) if len(word) > 1]


def retrieve_evidence(claim: str, repo: str, repo_name: str | None = None, top_k: int = TOP_K) -> list[dict[str, object]]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    
    if not Path(CHROMA_DIR).exists():
        raise RuntimeError("The repository index is missing. Run the analysis pipeline first.")
    
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(collection_name(),
                                                 metadata={"hnsw:space": "cosine"})
    
    semantic = collection.query(
        query_embeddings=embed_texts([claim]),
        n_results=min(top_k * 3, max(collection.count(), 1)),
        where={"repo": repo},
        include=["documents", "metadatas", "distances"],
    )
    
    semantic_items: list[dict[str, object]] = []
    
    for document, metadata, distance in zip(
        semantic["documents"][0],
        semantic["metadatas"][0],
        semantic["distances"][0],
    ):
        if document is None or metadata is None:
            continue
        semantic_items.append({"text": document, **metadata, "distance": distance})

    repo_name = repo_name or repo.rsplit("/", 1)[-1]
    lexical_items = load_repo_chunks(repo_name)
    query_counts = Counter(_terms(claim))
    
    scored: dict[tuple[str, int, int], dict[str, object]] = {}
    lexical_scores: dict[tuple[str, int, int], float] = {}
    
    for chunk in lexical_items:
        key = (str(chunk["path"]), int(chunk["start_line"]), int(chunk["end_line"]))
        counts = Counter(_terms(str(chunk["text"])))
        overlap = sum(min(count, counts[term]) for term, count in query_counts.items())
        if overlap:
            lexical_scores[key] = overlap / max(len(query_counts), 1)
        scored[key] = {**chunk, "repo": repo}
        
    for rank, item in enumerate(semantic_items, start=1):
        key = (str(item["path"]), int(item["start_line"]), int(item["end_line"]))
        scored[key] = item
        scored[key]["hybrid_score"] = 1 / (60 + rank)
        
    for key, value in lexical_scores.items():
        scored[key]["hybrid_score"] = float(scored[key].get("hybrid_score", 0)) + value
    ordered = sorted(scored.values(), key=lambda item: float(item.get("hybrid_score", 0)), reverse=True)
    
    return ordered[:top_k]
