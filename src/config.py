import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
REPO_DOCS_DIR = DATA_DIR / "repo_docs"
REPORTS_DIR = ROOT / "reports"
CHROMA_DIR = ROOT / "chroma_db"
REPO_LIST = DATA_DIR / "repos.txt"
CHUNK_LINES = 80
CHUNK_OVERLAP = 12
TOP_K = 6
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
