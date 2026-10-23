import json
import re
from pathlib import Path
from typing import Any
import sys
import os
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import REPORTS_DIR, ROOT
from src.download_repos import read_repos

app = FastAPI(title="ReproLens API", description="Check paper experiment claims against repository code.")
allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", *allowed_origins],
    allow_origin_regex=r"^https://[a-zA-Z0-9.-]+\.vercel\.app$",
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)
REPO_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
ASSETS_DIR = ROOT / "app" / "web" / "dist" / "assets"
if ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


def _report_path(repo_name: str) -> Path:
    if not REPO_NAME.fullmatch(repo_name) or repo_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid repository name.")
    return REPORTS_DIR / f"{repo_name}.json"


@app.get("/repos")
@app.get("/api/repos")
def list_repositories() -> list[dict[str, Any]]:
    available = {path.stem for path in REPORTS_DIR.glob("*.json")} if REPORTS_DIR.is_dir() else set()
    names = {repo.rsplit("/", 1)[-1]: repo for repo in read_repos()}
    for name in available:
        names.setdefault(name, name)
    return [
        {"name": name, "repo": repo, "has_report": name in available}
        for name, repo in sorted(names.items())
    ]


@app.get("/report/{repo_name}")
@app.get("/api/report/{repo_name}")
def get_report(repo_name: str) -> dict[str, Any]:
    path = _report_path(repo_name)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No analysis report found for {repo_name}.")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f"Could not read report: {error}") from error
    return report


@app.get("/")
def home():
    index = ROOT / "app" / "web" / "dist" / "index.html"
    if index.is_file():
        return FileResponse(index)
    return JSONResponse(
        {
            "name": "ReproLens API",
            "docs": "/docs",
            "web": "Run the web app with `cd app/web && npm install && npm run dev`.",
        }
    )
    
if __name__ == "__main__":
    reload_enabled = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
    uvicorn.run("main:app", port=8000, host="127.0.0.1", reload=reload_enabled)