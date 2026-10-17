import json
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import REPORTS_DIR, ROOT
from src.download_repos import read_repos

app = FastAPI(title="ReproLens API", description="Check paper experiment claims against repository code.")
REPO_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
ASSETS_DIR = ROOT / "app" / "web" / "dist" / "assets"
if ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


def _report_path(repo_name: str) -> Path:
    if not REPO_NAME.fullmatch(repo_name) or repo_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid repository name.")
    return REPORTS_DIR / f"{repo_name}.json"


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
