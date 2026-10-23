# ReproLens

**Does the code actually support the paper?** ReproLens extracts checkable experiment claims from a research paper, searches the associated GitHub repository, and reports whether the retrieved source supports, contradicts, or does not establish each claim. Verdicts include reasoning and citations to exact source lines.

ReproLens checks claims against code; it does **not** rerun experiments or verify reproducibility.

## How it works

1. Extract text from experiment and appendix sections in a PDF.
2. Ask an LLM to identify up to six explicit claims (hyperparameters, architecture, datasets, training, and preprocessing).
3. Clone or read the corresponding repository source files, preserving file paths and line numbers.
4. Embed line-numbered chunks into a repo-tagged ChromaDB collection.
5. Retrieve candidates with semantic search and keyword overlap, filtered to the selected repository; combine their rankings with reciprocal-rank fusion.
6. Ask the LLM to classify each claim using only retrieved evidence. Citations are checked against the actual retrieved file and line ranges before they are included in a report.
7. Display the report in the web UI and compare predictions to hand-labelled claims with the evaluation scripts.

## Requirements

- Python 3.11+
- Git on `PATH`
- Node.js 18+ and npm (for the web UI)
- A Groq API key for chat completions; embeddings run locally with a free model

## Setup

From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` in `.env` for chat completions. Embeddings use the free `BAAI/bge-small-en-v1.5` model locally through FastEmbed; it is downloaded the first time embeddings are generated and does not require an embeddings API key. `LLM_MODEL` selects the Groq model. Do not commit `.env`.

Add PDFs as `data/papers/<repo-name>.pdf` and one GitHub identifier per line to `data/repos.txt`:

```text
thuml/iTransformer
```

Or provide a repository and paper path directly:

```powershell
python -m src.pipeline --repo thuml/iTransformer --paper data/papers/iTransformer.pdf
```

The pipeline writes `reports/iTransformer.json` and caches completed reports. Use `--force` to rerun. It clones public repositories to `data/repos/` and exports eligible source files to `data/repo_docs/`; those downloaded assets are git-ignored.

## Run the app

Start the API in one terminal:

```powershell
uvicorn app.api.main:app --reload
```

Start the React UI in another:

```powershell
Set-Location app/web
npm install
npm run dev
```

Open the Vite URL shown in the terminal (usually `http://localhost:5173`). The API is available at `http://127.0.0.1:8000/docs`. To serve a production UI from FastAPI, build it with `npm run build`; then open the API root.

The API exposes `GET /repos` and `GET /report/<repo-name>` for repository availability and report JSON. The same endpoints are also available under `/api/` for the web UI.

### Deploy the frontend to Vercel and API to Render

Deploy `app/web` as the Vercel project root with the build command `npm run build` and output directory `dist`. The production frontend defaults to `https://reprolens-backend.onrender.com`; set the Vercel environment variable `VITE_API_BASE_URL` if your API uses a different URL.

The API allows requests from Vercel-hosted `*.vercel.app` origins. If you use a custom frontend domain, add it to Render's environment as `CORS_ALLOWED_ORIGINS` (comma-separated if there are multiple), for example `https://reprolens.example.com`. Redeploy after changing environment variables.

## Evaluation

Add hand-verified rows to `data/labels.csv` with columns `repo,claim_type,claim,ground_truth,evidence_file,rationale`. Determine each ground truth by inspecting the repository independently, not by copying the pipeline's verdict or citations. `ground_truth` must be `supported`, `contradicted`, or `not_found`; include source paths and a short rationale for each manual label. Populate `data/splits.json` with repository identifiers in separate `dev` and `test` arrays; keep repositories disjoint between splits. Use repository identifiers in labels and splits exactly as the pipeline report's `repo` value, for example `thuml/iTransformer`.

Run the retrieval-free baseline against existing reports and evaluate both systems:

```powershell
python -m eval.baseline
python -m eval.evaluate
```

The baseline asks the same claim about general model knowledge without code retrieval. It is not a human-quality oracle. Evaluation reports per-verdict one-vs-rest precision/recall and accuracy overall and by claim type. Empty splits are shown as `n/a`; **do not present an empty test split as measured test performance**. The generated `eval/results.md` is the place to put real dev/test numbers for a demo or submission.

## Limitations

- PDF section detection relies on extractable text and recognizable experiment/appendix headings; scanned PDFs need OCR first.
- The source-file allowlist and size limit intentionally skip binaries and many generated artifacts. A missing snippet means “not found,” not that the repository lacks an implementation.
- Search is limited to the checked-out repository snapshot and can miss dynamic configuration, indirect defaults, or external dependencies.
- LLM judgments can be wrong. Evidence is constrained to retrieved source and citation ranges are validated, but users should inspect cited lines.
- The baseline uses the same LLM without retrieval; compare it on an untouched test split and report label coverage.
- Cloned repos, PDFs, ChromaDB index files, and credentials are not committed.

## License

MIT. See [LICENSE](LICENSE).
