import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chromadb
from eval.evaluate import evaluate
from fastapi.testclient import TestClient
import src.index_repo as index_module
from src.download_repos import _safe_name
from src.index_repo import chunk_lines, index_repo
from src.parse_paper import select_experiment_sections
from src.retrieve import retrieve_evidence
from src.verify import verify_claim


class CoreTests(unittest.TestCase):
    def test_safe_repo_name_rejects_path_traversal(self):
        self.assertEqual(_safe_name("thuml/iTransformer"), "iTransformer")
        for repo in ("../outside", "owner/..", "owner\\..", "owner/repo/extra"):
            with self.subTest(repo=repo), self.assertRaises(ValueError):
                _safe_name(repo)

    def test_paper_section_includes_numbered_subsections(self):
        text = (
            "3 Method\nMethod details.\n"
            "4 Experiments\nExperimental overview.\n"
            "4.1 Implementation Details\nWe use Adam at 1e-3.\n"
            "4.2 Datasets\nDataset details.\n"
            "5 Conclusion\nConclusion text."
        )
        selected = select_experiment_sections(text)
        self.assertIn("We use Adam at 1e-3.", selected)
        self.assertIn("Dataset details.", selected)
        self.assertNotIn("Conclusion text.", selected)

    def test_chunk_lines_preserves_line_ranges_and_overlap(self):
        chunks = chunk_lines("train.py", "\n".join(f"line_{i}" for i in range(1, 90)))
        self.assertEqual((chunks[0]["start_line"], chunks[0]["end_line"]), (1, 80))
        self.assertEqual((chunks[1]["start_line"], chunks[1]["end_line"]), (69, 89))
        self.assertIn("line_69", chunks[0]["text"])
        self.assertIn("line_69", chunks[1]["text"])

    def test_index_and_search_are_scoped_to_repository(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "repo.jsonl").write_text(
                json.dumps({"path": "train.py", "content": "optimizer = Adam"}),
                encoding="utf-8",
            )
            (docs_dir / "other.jsonl").write_text(
                json.dumps({"path": "train.py", "content": "optimizer = SGD"}),
                encoding="utf-8",
            )
            chroma_dir = root / "chroma"
            chroma_dir.mkdir()
            client = chromadb.Client()
            with (
                patch.object(index_module, "REPO_DOCS_DIR", docs_dir),
                patch.object(index_module, "CHROMA_DIR", chroma_dir),
                patch.object(index_module, "embed_texts", side_effect=lambda texts: [[1.0, 0.0] for _ in texts]),
                patch.object(chromadb, "PersistentClient", return_value=client),
            ):
                self.assertEqual(index_repo("owner/repo"), 1)
                self.assertEqual(index_repo("owner/other"), 1)
            with (
                patch("src.retrieve.CHROMA_DIR", chroma_dir),
                patch("src.retrieve.embed_texts", return_value=[[1.0, 0.0]]),
                patch.object(index_module, "REPO_DOCS_DIR", docs_dir),
                patch.object(chromadb, "PersistentClient", return_value=client),
            ):
                evidence = retrieve_evidence("optimizer Adam", "owner/repo")
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["repo"], "owner/repo")
        self.assertIn("Adam", evidence[0]["text"])

    @patch("src.verify.complete_json")
    def test_verifier_accepts_only_line_ranges_from_retrieved_code(self, complete_json):
        complete_json.return_value = {
            "verdict": "supported",
            "reasoning": "The source sets this value.",
            "evidence": [{"file": "config.py", "start_line": 12, "end_line": 12}],
        }
        evidence = [
            {
                "path": "config.py",
                "start_line": 10,
                "end_line": 14,
                "text": "a\nb\nvalue = 96\nd\ne",
            }
        ]
        result = verify_claim("The window is 96.", "hyperparameter", evidence)
        self.assertEqual(result["verdict"], "supported")
        self.assertEqual(result["evidence"][0]["text"], "value = 96")

    @patch("src.verify.complete_json")
    def test_verifier_downgrades_unverifiable_citation(self, complete_json):
        complete_json.return_value = {
            "verdict": "contradicted",
            "reasoning": "The source differs.",
            "evidence": [{"file": "other.py", "start_line": 1, "end_line": 1}],
        }
        result = verify_claim(
            "The window is 96.",
            "hyperparameter",
            [{"path": "config.py", "start_line": 10, "end_line": 10, "text": "window = 48"}],
        )
        self.assertEqual(result["verdict"], "not_found")
        self.assertEqual(result["evidence"], [])

    def test_evaluation_accuracy_is_not_multiplied_by_class_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reports = root / "reports"
            reports.mkdir()
            (reports / "iTransformer.json").write_text(
                json.dumps(
                    {
                        "repo": "thuml/iTransformer",
                        "claims": [
                            {
                                "claim": "Uses Adam.",
                                "claim_type": "training",
                                "verdict": "supported",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            labels = root / "labels.csv"
            with labels.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["repo", "claim_type", "claim", "ground_truth", "evidence_file"])
                writer.writerow(["thuml/iTransformer", "training", "Uses Adam.", "supported", "train.py"])
            splits = root / "splits.json"
            splits.write_text(json.dumps({"dev": ["thuml/iTransformer"], "test": []}), encoding="utf-8")
            result = evaluate(labels, splits, reports)
        self.assertEqual(result["dev"]["retrieval"]["overall"]["accuracy"], 1.0)
        self.assertEqual(result["dev"]["retrieval"]["overall"]["count"], 1)
        self.assertIsNone(result["test"]["retrieval"]["overall"]["accuracy"])

    def test_evaluation_rejects_split_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels = root / "labels.csv"
            labels.write_text(
                "repo,claim_type,claim,ground_truth,evidence_file\n",
                encoding="utf-8",
            )
            splits = root / "splits.json"
            splits.write_text(
                json.dumps({"dev": ["owner/repo"], "test": ["owner/repo"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "both dev and test"):
                evaluate(labels, splits, root / "reports")

    def test_api_serves_only_existing_report(self):
        from app.api import main as api

        with tempfile.TemporaryDirectory() as temp:
            reports = Path(temp)
            (reports / "demo.json").write_text(
                json.dumps({"repo": "owner/demo", "claims": []}),
                encoding="utf-8",
            )
            with patch.object(api, "REPORTS_DIR", reports), patch.object(
                api, "read_repos", return_value=["owner/demo", "owner/missing"]
            ):
                client = TestClient(api.app)
                repos = client.get("/api/repos")
                report = client.get("/api/report/demo")
                missing = client.get("/api/report/missing")
        self.assertEqual(repos.status_code, 200)
        self.assertEqual(
            repos.json(),
            [
                {"name": "demo", "repo": "owner/demo", "has_report": True},
                {"name": "missing", "repo": "owner/missing", "has_report": False},
            ],
        )
        self.assertEqual(report.json()["repo"], "owner/demo")
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
