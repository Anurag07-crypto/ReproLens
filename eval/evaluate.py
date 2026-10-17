import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.config import DATA_DIR, REPORTS_DIR, ROOT

VALID_VERDICTS = {"supported", "contradicted", "not_found"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_predictions(directory: Path) -> dict[tuple[str, str], dict[str, Any]]:
    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    if not directory.is_dir():
        return predictions
    for path in directory.glob("*.json"):
        report = _read_json(path)
        repo = report.get("repo", path.stem)
        for claim in report.get("claims", []):
            key = (str(repo), " ".join(str(claim.get("claim", "")).split()).casefold())
            predictions[key] = claim
    return predictions


def _metrics(rows: list[dict[str, str]], predictions: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"count": len(rows), "accuracy": None, "classes": {}}
    if not rows:
        return metrics
    matches = sum(
        predictions.get((row["repo"], " ".join(row["claim"].split()).casefold()), {}).get("verdict")
        == row["ground_truth"]
        for row in rows
    )
    for verdict in sorted(VALID_VERDICTS):
        tp = fp = fn = 0
        for row in rows:
            prediction = predictions.get((row["repo"], " ".join(row["claim"].split()).casefold()), {})
            predicted = prediction.get("verdict")
            expected = row["ground_truth"]
            tp += predicted == verdict and expected == verdict
            fp += predicted == verdict and expected != verdict
            fn += predicted != verdict and expected == verdict
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics["classes"][verdict] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        }
    metrics["accuracy"] = matches / len(rows)
    return metrics


def evaluate(labels_path: Path, splits_path: Path, reports_dir: Path = REPORTS_DIR) -> dict[str, Any]:
    with labels_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"repo", "claim_type", "claim", "ground_truth", "evidence_file"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Labels CSV must have columns: {', '.join(sorted(required))}")
        labels = []
        for row in reader:
            normalized = {key: (row.get(key) or "").strip() for key in required}
            if normalized["ground_truth"] not in VALID_VERDICTS:
                raise ValueError(f"Invalid ground_truth for claim '{normalized['claim']}': {normalized['ground_truth']}")
            labels.append(normalized)

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    dev_repos = splits.get("dev", [])
    test_repos = splits.get("test", [])
    if not isinstance(dev_repos, list) or not isinstance(test_repos, list):
        raise ValueError("The dev and test split entries must be arrays.")
    overlap = set(dev_repos) & set(test_repos)
    if overlap:
        raise ValueError(f"Repositories must not occur in both dev and test splits: {', '.join(sorted(overlap))}")
    prediction_sets = {
        "retrieval": _load_predictions(reports_dir),
        "baseline": _load_predictions(reports_dir / "baseline"),
    }
    results: dict[str, Any] = {}
    for split_name in ("dev", "test"):
        selected_repos = set(splits[split_name])
        selected = [row for row in labels if row["repo"] in selected_repos]
        results[split_name] = {}
        for system, predictions in prediction_sets.items():
            overall = _metrics(selected, predictions)
            by_type: dict[str, Any] = {}
            for claim_type in sorted({row["claim_type"] for row in selected}):
                by_type[claim_type] = _metrics(
                    [row for row in selected if row["claim_type"] == claim_type],
                    predictions,
                )
            results[split_name][system] = {"overall": overall, "by_claim_type": by_type}
    return results


def _to_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Evaluation results",
        "",
        "Metrics are computed only for manually labelled claims assigned to each split. "
        "No benchmark values are included until labels and reports are available.",
        "",
        "| Split | System | Claims | Accuracy | Supported P/R | Contradicted P/R | Not found P/R |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split, systems in results.items():
        for system, data in systems.items():
            overall = data["overall"]
            classes = overall["classes"]
            fmt = lambda name, metric: (
                "n/a" if overall["count"] == 0 else f"{classes.get(name, {}).get(metric, 0):.3f}"
            )
            accuracy = "n/a" if overall["accuracy"] is None else f"{overall['accuracy']:.3f}"
            lines.append(
                f"| {split} | {system} | {overall['count']} | {accuracy} | "
                f"{fmt('supported', 'precision')}/{fmt('supported', 'recall')} | "
                f"{fmt('contradicted', 'precision')}/{fmt('contradicted', 'recall')} | "
                f"{fmt('not_found', 'precision')}/{fmt('not_found', 'recall')} |"
            )
    lines.extend(
        [
            "",
            "Precision and recall are one-vs-rest, per verdict. Empty splits are reported as n/a.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ReproLens reports against hand-labelled claims.")
    parser.add_argument("--labels", type=Path, default=DATA_DIR / "labels.csv")
    parser.add_argument("--splits", type=Path, default=DATA_DIR / "splits.json")
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "results.md")
    args = parser.parse_args()
    results = evaluate(args.labels, args.splits, args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_to_markdown(results), encoding="utf-8")
    print(_to_markdown(results))


if __name__ == "__main__":
    main()
