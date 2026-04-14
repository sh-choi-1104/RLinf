#!/usr/bin/env python3

import argparse
import json
import pathlib


def _load_json(path: pathlib.Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _count_jsonl_lines(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _format_rate(value):
    if value is None:
        return "-"
    return f"{value * 100.0:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare_dir", type=pathlib.Path, required=True)
    parser.add_argument("--output_json", type=pathlib.Path, required=True)
    parser.add_argument("--output_md", type=pathlib.Path, required=True)
    args = parser.parse_args()

    registry_path = args.compare_dir / "runs.json"
    registry = _load_json(registry_path)
    if registry is None:
        raise FileNotFoundError(f"Missing run registry: {registry_path}")
    manifest = _load_json(pathlib.Path(registry["manifest_path"]))
    expected_total_tasks = None
    if manifest is not None:
        expected_total_tasks = manifest.get("approx_total_tasks")

    rows = []
    taxonomy_keys = set()
    suite_keys = set()
    for entry in registry["runs"]:
        run_dir = pathlib.Path(entry["run_dir"])
        exp_name = entry["exp_name"]
        summary = _load_json(run_dir / exp_name / "summary.json")
        episode_count = _count_jsonl_lines(run_dir / exp_name / "episode_results.jsonl")
        status = "running"
        if summary is not None and expected_total_tasks is not None:
            if summary.get("total_tasks_evaluated", 0) >= expected_total_tasks:
                status = "completed"
        row = {
            "model_name": entry["model_name"],
            "model_path": entry["model_path"],
            "gpu": entry["gpu"],
            "run_dir": str(run_dir),
            "status": status,
            "episodes_seen": episode_count,
            "summary": summary,
        }
        if summary is not None:
            taxonomy_keys.update(summary.get("taxonomy_success_rates", {}).keys())
            suite_keys.update(summary.get("suite_success_rates", {}).keys())
        rows.append(row)

    comparison = {
        "compare_dir": str(args.compare_dir),
        "manifest_path": registry["manifest_path"],
        "rows": rows,
    }
    args.output_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    lines = [
        "# LIBERO-plus Approx Comparison",
        "",
        f"- compare_dir: `{args.compare_dir}`",
        f"- manifest: `{registry['manifest_path']}`",
        "",
        "| Model | Status | Episodes | Total Success |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in rows:
        total_rate = None
        if row["summary"] is not None:
            total_rate = row["summary"].get("total_success_rate")
        lines.append(
            f"| {row['model_name']} | {row['status']} | {row['episodes_seen']} | {_format_rate(total_rate)} |"
        )

    if taxonomy_keys:
        lines.extend(
            [
                "",
                "## Taxonomy",
                "",
                "| Model | " + " | ".join(sorted(taxonomy_keys)) + " |",
                "| --- | " + " | ".join(["---:"] * len(taxonomy_keys)) + " |",
            ]
        )
        for row in rows:
            summary = row["summary"] or {}
            taxonomy_rates = summary.get("taxonomy_success_rates", {})
            values = [_format_rate(taxonomy_rates.get(key)) for key in sorted(taxonomy_keys)]
            lines.append(f"| {row['model_name']} | " + " | ".join(values) + " |")

    if suite_keys:
        lines.extend(
            [
                "",
                "## Suites",
                "",
                "| Model | " + " | ".join(sorted(suite_keys)) + " |",
                "| --- | " + " | ".join(["---:"] * len(suite_keys)) + " |",
            ]
        )
        for row in rows:
            summary = row["summary"] or {}
            suite_rates = summary.get("suite_success_rates", {})
            values = [_format_rate(suite_rates.get(key)) for key in sorted(suite_keys)]
            lines.append(f"| {row['model_name']} | " + " | ".join(values) + " |")

    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
