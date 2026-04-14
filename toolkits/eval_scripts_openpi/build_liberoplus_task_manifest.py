#!/usr/bin/env python3

import argparse
import json
import pathlib
from collections import defaultdict


ALL_LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
LIBERO_PLUS_TAXONOMY_MAP = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}


def _resolve_default_classification_path() -> pathlib.Path:
    import liberoplus.liberoplus as libero_pkg

    return pathlib.Path(libero_pkg.__file__).resolve().parent / "benchmark" / "task_classification.json"


def _select_evenly_spaced(task_ids: list[int], count: int) -> list[int]:
    if count <= 0 or not task_ids:
        return []
    if count >= len(task_ids):
        return list(task_ids)

    selected = []
    for index in range(count):
        position = round((index + 0.5) * len(task_ids) / count - 0.5)
        selected.append(task_ids[position])
    return sorted(set(selected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classification_path",
        type=pathlib.Path,
        default=None,
        help="Path to LIBERO-plus task_classification.json. Defaults to the installed liberoplus package.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output manifest path.",
    )
    parser.add_argument(
        "--per_taxonomy_per_suite",
        type=int,
        default=7,
        help="How many task ids to sample per (suite, taxonomy).",
    )
    args = parser.parse_args()

    classification_path = args.classification_path or _resolve_default_classification_path()
    raw_data = json.loads(classification_path.read_text())
    task_ids_by_suite: dict[str, list[int]] = {}
    detailed_selection = {}
    total_tasks = 0

    for suite_name in ALL_LIBERO_SUITES:
        entries = raw_data[suite_name]
        grouped = defaultdict(list)
        for entry in entries:
            taxonomy = LIBERO_PLUS_TAXONOMY_MAP.get(entry["category"], entry["category"])
            grouped[taxonomy].append(
                {
                    "task_id": int(entry["id"]) - 1,
                    "task_name": entry["name"],
                    "category": entry["category"],
                    "taxonomy": taxonomy,
                }
            )

        selected_entries = []
        for taxonomy in sorted(grouped):
            candidates = sorted(grouped[taxonomy], key=lambda item: item["task_id"])
            selected_ids = _select_evenly_spaced(
                [item["task_id"] for item in candidates], args.per_taxonomy_per_suite
            )
            selected_lookup = set(selected_ids)
            selected_entries.extend(
                item for item in candidates if item["task_id"] in selected_lookup
            )

        selected_entries = sorted(selected_entries, key=lambda item: item["task_id"])
        detailed_selection[suite_name] = selected_entries
        task_ids_by_suite[suite_name] = [item["task_id"] for item in selected_entries]
        total_tasks += len(selected_entries)

    manifest = {
        "libero_type": "plus",
        "classification_path": str(classification_path),
        "selection_strategy": "evenly_spaced_per_suite_and_taxonomy",
        "per_taxonomy_per_suite": args.per_taxonomy_per_suite,
        "approx_total_tasks": total_tasks,
        "task_ids_by_suite": task_ids_by_suite,
        "detailed_selection": detailed_selection,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "approx_total_tasks": total_tasks}, indent=2))


if __name__ == "__main__":
    main()
