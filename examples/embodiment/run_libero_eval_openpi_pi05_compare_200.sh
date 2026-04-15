#! /bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$(dirname "$(dirname "${SCRIPT_DIR}")")"
OPENPI_VENV_PYTHON="${OPENPI_VENV_PYTHON:-${REPO_PATH}/.venv-openpi-liberoplus/bin/python}"
BASE_LAUNCHER="${SCRIPT_DIR}/run_liberoplus_eval_openpi_pi05_base.sh"
SUMMARY_SCRIPT="${REPO_PATH}/toolkits/eval_scripts_openpi/summarize_liberoplus_compare.py"
GPU_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

COMPARE_ROOT="${COMPARE_ROOT:-/data/rlinf_eval_runs_compare}"
COMPARE_STAMP="${COMPARE_STAMP:-$(date +'%Y%m%d-%H:%M:%S')}"
COMPARE_NAME="${COMPARE_NAME:-libero_pi05_compare_200}"
COMPARE_DIR="${COMPARE_ROOT}/${COMPARE_STAMP}-${COMPARE_NAME}"
RUNS_ROOT="${COMPARE_DIR}/runs"
MANIFEST_PATH="${COMPARE_DIR}/task_manifest.json"
RUNS_JSON="${COMPARE_DIR}/runs.json"
SUMMARY_JSON="${COMPARE_DIR}/comparison_summary.json"
SUMMARY_MD="${COMPARE_DIR}/comparison_summary.md"
WANDB_PROJECT="${WANDB_PROJECT:-libero-eval}"
WANDB_GROUP="${WANDB_GROUP:-${COMPARE_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"
NUM_SAVE_VIDEOS="${NUM_SAVE_VIDEOS:-200}"
COMMON_NORM_STATS_DIR="${COMMON_NORM_STATS_DIR:-/data/models/pi05_base/physical-intelligence/libero}"
OPENPI_REPO_ID="${OPENPI_REPO_ID:-physical-intelligence/libero}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-5}"
TRIAL_SAMPLING="${TRIAL_SAMPLING:-evenly_spaced}"

mkdir -p "${COMPARE_DIR}" "${RUNS_ROOT}"

python - <<PY
import json
from pathlib import Path

compare_dir = Path("${COMPARE_DIR}")
suite_names = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
manifest = {
    "libero_type": "standard",
    "selection_strategy": "all_tasks_evenly_spaced_trials",
    "suite_names": suite_names,
    "tasks_per_suite": 10,
    "approx_total_tasks": 40,
    "num_trials_per_task": int("${NUM_TRIALS_PER_TASK}"),
    "trial_sampling": "${TRIAL_SAMPLING}",
    "approx_total_cases": 40 * int("${NUM_TRIALS_PER_TASK}"),
}
Path("${MANIFEST_PATH}").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

runs = [
    {
        "model_name": "pi05_base",
        "model_path": "/data/models/pi05_base",
        "gpu": "${GPU_DEVICE}",
        "exp_name": "pi05_base_libero_200",
        "run_dir": str(compare_dir / "runs" / "${COMPARE_STAMP}-pi05_base_libero_200"),
    },
    {
        "model_name": "pi05_libero",
        "model_path": "/data/models/pi05_libero",
        "gpu": "${GPU_DEVICE}",
        "exp_name": "pi05_libero_libero_200",
        "run_dir": str(compare_dir / "runs" / "${COMPARE_STAMP}-pi05_libero_libero_200"),
    },
    {
        "model_name": "pi05_libero_base",
        "model_path": "/data/models/pi05_libero_base",
        "gpu": "${GPU_DEVICE}",
        "exp_name": "pi05_libero_base_libero_200",
        "run_dir": str(compare_dir / "runs" / "${COMPARE_STAMP}-pi05_libero_base_libero_200"),
    },
]
payload = {
    "compare_dir": str(compare_dir),
    "manifest_path": "${MANIFEST_PATH}",
    "runs": runs,
}
Path("${RUNS_JSON}").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

run_one() {
    local gpu="$1"
    local model_name="$2"
    local model_path="$3"
    local exp_name="$4"
    local outer_log="${COMPARE_DIR}/${model_name}.launcher.log"

    mkdir -p "${RUNS_ROOT}/${COMPARE_STAMP}-${exp_name}"

    CUDA_VISIBLE_DEVICES="${gpu}" \
    RLINF_EVAL_ROOT="${RUNS_ROOT}" \
    RUN_STAMP="${COMPARE_STAMP}" \
    EXP_NAME="${exp_name}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_GROUP="${WANDB_GROUP}" \
    WANDB_NAME="${COMPARE_NAME}_${model_name}" \
    WANDB_MODE="${WANDB_MODE}" \
    LIBERO_BENCHMARK_TYPE="standard" \
    LIBERO_TASK_SUITE_NAME="all" \
    NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
    TRIAL_SAMPLING="${TRIAL_SAMPLING}" \
    NUM_SAVE_VIDEOS="${NUM_SAVE_VIDEOS}" \
    SAVE_ONLY_FAILURES="1" \
    PI05_MODEL_PATH="${model_path}" \
    OPENPI_REPO_ID="${OPENPI_REPO_ID}" \
    NORM_STATS_DIR="${COMMON_NORM_STATS_DIR}" \
    bash "${BASE_LAUNCHER}" \
        > "${outer_log}" 2>&1
}

echo "Using CUDA_VISIBLE_DEVICES=${GPU_DEVICE} for all compare runs"
run_one "${GPU_DEVICE}" "pi05_base" "/data/models/pi05_base" "pi05_base_libero_200"
run_one "${GPU_DEVICE}" "pi05_libero" "/data/models/pi05_libero" "pi05_libero_libero_200"
run_one "${GPU_DEVICE}" "pi05_libero_base" "/data/models/pi05_libero_base" "pi05_libero_base_libero_200"

"${OPENPI_VENV_PYTHON}" "${SUMMARY_SCRIPT}" \
    --compare_dir "${COMPARE_DIR}" \
    --output_json "${SUMMARY_JSON}" \
    --output_md "${SUMMARY_MD}"

echo "COMPARE_DIR=${COMPARE_DIR}"
echo "SUMMARY_JSON=${SUMMARY_JSON}"
echo "SUMMARY_MD=${SUMMARY_MD}"
