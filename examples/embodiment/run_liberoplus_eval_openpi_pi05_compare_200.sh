#! /bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$(dirname "$(dirname "${SCRIPT_DIR}")")"
OPENPI_VENV_PYTHON="${OPENPI_VENV_PYTHON:-${REPO_PATH}/.venv-openpi-liberoplus/bin/python}"
BASE_LAUNCHER="${SCRIPT_DIR}/run_liberoplus_eval_openpi_pi05_base.sh"
MANIFEST_BUILDER="${REPO_PATH}/toolkits/eval_scripts_openpi/build_liberoplus_task_manifest.py"
SUMMARY_SCRIPT="${REPO_PATH}/toolkits/eval_scripts_openpi/summarize_liberoplus_compare.py"
GPU_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

COMPARE_ROOT="${COMPARE_ROOT:-/data/rlinf_eval_runs_compare}"
COMPARE_STAMP="${COMPARE_STAMP:-$(date +'%Y%m%d-%H:%M:%S')}"
COMPARE_NAME="${COMPARE_NAME:-liberoplus_pi05_compare_approx}"
COMPARE_DIR="${COMPARE_ROOT}/${COMPARE_STAMP}-${COMPARE_NAME}"
RUNS_ROOT="${COMPARE_DIR}/runs"
MANIFEST_PATH="${COMPARE_DIR}/task_manifest.json"
RUNS_JSON="${COMPARE_DIR}/runs.json"
SUMMARY_JSON="${COMPARE_DIR}/comparison_summary.json"
SUMMARY_MD="${COMPARE_DIR}/comparison_summary.md"
WANDB_PROJECT="${WANDB_PROJECT:-libero-plus-eval}"
WANDB_GROUP="${WANDB_GROUP:-${COMPARE_NAME}}"
WANDB_MODE="${WANDB_MODE:-online}"
NUM_SAVE_VIDEOS="${NUM_SAVE_VIDEOS:-20}"
PER_TAXONOMY_PER_SUITE="${PER_TAXONOMY_PER_SUITE:-7}"
COMMON_NORM_STATS_DIR="${COMMON_NORM_STATS_DIR:-/data/models/pi05_base/libero_plus_lerobot}"

mkdir -p "${COMPARE_DIR}" "${RUNS_ROOT}"

"${OPENPI_VENV_PYTHON}" "${MANIFEST_BUILDER}" \
    --output "${MANIFEST_PATH}" \
    --per_taxonomy_per_suite "${PER_TAXONOMY_PER_SUITE}"

python - <<PY
import json
from pathlib import Path

compare_dir = Path("${COMPARE_DIR}")
runs = [
    {
        "model_name": "pi05_base",
        "model_path": "/data/models/pi05_base",
        "gpu": "${GPU_DEVICE}",
        "exp_name": "pi05_base_approx",
        "run_dir": str(compare_dir / "runs" / "${COMPARE_STAMP}-pi05_base_approx"),
    },
    {
        "model_name": "pi05_libero",
        "model_path": "/data/models/pi05_libero",
        "gpu": "${GPU_DEVICE}",
        "exp_name": "pi05_libero_approx",
        "run_dir": str(compare_dir / "runs" / "${COMPARE_STAMP}-pi05_libero_approx"),
    },
    {
        "model_name": "pi05_libero_base",
        "model_path": "/data/models/pi05_libero_base",
        "gpu": "${GPU_DEVICE}",
        "exp_name": "pi05_libero_base_approx",
        "run_dir": str(compare_dir / "runs" / "${COMPARE_STAMP}-pi05_libero_base_approx"),
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
    LIBERO_TASK_SUITE_NAME="all" \
    NUM_TRIALS_PER_TASK="1" \
    NUM_SAVE_VIDEOS="${NUM_SAVE_VIDEOS}" \
    SAVE_ONLY_FAILURES="1" \
    PI05_MODEL_PATH="${model_path}" \
    LIBERO_PLUS_REPO_ID="libero_plus_lerobot" \
    NORM_STATS_DIR="${COMMON_NORM_STATS_DIR}" \
    bash "${BASE_LAUNCHER}" \
        --task_manifest "${MANIFEST_PATH}" \
        > "${outer_log}" 2>&1
}

echo "Using CUDA_VISIBLE_DEVICES=${GPU_DEVICE} for all compare runs"
run_one "${GPU_DEVICE}" "pi05_base" "/data/models/pi05_base" "pi05_base_approx"
run_one "${GPU_DEVICE}" "pi05_libero" "/data/models/pi05_libero" "pi05_libero_approx"
run_one "${GPU_DEVICE}" "pi05_libero_base" "/data/models/pi05_libero_base" "pi05_libero_base_approx"

"${OPENPI_VENV_PYTHON}" "${SUMMARY_SCRIPT}" \
    --compare_dir "${COMPARE_DIR}" \
    --output_json "${SUMMARY_JSON}" \
    --output_md "${SUMMARY_MD}"

echo "COMPARE_DIR=${COMPARE_DIR}"
echo "SUMMARY_JSON=${SUMMARY_JSON}"
echo "SUMMARY_MD=${SUMMARY_MD}"
