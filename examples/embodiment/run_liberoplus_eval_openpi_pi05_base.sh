#! /bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$(dirname "$(dirname "${SCRIPT_DIR}")")"
OPENPI_VENV_PYTHON="${OPENPI_VENV_PYTHON:-${REPO_PATH}/.venv-openpi-liberoplus/bin/python}"
EVAL_SCRIPT="${REPO_PATH}/toolkits/eval_scripts_openpi/libero_eval.py"
OPENPI_VENV_ROOT="$(dirname "$(dirname "${OPENPI_VENV_PYTHON}")")"

PI05_MODEL_PATH="${PI05_MODEL_PATH:-/data/models/pi05_base}"
RLINF_EVAL_ROOT="${RLINF_EVAL_ROOT:-/data/rlinf_eval_runs}"
LIBERO_BENCHMARK_TYPE="${LIBERO_BENCHMARK_TYPE:-plus}"
LIBERO_TASK_SUITE_NAME="${LIBERO_TASK_SUITE_NAME:-all}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
TRIAL_SAMPLING="${TRIAL_SAMPLING:-sequential}"
NUM_SAVE_VIDEOS="${NUM_SAVE_VIDEOS:-100}"
SAVE_ONLY_FAILURES="${SAVE_ONLY_FAILURES:-1}"
TASK_OFFSET="${TASK_OFFSET:-0}"
MAX_TASKS="${MAX_TASKS:-}"
WANDB_PROJECT="${WANDB_PROJECT:-libero-plus-eval}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_NAME="${WANDB_NAME:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_VIDEOS="${WANDB_LOG_VIDEOS:-0}"
LIBERO_PLUS_REPO_ID="${LIBERO_PLUS_REPO_ID:-libero_plus_lerobot}"
OPENPI_REPO_ID="${OPENPI_REPO_ID:-${LIBERO_PLUS_REPO_ID}}"
NORM_STATS_DIR="${NORM_STATS_DIR:-}"
EXP_NAME="${EXP_NAME:-libero_${LIBERO_BENCHMARK_TYPE}_${LIBERO_TASK_SUITE_NAME}_pi05_base}"

PI05_MODEL_PATH="${PI05_MODEL_PATH%/}"
RLINF_EVAL_ROOT="${RLINF_EVAL_ROOT%/}"

if [ ! -d "${PI05_MODEL_PATH}" ]; then
    echo "PI05_MODEL_PATH does not exist: ${PI05_MODEL_PATH}" >&2
    exit 1
fi

if [ ! -x "${OPENPI_VENV_PYTHON}" ]; then
    echo "OpenPI/libero-plus venv not found at ${OPENPI_VENV_PYTHON}" >&2
    exit 1
fi

if ! [[ "${NUM_TRIALS_PER_TASK}" =~ ^[0-9]+$ ]] || [ "${NUM_TRIALS_PER_TASK}" -le 0 ]; then
    echo "NUM_TRIALS_PER_TASK must be a positive integer, got: ${NUM_TRIALS_PER_TASK}" >&2
    exit 1
fi

if ! [[ "${NUM_SAVE_VIDEOS}" =~ ^[0-9]+$ ]] || [ "${NUM_SAVE_VIDEOS}" -lt 0 ]; then
    echo "NUM_SAVE_VIDEOS must be a non-negative integer, got: ${NUM_SAVE_VIDEOS}" >&2
    exit 1
fi

if ! [[ "${TASK_OFFSET}" =~ ^[0-9]+$ ]] || [ "${TASK_OFFSET}" -lt 0 ]; then
    echo "TASK_OFFSET must be a non-negative integer, got: ${TASK_OFFSET}" >&2
    exit 1
fi

if [ -n "${MAX_TASKS}" ] && { ! [[ "${MAX_TASKS}" =~ ^[0-9]+$ ]] || [ "${MAX_TASKS}" -le 0 ]; }; then
    echo "MAX_TASKS must be a positive integer when set, got: ${MAX_TASKS}" >&2
    exit 1
fi

case "${LIBERO_TASK_SUITE_NAME}" in
    all|libero_spatial|libero_object|libero_goal|libero_10|libero_90)
        ;;
    *)
        echo "Unsupported LIBERO_TASK_SUITE_NAME=${LIBERO_TASK_SUITE_NAME}" >&2
        exit 1
        ;;
esac

case "${SAVE_ONLY_FAILURES}" in
    0|1)
        ;;
    *)
        echo "SAVE_ONLY_FAILURES must be 0 or 1, got: ${SAVE_ONLY_FAILURES}" >&2
        exit 1
        ;;
esac

case "${WANDB_LOG_VIDEOS}" in
    0|1)
        ;;
    *)
        echo "WANDB_LOG_VIDEOS must be 0 or 1, got: ${WANDB_LOG_VIDEOS}" >&2
        exit 1
        ;;
esac

case "${WANDB_MODE}" in
    online|offline|disabled)
        ;;
    *)
        echo "WANDB_MODE must be one of online/offline/disabled, got: ${WANDB_MODE}" >&2
        exit 1
        ;;
esac

case "${LIBERO_BENCHMARK_TYPE}" in
    standard|pro|plus)
        ;;
    *)
        echo "LIBERO_BENCHMARK_TYPE must be one of standard/pro/plus, got: ${LIBERO_BENCHMARK_TYPE}" >&2
        exit 1
        ;;
esac

case "${TRIAL_SAMPLING}" in
    sequential|evenly_spaced)
        ;;
    *)
        echo "TRIAL_SAMPLING must be one of sequential/evenly_spaced, got: ${TRIAL_SAMPLING}" >&2
        exit 1
        ;;
esac

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data}"
export LIBERO_TYPE="${LIBERO_BENCHMARK_TYPE}"
export RLINF_OPENPI_DISABLE_TORCH_COMPILE="${RLINF_OPENPI_DISABLE_TORCH_COMPILE:-1}"

LIBERO_ASSET_ROOT="${LIBERO_ASSET_ROOT:-}"
if [ "${LIBERO_BENCHMARK_TYPE}" = "plus" ]; then
    LIBERO_PLUS_PKG_ROOT="${OPENPI_VENV_ROOT}/libero_plus/liberoplus/liberoplus"
    LIBERO_PLUS_ASSET_SOURCE="${LIBERO_PLUS_ASSET_SOURCE:-/data/libero_plus_assets_merged}"
    LIBERO_PLUS_ASSET_TARGET="${LIBERO_PLUS_PKG_ROOT}/assets"

    if [ -L "${LIBERO_PLUS_ASSET_TARGET}" ] && [ -d "${LIBERO_PLUS_ASSET_SOURCE}" ]; then
        if [ "$(readlink -f "${LIBERO_PLUS_ASSET_TARGET}")" != "$(readlink -f "${LIBERO_PLUS_ASSET_SOURCE}")" ]; then
            rm "${LIBERO_PLUS_ASSET_TARGET}"
        fi
    fi

    if [ ! -e "${LIBERO_PLUS_ASSET_TARGET}" ] && [ -d "${LIBERO_PLUS_ASSET_SOURCE}" ]; then
        ln -s "${LIBERO_PLUS_ASSET_SOURCE}" "${LIBERO_PLUS_ASSET_TARGET}"
    fi

    if [ ! -d "${LIBERO_PLUS_ASSET_TARGET}" ]; then
        echo "LIBERO Plus assets not found." >&2
        echo "Expected either ${LIBERO_PLUS_ASSET_TARGET} or ${LIBERO_PLUS_ASSET_SOURCE}" >&2
        exit 1
    fi

    LIBERO_ASSET_ROOT="${LIBERO_PLUS_ASSET_SOURCE}"
fi

export LIBERO_ASSET_ROOT

RUN_STAMP="${RUN_STAMP:-$(date +'%Y%m%d-%H:%M:%S')}"
RUN_DIR="${RLINF_EVAL_ROOT}/${RUN_STAMP}-${EXP_NAME}"
LOG_FILE="${RUN_DIR}/launcher.log"
mkdir -p "${RUN_DIR}"

CMD=(
    "${OPENPI_VENV_PYTHON}" "${EVAL_SCRIPT}"
    --log_dir "${RUN_DIR}"
    --exp_name "${EXP_NAME}"
    --config_name "pi05_libero"
    --pretrained_path "${PI05_MODEL_PATH}"
    --repo_id "${OPENPI_REPO_ID}"
    --task_suite_name "${LIBERO_TASK_SUITE_NAME}"
    --libero_type "${LIBERO_BENCHMARK_TYPE}"
    --num_trials_per_task "${NUM_TRIALS_PER_TASK}"
    --trial_sampling "${TRIAL_SAMPLING}"
    --num_save_videos "${NUM_SAVE_VIDEOS}"
    --task_offset "${TASK_OFFSET}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_mode "${WANDB_MODE}"
)

if [ "${SAVE_ONLY_FAILURES}" = "1" ]; then
    CMD+=(--save_only_failures)
fi

if [ "${WANDB_LOG_VIDEOS}" = "1" ]; then
    CMD+=(--wandb_log_videos)
fi

if [ -n "${MAX_TASKS}" ]; then
    CMD+=(--max_tasks "${MAX_TASKS}")
fi

if [ -n "${WANDB_ENTITY}" ]; then
    CMD+=(--wandb_entity "${WANDB_ENTITY}")
fi

if [ -n "${WANDB_GROUP}" ]; then
    CMD+=(--wandb_group "${WANDB_GROUP}")
fi

if [ -n "${WANDB_NAME}" ]; then
    CMD+=(--wandb_name "${WANDB_NAME}")
fi

if [ -n "${NORM_STATS_DIR}" ]; then
    CMD+=(--norm_stats_dir "${NORM_STATS_DIR}")
fi

if [ -n "${LIBERO_ASSET_ROOT}" ]; then
    CMD+=(--asset_root "${LIBERO_ASSET_ROOT}")
fi

if [ "$#" -gt 0 ]; then
    CMD+=("$@")
fi

echo "Using Python at ${OPENPI_VENV_PYTHON}"
echo "Using PI05_MODEL_PATH=${PI05_MODEL_PATH}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID}"
echo "Using EGL_DEVICE_ID=${EGL_DEVICE_ID}"
echo "Using LIBERO_TASK_SUITE_NAME=${LIBERO_TASK_SUITE_NAME}"
echo "Using NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
echo "Using TRIAL_SAMPLING=${TRIAL_SAMPLING}"
echo "Using NUM_SAVE_VIDEOS=${NUM_SAVE_VIDEOS}"
echo "Using SAVE_ONLY_FAILURES=${SAVE_ONLY_FAILURES}"
echo "Using OPENPI_REPO_ID=${OPENPI_REPO_ID}"
echo "Using LIBERO_BENCHMARK_TYPE=${LIBERO_BENCHMARK_TYPE}"
echo "Using LIBERO_ASSET_ROOT=${LIBERO_ASSET_ROOT}"
echo "Using WANDB_PROJECT=${WANDB_PROJECT}"
echo "Using WANDB_MODE=${WANDB_MODE}"
printf '%q ' "${CMD[@]}" | tee "${LOG_FILE}"
echo | tee -a "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
