#! /bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$(dirname "$(dirname "${SCRIPT_DIR}")")"
OPENPI_VENV_PYTHON="${OPENPI_VENV_PYTHON:-${REPO_PATH}/.venv-openpi-liberoplus/bin/python}"
DEBUG_SCRIPT="${REPO_PATH}/toolkits/eval_scripts_openpi/libero_debug_action_chunk.py"
OPENPI_VENV_ROOT="$(dirname "$(dirname "${OPENPI_VENV_PYTHON}")")"

PI05_MODEL_PATH="${PI05_MODEL_PATH:-/data/models/pi05_base}"
RLINF_DEBUG_ROOT="${RLINF_DEBUG_ROOT:-/data/rlinf_debug_runs}"
LIBERO_BENCHMARK_TYPE="${LIBERO_BENCHMARK_TYPE:-plus}"
LIBERO_TASK_SUITE_NAME="${LIBERO_TASK_SUITE_NAME:-libero_spatial}"
LIBERO_TASK_ID="${LIBERO_TASK_ID:-0}"
LIBERO_TRIAL_IDX="${LIBERO_TRIAL_IDX:-0}"
ACTION_CHUNK="${ACTION_CHUNK:-10}"
NUM_STEPS="${NUM_STEPS:-10}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
MAX_STEPS="${MAX_STEPS:-}"
STOP_AFTER_CHUNKS="${STOP_AFTER_CHUNKS:-}"
VIDEO_TEMP_SUBSAMPLE="${VIDEO_TEMP_SUBSAMPLE:-1}"
INTERACTIVE_CHUNK_EDIT="${INTERACTIVE_CHUNK_EDIT:-1}"
PAUSE_AFTER_ACTION="${PAUSE_AFTER_ACTION:-0}"
PRINT_POLICY_CHUNKS="${PRINT_POLICY_CHUNKS:-1}"
PRINT_OBSERVATION_STATE="${PRINT_OBSERVATION_STATE:-1}"
PRINT_STEP_STATE="${PRINT_STEP_STATE:-1}"
SAVE_ROLLOUT_VIDEO="${SAVE_ROLLOUT_VIDEO:-1}"
SAVE_CHUNK_VIDEOS="${SAVE_CHUNK_VIDEOS:-1}"
SAVE_LIVE_PREVIEW="${SAVE_LIVE_PREVIEW:-1}"
LIBERO_PLUS_REPO_ID="${LIBERO_PLUS_REPO_ID:-libero_plus_lerobot}"
OPENPI_REPO_ID="${OPENPI_REPO_ID:-${LIBERO_PLUS_REPO_ID}}"
NORM_STATS_DIR="${NORM_STATS_DIR:-}"
EXP_NAME="${EXP_NAME:-libero_${LIBERO_BENCHMARK_TYPE}_${LIBERO_TASK_SUITE_NAME}_task${LIBERO_TASK_ID}_trial${LIBERO_TRIAL_IDX}_debug}"

PI05_MODEL_PATH="${PI05_MODEL_PATH%/}"
RLINF_DEBUG_ROOT="${RLINF_DEBUG_ROOT%/}"

if [ ! -d "${PI05_MODEL_PATH}" ]; then
    echo "PI05_MODEL_PATH does not exist: ${PI05_MODEL_PATH}" >&2
    exit 1
fi

if [ ! -x "${OPENPI_VENV_PYTHON}" ]; then
    echo "OpenPI/libero-plus venv not found at ${OPENPI_VENV_PYTHON}" >&2
    exit 1
fi

case "${LIBERO_BENCHMARK_TYPE}" in
    standard|pro|plus)
        ;;
    *)
        echo "LIBERO_BENCHMARK_TYPE must be one of standard/pro/plus, got: ${LIBERO_BENCHMARK_TYPE}" >&2
        exit 1
        ;;
esac

case "${LIBERO_TASK_SUITE_NAME}" in
    libero_spatial|libero_object|libero_goal|libero_10|libero_90)
        ;;
    *)
        echo "Unsupported LIBERO_TASK_SUITE_NAME=${LIBERO_TASK_SUITE_NAME}" >&2
        exit 1
        ;;
esac

for flag_name in \
    INTERACTIVE_CHUNK_EDIT \
    PAUSE_AFTER_ACTION \
    PRINT_POLICY_CHUNKS \
    PRINT_OBSERVATION_STATE \
    PRINT_STEP_STATE \
    SAVE_ROLLOUT_VIDEO \
    SAVE_CHUNK_VIDEOS \
    SAVE_LIVE_PREVIEW
do
    flag_value="${!flag_name}"
    case "${flag_value}" in
        0|1)
            ;;
        *)
            echo "${flag_name} must be 0 or 1, got: ${flag_value}" >&2
            exit 1
            ;;
    esac
done

EXTRA_PYTHONPATH="${REPO_PATH}"
if [ -d "${OPENPI_VENV_ROOT}/libero" ]; then
    EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH}:${OPENPI_VENV_ROOT}/libero"
fi
if [ -d "${OPENPI_VENV_ROOT}/libero_plus" ]; then
    EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH}:${OPENPI_VENV_ROOT}/libero_plus"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${EXTRA_PYTHONPATH}:${PYTHONPATH:-}"
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
RUN_DIR="${RLINF_DEBUG_ROOT}/${RUN_STAMP}-${EXP_NAME}"
LOG_FILE="${RUN_DIR}/launcher.log"
mkdir -p "${RUN_DIR}"

CMD=(
    "${OPENPI_VENV_PYTHON}" "${DEBUG_SCRIPT}"
    --log_dir "${RUN_DIR}"
    --exp_name "${EXP_NAME}"
    --config_name "pi05_libero"
    --pretrained_path "${PI05_MODEL_PATH}"
    --repo_id "${OPENPI_REPO_ID}"
    --task_suite_name "${LIBERO_TASK_SUITE_NAME}"
    --task_id "${LIBERO_TASK_ID}"
    --trial_idx "${LIBERO_TRIAL_IDX}"
    --libero_type "${LIBERO_BENCHMARK_TYPE}"
    --action_chunk "${ACTION_CHUNK}"
    --num_steps "${NUM_STEPS}"
    --num_steps_wait "${NUM_STEPS_WAIT}"
    --video_temp_subsample "${VIDEO_TEMP_SUBSAMPLE}"
)

if [ -n "${MAX_STEPS}" ]; then
    CMD+=(--max_steps "${MAX_STEPS}")
fi

if [ -n "${STOP_AFTER_CHUNKS}" ]; then
    CMD+=(--stop_after_chunks "${STOP_AFTER_CHUNKS}")
fi

if [ -n "${NORM_STATS_DIR}" ]; then
    CMD+=(--norm_stats_dir "${NORM_STATS_DIR}")
fi

if [ -n "${LIBERO_ASSET_ROOT}" ]; then
    CMD+=(--asset_root "${LIBERO_ASSET_ROOT}")
fi

if [ "${INTERACTIVE_CHUNK_EDIT}" = "1" ]; then
    CMD+=(--interactive_chunk_edit)
fi

if [ "${PAUSE_AFTER_ACTION}" = "1" ]; then
    CMD+=(--pause_after_action)
fi

if [ "${PRINT_POLICY_CHUNKS}" = "1" ]; then
    CMD+=(--print_policy_chunks)
fi

if [ "${PRINT_OBSERVATION_STATE}" = "1" ]; then
    CMD+=(--print_observation_state)
fi

if [ "${PRINT_STEP_STATE}" = "1" ]; then
    CMD+=(--print_step_state)
fi

if [ "${SAVE_ROLLOUT_VIDEO}" = "1" ]; then
    CMD+=(--save_rollout_video)
fi

if [ "${SAVE_CHUNK_VIDEOS}" = "1" ]; then
    CMD+=(--save_chunk_videos)
fi

if [ "${SAVE_LIVE_PREVIEW}" = "1" ]; then
    CMD+=(--save_live_preview)
fi

if [ "$#" -gt 0 ]; then
    CMD+=("$@")
fi

echo "Using Python at ${OPENPI_VENV_PYTHON}"
echo "Using PI05_MODEL_PATH=${PI05_MODEL_PATH}"
echo "Using LIBERO_BENCHMARK_TYPE=${LIBERO_BENCHMARK_TYPE}"
echo "Using LIBERO_TASK_SUITE_NAME=${LIBERO_TASK_SUITE_NAME}"
echo "Using LIBERO_TASK_ID=${LIBERO_TASK_ID}"
echo "Using LIBERO_TRIAL_IDX=${LIBERO_TRIAL_IDX}"
echo "Using ACTION_CHUNK=${ACTION_CHUNK}"
echo "Using NUM_STEPS=${NUM_STEPS}"
echo "Using NUM_STEPS_WAIT=${NUM_STEPS_WAIT}"
echo "Using OPENPI_REPO_ID=${OPENPI_REPO_ID}"
echo "Using LIBERO_ASSET_ROOT=${LIBERO_ASSET_ROOT}"
printf '%q ' "${CMD[@]}" | tee "${LOG_FILE}"
echo | tee -a "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
