#! /bin/bash

set -euo pipefail

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$(dirname "$(dirname "${SCRIPT_DIR}")")"

HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-}"
PI05_MODEL_PATH="${PI05_MODEL_PATH:-}"
LIBERO_REPO_ID="${LIBERO_REPO_ID:-physical-intelligence/libero}"
RUN_NORM_STATS="${RUN_NORM_STATS:-1}"
LIBERO_NORM_STATS_SOURCE="${LIBERO_NORM_STATS_SOURCE:-${REPO_PATH}/assets/pi0_libero/physical-intelligence/libero/norm_stats.json}"

HF_LEROBOT_HOME="${HF_LEROBOT_HOME%\"}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME#\"}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME%\'}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME#\'}"
LIBERO_REPO_ID="${LIBERO_REPO_ID%\"}"
LIBERO_REPO_ID="${LIBERO_REPO_ID#\"}"
LIBERO_REPO_ID="${LIBERO_REPO_ID%\'}"
LIBERO_REPO_ID="${LIBERO_REPO_ID#\'}"
PI05_MODEL_PATH="${PI05_MODEL_PATH%\"}"
PI05_MODEL_PATH="${PI05_MODEL_PATH#\"}"
PI05_MODEL_PATH="${PI05_MODEL_PATH%\'}"
PI05_MODEL_PATH="${PI05_MODEL_PATH#\'}"

LIBERO_REPO_ID="${LIBERO_REPO_ID%/}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME%/}"
PI05_MODEL_PATH="${PI05_MODEL_PATH%/}"

if [[ "${LIBERO_REPO_ID}" == /* ]]; then
    HF_LEROBOT_HOME="$(dirname "${LIBERO_REPO_ID}")"
    LIBERO_REPO_ID="$(basename "${LIBERO_REPO_ID}")"
fi

if [ -f "${HF_LEROBOT_HOME}/meta/info.json" ]; then
    dataset_dir_basename="$(basename "${HF_LEROBOT_HOME}")"
    HF_LEROBOT_HOME="$(dirname "${HF_LEROBOT_HOME}")"
    if [ "${LIBERO_REPO_ID}" = "physical-intelligence/libero" ] || [ "${LIBERO_REPO_ID}" = "${dataset_dir_basename}" ]; then
        LIBERO_REPO_ID="${dataset_dir_basename}"
    fi
fi

if [ -z "${HF_LEROBOT_HOME}" ]; then
    echo "HF_LEROBOT_HOME is not set. Point it to the LeRobot dataset root." >&2
    exit 1
fi

if [ -z "${PI05_MODEL_PATH}" ]; then
    echo "PI05_MODEL_PATH is not set. Point it to the pi0.5 checkpoint directory." >&2
    exit 1
fi

export HF_LEROBOT_HOME
export PI05_MODEL_PATH
export LIBERO_REPO_ID
export RAY_TMPDIR="${RAY_TMPDIR:-/data/ray_tmp}"
mkdir -p "${RAY_TMPDIR}"

OPENPI_VENV_PYTHON="${OPENPI_VENV_PYTHON:-${REPO_PATH}/.venv-openpi-liberoplus/bin/python}"
if [ -x "${OPENPI_VENV_PYTHON}" ]; then
    export PATH="$(dirname "${OPENPI_VENV_PYTHON}"):${PATH}"
fi

export RLINF_RUN_ROOT="${RLINF_RUN_ROOT:-/data/rlinf_runs}"

echo "Using LeRobot root: ${HF_LEROBOT_HOME}"
echo "Using dataset repo_id: ${LIBERO_REPO_ID}"
if [ ! -f "${HF_LEROBOT_HOME}/${LIBERO_REPO_ID}/meta/info.json" ]; then
    echo "Dataset metadata not found at: ${HF_LEROBOT_HOME}/${LIBERO_REPO_ID}/meta/info.json" >&2
    echo "Set HF_LEROBOT_HOME to dataset root, and LIBERO_REPO_ID to dataset name only." >&2
    exit 1
fi

DATASET_DIR="${HF_LEROBOT_HOME}/${LIBERO_REPO_ID}"
MODEL_NORM="${PI05_MODEL_PATH}/${LIBERO_REPO_ID}/norm_stats.json"

if [ -f "${MODEL_NORM}" ]; then
    echo "norm_stats.json already present at model path: ${MODEL_NORM}"
elif [ -f "${DATASET_DIR}/norm_stats.json" ]; then
    echo "Copying existing norm_stats.json from dataset to model path..."
    mkdir -p "$(dirname "${MODEL_NORM}")"
    cp "${DATASET_DIR}/norm_stats.json" "${MODEL_NORM}"
    echo "  -> ${MODEL_NORM}"
elif [ -f "${LIBERO_NORM_STATS_SOURCE}" ]; then
    echo "Copying bundled LIBERO norm_stats.json to model path..."
    mkdir -p "$(dirname "${MODEL_NORM}")"
    cp "${LIBERO_NORM_STATS_SOURCE}" "${MODEL_NORM}"
    echo "  -> ${MODEL_NORM}"
elif [ "${RUN_NORM_STATS}" = "1" ]; then
    echo "No existing norm_stats.json found; computing from scratch..."
    python "${REPO_PATH}/toolkits/replay_buffer/calculate_norm_stats.py" \
        --config-name pi05_libero \
        --repo-id "${LIBERO_REPO_ID}" \
        --model-path "${PI05_MODEL_PATH}"
else
    echo "WARNING: norm_stats.json not found and RUN_NORM_STATS=0; skipping." >&2
fi

bash "${SCRIPT_DIR}/run_vla_sft.sh" libero_sft_openpi_pi05_2gpu "$@"
