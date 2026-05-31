#!/bin/bash
# End-to-end: build_csv → inference_short.sh, consistently using the oneshot environment
#
# Location: ONE-SHOT/scripts/run_pipeline.sh
# Usage (can be called from any directory; paths are resolved automatically; arguments are exactly the same as build_csv.py):
#   bash scripts/run_pipeline.sh <task> --video_path ... --prompt "..." [...]
#   task = id_swap | motion_swap | scene_swap
# Note: No need to pass --out_csv; it will automatically generate Preprocessing/tmpout/<task>_<timestamp>.csv.
#
# Environment convention:
#   Preprocessing + inference: /root/anaconda3/envs/${ONESHOT_ENV:-oneshot}
#   (The old split_env version is kept in run_pipeline.split_env.sh.bak)
#
# GPU allocation:
#   inference --nproc_per_node = min(number of CSV rows, number of available system GPUs)
#   The script also uses nvidia-smi to automatically select the GPUs with the most free memory as CUDA_VISIBLE_DEVICES.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <id_swap|motion_swap|scene_swap> [build_csv args...]" >&2
    exit 2
fi

TASK="$1"; shift
case "$TASK" in
    id_swap|motion_swap|scene_swap) ;;
    *) echo "ERROR: unknown task '$TASK'"; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PREPROC_DIR="${REPO_ROOT}/Preprocessing"

HUMAN3R_PY="/root/anaconda3/envs/${ONESHOT_ENV:-oneshot}/bin/python"
COGVIDEO_BIN="/root/anaconda3/envs/${ONESHOT_ENV:-oneshot}/bin"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${PREPROC_DIR}/tmpout"
mkdir -p "$OUT_DIR"
CSV="${OUT_DIR}/${TASK}_${TS}.csv"

# ---------- 1) build CSV (oneshot env) ----------
echo "[run_pipeline] === build_csv (${TASK}) → ${CSV} ==="
"$HUMAN3R_PY" "${PREPROC_DIR}/build_csv.py" "$TASK" "$@" --out_csv "$CSV"

if [[ ! -s "$CSV" ]]; then
    echo "[run_pipeline] ERROR: csv not produced: $CSV" >&2
    exit 1
fi

# ---------- 2) infer GPU/parallel config ----------
N_ROWS=$(($(wc -l < "$CSV") - 1))   # Subtract the header
if [[ $N_ROWS -le 0 ]]; then
    echo "[run_pipeline] ERROR: csv has 0 data rows" >&2
    exit 1
fi
N_GPU_TOTAL=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
NPROC=$(( N_ROWS < N_GPU_TOTAL ? N_ROWS : N_GPU_TOTAL ))
[[ $NPROC -lt 1 ]] && NPROC=1

# Select the NPROC GPUs with the most free memory
GPU_IDS=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | sort -t',' -k2 -n -r | head -n "$NPROC" | awk -F',' '{gsub(/ /,"",$1); print $1}' \
          | tr '\n' ',' | sed 's/,$//')

echo "[run_pipeline] CSV rows=${N_ROWS}, total GPUs=${N_GPU_TOTAL} → nproc=${NPROC}, CUDA_VISIBLE_DEVICES=${GPU_IDS}"

# ---------- 3) run inference (oneshot env) ----------
echo "[run_pipeline] === inference_short.sh (oneshot env) ==="
export PATH="${COGVIDEO_BIN}:${PATH}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export CSV_NAME="$CSV"
export NPROC_PER_NODE="$NPROC"
bash "${REPO_ROOT}/scripts/inference_short.sh"

echo "[run_pipeline] === DONE ==="
echo "[run_pipeline] CSV : $CSV"