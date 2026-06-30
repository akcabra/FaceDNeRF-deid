#!/bin/bash

#SBATCH --job-name=facednerf-deid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

CONTAINER="nvcr.io#nvidia/pytorch:23.08-py3"

# --- configurable params (override via --export) ---
IMAGE_ID="${IMAGE_ID:-00018}"
PP="${PP:-0.5}"
NETWORK="${NETWORK:-./networks/ffhqrebalanced512-128.pkl}"
OUTDIR="${OUTDIR:-./output_deid}"
NUM_STEPS="${NUM_STEPS:-1000}"
NUM_STEPS_PTI="${NUM_STEPS_PTI:-400}"
lambda_DEID="${lambda_DEID:-2.5}"
lambda_ORIGIN="${lambda_ORIGIN:-1.0}"
lambda_GENDER="${lambda_GENDER:-0.01}"
lambda_EXPR="${lambda_EXPR:-0.01}"
lambda_LATENT="${lambda_LATENT:-0.0016}"

RESULT_DIR="${OUTDIR}/${IMAGE_ID}_deid_pp${PP}_${lambda_DEID}_${lambda_ORIGIN}_${lambda_GENDER}_${lambda_EXPR}"

echo "Image=${IMAGE_ID} pp=${PP} GPUs=${CUDA_VISIBLE_DEVICES:-auto}"

srun \
  --gres=gpu:2 \
  --container-image "${CONTAINER}" \
  --container-mounts "${PWD}":"${PWD}" \
  --container-workdir "${PWD}" \
  bash -c "
    set -euo pipefail
    echo '[diag] nvidia-smi:'
    nvidia-smi || true
python - <<'PY'
import torch
print('[diag] torch:', torch.__version__)
print('[diag] torch.cuda:', torch.version.cuda)
print('[diag] cuda_available:', torch.cuda.is_available())
print('[diag] device_count:', torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available inside container step. Check Slurm/Pyxis GPU passthrough.')
PY
    pip install --no-cache-dir -q --force-reinstall huggingface-hub==0.13.4
    pip install --no-cache-dir -q -r requirements.txt
    python run.py \
      --outdir=${OUTDIR} \
      --network=${NETWORK} \
      --sample_mult=2 \
      --image_path ./test_data/${IMAGE_ID}.png \
      --c_path ./test_data/${IMAGE_ID}.npy \
      --num_steps ${NUM_STEPS} \
      --num_steps_pti ${NUM_STEPS_PTI} \
      --mode deid \
      --pp ${PP} \
      --lambda_deid ${lambda_DEID} \
      --lambda_origin ${lambda_ORIGIN} \
      --lambda_gender ${lambda_GENDER} \
      --lambda_expr ${lambda_EXPR} \
      --lambda_latent ${lambda_LATENT}
    python gen_videos_from_given_latent_code.py \
      --outdir=${RESULT_DIR} \
      --trunc=0.7 \
      --npy_path ${RESULT_DIR}/checkpoints/${IMAGE_ID}.npy \
      --network=${RESULT_DIR}/checkpoints/fintuned_generator.pkl \
      --sample_mult=2
  "

echo "Done. Results: ${RESULT_DIR}"
