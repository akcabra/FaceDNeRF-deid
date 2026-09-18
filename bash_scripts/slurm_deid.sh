#!/bin/bash

#SBATCH --job-name=facednerf-deid
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --gpus 1
#SBATCH --mem-per-gpu 32G
#SBATCH --partition=gpu
#SBATCH --time 2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

module load GCC/11.3.0
module load CUDA/11.7.0
module load Anaconda3/2023.07-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate facednerf

# --- configurable params (override via --export) ---
IMAGE_ID="${IMAGE_ID:-00018}"
INPUT_DIR="${INPUT_DIR:-./test_data}"
PP="${PP:-0.3}"
NETWORK="${NETWORK:-./networks/ffhqrebalanced512-128.pkl}"
OUTDIR="${OUTDIR:-./output_deid}"
NUM_STEPS="${NUM_STEPS:-500}"
NUM_STEPS_PTI="${NUM_STEPS_PTI:-0}"
NUM_STEPS_INVERSION="${NUM_STEPS_INVERSION:-300}"
lambda_DEID="${lambda_DEID:-2.5}"
lambda_ORIGIN="${lambda_ORIGIN:-1.0}"
lambda_GENDER="${lambda_GENDER:-0.01}"
lambda_EXPR="${lambda_EXPR:-0.01}"
lambda_LATENT="${lambda_LATENT:-0.0016}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-job_${SLURM_JOB_ID:-manual}_seed${SEED}}"
lambda_INV_PERCEPTUAL="${lambda_INV_PERCEPTUAL:-0.8}"
lambda_INV_IDENTITY="${lambda_INV_IDENTITY:-0.1}"
lambda_STAGE2_PERCEPTUAL="${lambda_STAGE2_PERCEPTUAL:-0.4}"
GRADIENT_LOG_INTERVAL="${GRADIENT_LOG_INTERVAL:-0}"
STAGE2_LATENT_NOISE="${STAGE2_LATENT_NOISE:-1}"
STAGE2_PIXEL_RESOLUTION="${STAGE2_PIXEL_RESOLUTION:-512}"
SAVE_PROGRESS_IMAGES="${SAVE_PROGRESS_IMAGES:-0}"

IMAGE_PATH="${INPUT_DIR}/${IMAGE_ID}.png"
CAMERA_PATH="${INPUT_DIR}/${IMAGE_ID}.npy"

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "Input image not found: ${IMAGE_PATH}" >&2
  exit 2
fi
if [[ ! -f "${CAMERA_PATH}" ]]; then
  echo "Camera parameters not found: ${CAMERA_PATH}" >&2
  exit 2
fi

if [[ "${STAGE2_LATENT_NOISE}" == "1" ]]; then
  STAGE2_NOISE_ARG="--stage2_latent_noise"
else
  STAGE2_NOISE_ARG="--no-stage2_latent_noise"
fi

if [[ "${SAVE_PROGRESS_IMAGES}" == "1" ]]; then
  PROGRESS_IMAGE_ARG="--save-progress-images"
else
  PROGRESS_IMAGE_ARG="--final-images-only"
fi

echo "Image=${IMAGE_ID} input_dir=${INPUT_DIR} pp=${PP} seed=${SEED} run=${RUN_NAME} GPUs=${CUDA_VISIBLE_DEVICES:-auto}"
echo "Python: $(command -v python)"
echo "GCC: $(command -v gcc)"
echo "GCC version: $(gcc -dumpfullversion -dumpversion)"
echo "NVCC: $(command -v nvcc)"
nvcc --version
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"

srun python run.py \
  --outdir="${OUTDIR}" \
  --network="${NETWORK}" \
  --sample_mult=2 \
  --image_path "${IMAGE_PATH}" \
  --c_path "${CAMERA_PATH}" \
  --num_steps ${NUM_STEPS} \
  --num_steps_pti ${NUM_STEPS_PTI} \
  --num_steps_inversion ${NUM_STEPS_INVERSION} \
  --pp ${PP} \
  --lambda_deid ${lambda_DEID} \
  --lambda_origin ${lambda_ORIGIN} \
  --lambda_gender ${lambda_GENDER} \
  --lambda_expr ${lambda_EXPR} \
  --lambda_latent ${lambda_LATENT} \
  --lambda_inversion_perceptual ${lambda_INV_PERCEPTUAL} \
  --lambda_inversion_identity ${lambda_INV_IDENTITY} \
  --lambda_stage2_perceptual ${lambda_STAGE2_PERCEPTUAL} \
  --gradient_log_interval ${GRADIENT_LOG_INTERVAL} \
  ${STAGE2_NOISE_ARG} \
  ${PROGRESS_IMAGE_ARG} \
  --stage2_pixel_resolution ${STAGE2_PIXEL_RESOLUTION} \
  --seed ${SEED} \
  --run_name "${RUN_NAME}"

echo "Done. See the 'Run output:' path above for results."
