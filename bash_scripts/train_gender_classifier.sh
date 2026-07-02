#!/bin/bash

#SBATCH --job-name=train-gender
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

srun \
  --container-image "nvcr.io#nvidia/pytorch:23.08-py3" \
  --container-mounts "${PWD}":"${PWD}" \
  --container-workdir "${PWD}" \
  bash -c "
    pip install --no-cache-dir -q -r requirements.txt
    python train_gender_classifier.py \
      --data_dir ./data/celeba \
      --save_path ./networks/gender_classifier.pth \
      --epochs 10 \
      --batch_size 128 \
      --lr 1e-3 \
      --num_workers 8 \
      --gpu 0
  "
