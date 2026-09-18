#!/bin/bash

#SBATCH --job-name=train-classifier
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --gpus 1
#SBATCH --mem-per-gpu 32G
#SBATCH --partition=gpu
#SBATCH --time 2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load Anaconda3/2023.07-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate facednerf

echo "Python: $(command -v python)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())"

srun python train_gender_classifier.py \
  --data_dir ./data/celeba \
  --save_path ./networks/gender_classifier.pth \
  --epochs 10 \
  --batch_size 128 \
  --lr 1e-3 \
  --num_workers "${SLURM_CPUS_PER_TASK}" \
  --gpu 0
