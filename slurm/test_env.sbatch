#!/bin/bash -l
# Check the GPU, CUDA, PyTorch and CLIP setup. Submit from the repository root:
#   sbatch slurm/test_env.sbatch
#
# ClsGenAI is the GAIVI class partition. Other clusters must change it.
#SBATCH --job-name=test_env
#SBATCH --partition=ClsGenAI
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=test_env_%j.out

set -euo pipefail

if [ ! -f train.py ]; then
    echo "ERROR: expected to run from the repository root, found no train.py in ${PWD}." >&2
    exit 1
fi

# conda's activation scripts read variables that may be unset.
set +u
conda activate maple
set -u

echo "git commit: $(git rev-parse HEAD)"
srun nvidia-smi
# The repository root must be importable, because the script itself is in tools/.
PYTHONPATH="${PWD}" srun python tools/check_env.py
