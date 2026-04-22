#!/bin/bash
#SBATCH --chdir=.
#SBATCH --account=digital_human_jobs
#SBATCH --time=00:10:00
#SBATCH --output=/home/%u/slurm_output__%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --gpus=5060ti:1
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=24G

set -euo pipefail
set -x

echo "PWD: $(pwd)"
echo "HOST: $(hostname)"
echo "STARTING AT $(date)"

# System modules are not automatically available in batch jobs.
. /etc/profile.d/modules.sh
module add cuda/13.0

# Environment. This assumes you followed the README and created `digihum`.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate digihum

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available?', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
nvidia-smi

DATA_DIR="${DIGIHUM_TUTORIAL_DATA:-/work/scratch/$USER/digihum-tutorial-data}"
mkdir -p "$DATA_DIR"

python train_mnist.py --epochs 1 --batch-size 256 --data-dir "$DATA_DIR"

echo "Done."
echo "FINISHED AT $(date)"
