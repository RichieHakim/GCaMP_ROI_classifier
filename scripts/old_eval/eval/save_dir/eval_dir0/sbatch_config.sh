#!/usr/bin/bash
#SBATCH --job-name=job_name
#SBATCH --output=/n/data1/hms/neurobio/sabatini/josh/github_repos/GCaMP_ROI_classifier/scripts/eval2/save_dir/eval_dir0/print_log_%j.log
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --partition=gpu_requeue
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --mem=100GB
#SBATCH --time=0-00:45:00
unset XDG_RUNTIME_DIR
cd /n/data1/hms/neurobio/sabatini/josh/
date
echo "loading modules"
module load gcc/9.2.0
echo "activating environment"
source activate NBAP
echo "starting job"
python "$@"
