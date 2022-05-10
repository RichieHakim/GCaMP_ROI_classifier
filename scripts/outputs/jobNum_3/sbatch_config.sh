#!/usr/bin/bash
#SBATCH --job-name=python_test
#SBATCH --output=/n/data1/hms/neurobio/sabatini/josh/github_repos/GCaMP_ROI_classifier/scripts/20220508_simCLR_JZfix/logs/python_01_%j.log
#SBATCH --partition=gpu_quad
#SBATCH --gres=gpu:rtx8000:1
#SBATCH -c 16
#SBATCH -n 1
#SBATCH --mem=100GB
#SBATCH --time=0-01:00:00

python "$@"
