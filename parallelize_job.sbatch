#!/bin/bash
#SBATCH --account=aip-lelis
#SBATCH --time=01:00:00
#SBATCH --array=0-999
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=700M
#SBATCH -o jobs_outputs/job-%A_%a.out
#SBATCH -e jobs_outputs/job-%A_%a.err

# The starting index passed from the submission script
START_INDEX=$1

# Calculate the global task ID
GLOBAL_TASK_ID=$((START_INDEX + SLURM_ARRAY_TASK_ID))

echo "Running job with global task ID: ${GLOBAL_TASK_ID}"
python test_ppo_parallel.py ${GLOBAL_TASK_ID}

  