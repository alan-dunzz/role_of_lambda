#!/bin/bash
#SBATCH --account=aip-lelis
#SBATCH --time=01:00:00
#SBATCH --array=0-11099
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=700M
#SBATCH -o jobs_outputs/job-%A_%a.out
#SBATCH -e jobs_outputs/job-%A_%a.err

python test_ppo_parallel.py $SLURM_ARRAY_TASK_ID