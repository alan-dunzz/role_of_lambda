#!/bin/bash
#SBATCH --account=aip-lelis
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=700M
#SBATCH -o jobs_outputs/job-%A_%a.out
#SBATCH -e jobs_outputs/job-%A_%a.err
# Note: Don't add #SBATCH --array. It is removed from here because it is now passed dynamically via the command line in Script 1.


# The starting index passed from the submission script (Argument 1)
START_INDEX=$1

# The python script name passed from the submission script (Argument 2)
SCRIPT_NAME=$2

# Calculate the global task ID
GLOBAL_TASK_ID=$((START_INDEX + SLURM_ARRAY_TASK_ID))

echo "Running ${SCRIPT_NAME} with global task ID: ${GLOBAL_TASK_ID}"

# Run the specific script passed as an argument
python ${SCRIPT_NAME} ${GLOBAL_TASK_ID}