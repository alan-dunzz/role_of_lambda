#!/bin/bash
# Script to submit over 1000 jobs by breaking them into multiple job arrays of 1000 jobs each.
# Usage: bash submit_over_1000_jobs.sh <python_script_name> <total_number_of_jobs>
# Example usage: bash submit_over_1000_jobs.sh test_ppo_parallel.py 2500

# Total number of jobs to run
TOTAL_JOBS=$2
# Maximum number of jobs per array
MAX_ARRAY_SIZE=1000
# The python script to run
PYTHON_SCRIPT=$1 #Example: "test_ppo_parallel.py"

# Loop to submit multiple job arrays
for (( i=0; i<${TOTAL_JOBS}; i+=${MAX_ARRAY_SIZE} )); do
  
  # Calculate how many jobs are remaining including this batch
  REMAINING_JOBS=$((TOTAL_JOBS - i))
  
  # Determine the array limit for this specific submission
  if [ ${REMAINING_JOBS} -lt ${MAX_ARRAY_SIZE} ]; then
     # If remaining jobs are less than max size, limit the array to the remainder
     # Subtract 1 because array is 0-indexed (e.g., 50 jobs = 0-49)
     CURRENT_LIMIT=$((REMAINING_JOBS - 1))
  else
     # Otherwise use the full max size
     CURRENT_LIMIT=$((MAX_ARRAY_SIZE - 1))
  fi

  echo "Submitting jobs starting at global index $i with array range 0-${CURRENT_LIMIT}"
  
  # Submit the Slurm script.
  # We pass the --array flag HERE to override/set the array size dynamically.
  # We pass $i (start index) as arg $1
  # We pass $PYTHON_SCRIPT as arg $2
  sbatch --array=0-${CURRENT_LIMIT} parallelize_job.sbatch $i $PYTHON_SCRIPT
done