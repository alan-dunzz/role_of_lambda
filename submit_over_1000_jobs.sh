    
#!/bin/bash

# Total number of jobs to run
TOTAL_JOBS=11100
# Maximum number of jobs per array
MAX_ARRAY_SIZE=1000

# Loop to submit multiple job arrays
for (( i=0; i<${TOTAL_JOBS}; i+=${MAX_ARRAY_SIZE} )); do
  # The start index for the current array
  START_INDEX=$i
  # The end index for the current array
  END_INDEX=$((i + MAX_ARRAY_SIZE - 1))

  # Ensure the end index doesn't exceed the total number of jobs
  if [ ${END_INDEX} -ge ${TOTAL_JOBS} ]; then
    END_INDEX=$((TOTAL_JOBS - 1))
  fi

  echo "Submitting jobs from ${START_INDEX} to ${END_INDEX}"
  # Submit the Slurm script with the start index as an argument
  sbatch parallelize.sbatch ${START_INDEX}
done

  