#!/bin/bash

TEMPLATE=job-template.yaml

# List of job names and corresponding data files
declare -a JOBS=(
    "chunk_00.mat"
    "chunk_01.mat"
    "chunk_02.mat"
    "chunk_03.mat"
    "chunk_04.mat"
    "chunk_05.mat"
    "chunk_06.mat"
    "chunk_07.mat"
    "chunk_08.mat"
    "chunk_09.mat"
    "chunk_10.mat"
    "chunk_11.mat"
    "chunk_12.mat"
    "chunk_13.mat"
    "chunk_14.mat"
    "chunk_15.mat"
    "chunk_16.mat"
    "chunk_17.mat"
    "chunk_18.mat"
)

for entry in "${JOBS[@]}"; do
  read -r JOB_PREFIX DATA_FILE <<< "$entry"
  JOB_NAME=$(echo "$JOB_PREFIX-$(basename "$DATA_FILE" .mat)" | tr '_' '-')

  # Generate YAML with substitutions
  sed "s/JOB_NAME_PLACEHOLDER/${JOB_NAME}/g; s|JOB_DATA_PLACEHOLDER|${DATA_FILE}|g" $TEMPLATE > "job-$JOB_NAME.yaml"

  # Submit the job
  kubectl apply -f "job-$JOB_NAME.yaml"

  echo "✅ Submitted job: $JOB_NAME"
done
