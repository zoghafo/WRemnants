#!/usr/bin/env bash

# Array of ptll boundaries
NUMBERS=(0 2 4 6 8 10 12 14 16 18 20)

# Base command parameters
SCRIPT="/home/z/zoghafoo/WRemnants/scripts/histmakers/mz_dilepton.py"
COMMON_ARGS=(
  -j 200
  --maxFiles 100
  --axes ptll mll yll absYll xmaxll xminll
  -v 4
  --dataPath /scratch/shared/NanoAOD/
)

# Loop over consecutive pairs
for ((i=0; i<${#NUMBERS[@]}-1; i++)); do
  PTMIN=${NUMBERS[i]}
  PTMAX=${NUMBERS[i+1]}

  CMD=(
    python "$SCRIPT"
    "${COMMON_ARGS[@]}"
    --ptllMin "$PTMIN"
    --ptllMax "$PTMAX"
  )

  echo
  echo "============================================================"
  echo "Running command:"
  printf '%q ' "${CMD[@]}"
  echo
  echo "============================================================"

  "${CMD[@]}"

done
