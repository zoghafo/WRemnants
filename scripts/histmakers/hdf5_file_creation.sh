#!/usr/bin/env bash

# Array of ptll boundaries
NUMBERS=(0 20)

# Base command parameters
SCRIPT="/home/z/zoghafoo/WRemnants/scripts/histmakers/mz_dilepton.py"
COMMON_ARGS=(
  -j 200
  --axes ptll mll yll absYll
  -v 4
  --dataPath /scratch/shared/NanoAOD/
  --outfolder /scratch/zoghafoo/
  --saveEventCsv
  --eventCsvOutDir /home/z/zoghafoo/WRemnants/csv_files/allFiles
  --postfix allFiles_mZ20to30000
  # --maxFiles 100
  # --postfix mZ_20_30000
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
