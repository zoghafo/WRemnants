#!/bin/bash

VARS=(ptll mll yll)
NUMBERS=(0 2 4 6 8 10 12 14 16 18 20)

for ((i=0; i<${#NUMBERS[@]}-1; i++)); do
  low=${NUMBERS[i]}
  high=${NUMBERS[i+1]}

  FILES=(mz_dilepton_scetlib_dyturboCorr_maxFiles_100_ZpT${low}to${high}.hdf5)

  for f in "${FILES[@]}"; do
    for v in "${VARS[@]}"; do
        echo "Processing $f with variable $v"
        if [ "$v" == "ptll" ]; then
            python /home/z/zoghafoo/WRemnants/scripts/plotting/makeDataMCStackPlot.py /home/z/zoghafoo/WRemnants/${f} --hists ${v} --ptcut ZpT${low}to${high} --outfolder /home/z/zoghafoo/www/pT_Cuts/ --extraText "${low} GeV <= pT < ${high} GeV" --extraTextLoc 0.5 0.8
        else
          python /home/z/zoghafoo/WRemnants/scripts/plotting/makeDataMCStackPlot.py /home/z/zoghafoo/WRemnants/${f} --hists ${v} --ptcut ZpT${low}to${high} --outfolder /home/z/zoghafoo/www/pT_Cuts/ --extraText "${low} GeV <= pT < ${high} GeV"
        fi
    done
  done
done
