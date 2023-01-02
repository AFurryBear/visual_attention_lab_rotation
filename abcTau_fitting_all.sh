#!/bin/bash

# insert your commands here
# python main_abcmodel.py PEN288 1 OUT PreFirstDimDataBiZSc Supra oneTauOU_oscil
atten_arr=(RF OUT)
epoch_arr=(PreFirstDimDataBiZSc StationaryDataBiZSc)
layer_arr=(Supra Granr Infra)
model_arr=(oneTauOU_oscil)


while IFS="," read -r rec_column1 rec_column2 rec_column3
do
  for layer in ${layer_arr[@]}
  do
    for epoch in ${epoch_arr[@]}
    do
      for atten in ${atten_arr[@]}
      do
      sbatch --account levina ./abcTau_slurm.sh $rec_column3 $rec_column1 $atten $epoch $layer oneTauOU_oscil
#      echo "sbatch ./abcTau_slurm.sh $atten $epoch $layer oneTauOU_oscil"
#      echo "sbatch ./abcTau_slurm.sh $rec_column3 $rec_column1 $atten $epoch $layer oneTauOU_oscil"
      echo "sessionID: $rec_column3"
      echo "monkeyName: $rec_column2"
      echo "========="
      done
    done
  done
done < Subject_Info_V4_Slurm.csv
