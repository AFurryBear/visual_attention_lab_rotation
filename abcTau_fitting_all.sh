#!/bin/bash

# insert your commands here
# python main_abcmodel.py PEN288 1 OUT PreFirstDimDataBiZSc Supra oneTauOU_oscil
atten_arr=(RF OUT)
epoch_arr=(PreFirstDimDataBiZSc)
layer_arr=(Supra Granr Infra)
model_arr=(oneTauOU oneTauOU_oscil)


while IFS="," read -r rec_column1 rec_column2 rec_column3
do
  for model in ${model_arr[@]}
  do
    for layer in ${layer_arr[@]}
    do
      for epoch in ${epoch_arr[@]}
      do
        for atten in ${atten_arr[@]}
        do
        sbatch --account levina /home/wu/yxiong34/annalab/code/abcTau/abcTau_slurm.sh $rec_column3 $rec_column1 $atten $epoch $layer $model
  #      echo "sbatch ./abcTau_slurm.sh $atten $epoch $layer oneTauOU_oscil"
  #      echo "sbatch ./abcTau_slurm.sh $rec_column3 $rec_column1 $atten $epoch $layer oneTauOU_oscil"
        echo "sessionID: $rec_column3"
        echo "monkeyName: $rec_column2"
        echo "========="
        done
      done
    done
  done
done < /mnt/qb/work/wu/yxiong34/levina_data/Subject_Info_V4_test.csv
