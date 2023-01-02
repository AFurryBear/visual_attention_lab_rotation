#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=3         # Number of CPU cores per task
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=0-12:00          # Runtime in D-HH:MM
#SBATCH --mem=50G                # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/qb/work/wu/yxiong34/annalab/log/hostname_%j.out  # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/qb/work/wu/yxiong34/annalab/log/hostname_%j.err   # File to which STDERR will be written - make sure this is not on $HOME

# print info about current job
scontrol show job $SLURM_JOB_ID
source /etc/profile.d/conda.sh
conda activate abcTau_yirong
python main_abcmodel.py $1 $2 $3 $4 $5 $6