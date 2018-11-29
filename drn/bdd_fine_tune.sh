#!/bin/bash

#SBATCH --job-name=fine_tune_bdd               # Job name
#SBATCH --output=fine_tune_bdd.o%j             # Name of stdout output file (%j expands to jobId)
#SBATCH --error=fine_tune_bdd.o%j              # Name of stderr output file
#SBATCH ---nodes=1                             # Total number of CPU nodes requested
#SBATCH --ntasks-per-node=1                    # Total number of CPU cores requrested
#SBATCH --mem=5G                               # CPU Memory pool for all cores
#SBATCH --time=48:00:00                        # Run time (hh:mm:ss)
#SBATCH --partition=default_gpu --gres=gpu:2   # Which queue to run on, and what resources to use
#SBATCH --partition=default_gpu                # Specify the partition to use
#SBATCH --gres=gpu:1                           # Use 1 GPU of any type
#SBATCH --gres=gpu:1080ti:1                    # Use 1 GTX 1080TI GPU

# Use an anaconda environment
. /home/sh2442/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

# Check if BDD dataset exists
if [ ! -d /scratch/datasets/bdd ]; then
  cp -r /home/sh2442/pedestrian-prediction/drn/datasets/bdd /scratch/datasets
fi

# Fine tune on BDD
python3 segment.py train --data-dir /scratch/datasets/bdd --classes 19 --crop-size 840 \
--arch drn_d_22 --batch-size 8 --epochs 250 --lr 0.01 --momentum 0.9 \
--step 100 --pretrained pretrained/drn_d_22_cityscapes.pth