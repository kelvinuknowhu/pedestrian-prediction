#!/bin/bash

#SBATCH -J fine_tune_bdd        # Job name
#SBATCH -o fine_tune_bdd.o%j    # Name of stdout output file (%j expands to jobId)
#SBATCH -e fine_tune_bdd.o%j    # Name of stderr output file
#SBATCH -N 1                    # Total number of CPU nodes requested
#SBATCH -n 1                    # Total number of CPU cores requrested
#SBATCH --mem=5G                # CPU Memory pool for all cores
#SBATCH -t 48:00:00             # Run time (hh:mm:ss)
#SBATCH --partition=default_gpu --gres=gpu:2    # Which queue to run on, and what resources to use
                                               # --partition=<queue> - Use the `<queue>` queue
                                               # --gres=gpu:1 - Use 1 GPU of any type
                                               # --gres=gpu:1080ti:1 - Use 1 GTX 1080TI GPU

# OPTIONAL: Uncomment this if you're using an anaconda environment
. /home/sh2442/anaconda3/etc/profile.d/conda.sh
conda activate pytorch


# OPTIONAL: Uncomment this if you need to copy a dataset over to scratch
#           This checks to see if the dataset already exists
if [ ! -d /scratch/datasets/bdd ]; then
  cp -r /home/sh2442/pedestrian-prediction/drn/datasets/bdd /scratch/datasets
fi

# if [ ! -d /scratch/datasets/mapillary ]; then
#   cp -r /home/sh2442/pedestrian-prediction/drn/datasets/mapillary /scratch/datasets
# fi

# python3 segment.py train --data-dir /scratch/datasets/bdd --classes 19 --crop-size 840 \
# --arch drn_d_22 --batch-size 8 --epochs 10 --lr 0.01 --momentum 0.9 \
# --step 100 --pretrained pretrained/drn_d_22_cityscapes.pth

python3 segment.py train --data-dir /scratch/datasets/bdd --classes 19 --crop-size 840 \
--arch drn_d_105 --random-scale 2 --random-rotate 10 --batch-size 16 --epochs 250 --lr 0.01 --momentum 0.9 \
--workers 16 --lr-mode poly --bn-sync
