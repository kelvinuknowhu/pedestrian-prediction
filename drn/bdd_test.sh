#!/bin/bash

# Use an anaconda environment
. /home/sh2442/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

python segment.py test --data-dir /scratch/datasets/bdd --classes 19 \
--arch drn_d_22 --pretrained pretrained/bdd_model_best.pth \
--phase test --batch-size 1