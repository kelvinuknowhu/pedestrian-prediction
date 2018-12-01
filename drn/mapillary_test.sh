#!/bin/bash

# Use an anaconda environment
. /home/sh2442/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

python segment.py test --data-dir /scratch/datasets/mapillary --classes 2 \
--arch drn_d_38 --pretrained pretrained/mapillary_model_best.pth \
--phase test --batch-size 1