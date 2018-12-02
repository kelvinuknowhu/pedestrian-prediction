#!/bin/bash

# Use an anaconda environment
. /home/sh2442/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

python process_clips.py