# Compute mean and std
python compute_mean_std.py --data-dir bdd/seg

# Fine tune the pretrained Cityscapes model on BDD
python3 segment.py train -d datasets/bdd/seg -c 19 -s 896 \
    --arch drn_d_22 --batch-size 32 --epochs 250 --lr 0.01 --momentum 0.9 --step 100