# Compute mean and std
python compute_mean_std.py --data-dir bdd/seg

# Fine tune the pretrained Cityscapes model on BDD
python3 segment.py train -d datasets/bdd/seg -c 19 -s 896 \
    --arch drn_d_22 --batch-size 32 --epochs 250 --lr 0.01 --momentum 0.9 --step 100 

# Test the pretrained Cityscapes on BDD
python segment.py test -d ../bdd100k/seg -c 19 --arch drn_d_22 --pretrained ./pretrained/drn_d_22_cityscapes.pth --phase test --batch-size 1

# Move files to Graphite
scp [file] sh2442@graphite.coecis.cornell.edu:/home/sh2442

