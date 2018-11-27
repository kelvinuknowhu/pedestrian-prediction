# Compute mean and std
python compute_mean_std.py --data-dir bdd/seg


# Test the pretrained Cityscapes on BDD
python segment.py test -d datasets/bdd/seg -c 19 \
    --arch drn_d_22 --pretrained pretrained/drn_d_22_cityscapes.pth \
    --phase test --batch-size 1

# Move files to Graphite
scp [file] sh2442@graphite.coecis.cornell.edu:/home/sh2442

# Check GPU information
cat /proc/driver/nvidia/gpus/0000\:02\:00.0/information 

# Submit a batch (non-interactive) job
sbatch -J fine_tune_bdd -o bdd_fine_tune.o%j --nodelist=hinton --partition=cuvl --requeue -n 2 --gres=gpu:2 --mem=64G fine_tune.sh