# Activate pytorch environment on Graphite
. /home/sh2442/anaconda3/etc/profile.d/conda.sh
conda activate pytorch

# Compute mean and std
python compute_mean_std.py --data-dir bdd/seg

# Test a pretrained model on Cityscapes/BDD
python segment.py test --data-dir datasets/bdd --classes 19 --arch drn_d_22 --pretrained pretrained/model_best.pth.tar --phase test --batch-size 1

# Move files to Graphite
scp [file] sh2442@graphite.coecis.cornell.edu:/home/sh2442
scp [file] Kelvin@[ip-address]:~/Downloads

# Check GPU information
cat /proc/driver/nvidia/gpus/0000\:02\:00.0/information 

# Submit a batch (non-interactive) job
sbatch --job-name=bdd --output=bdd.o%j --nodelist=hinton --partition=cuvl --requeue --cpus-per-task=2 --gres=gpu:2 --mem=64G fine_tune.sh

sbatch --job-name=mapillary --output=mapillary.o%j --nodelist=hinton --partition=cuvl --requeue --cpus-per-task=2 --gres=gpu:2 --mem=64G fine_tune.sh

sbatch --job-name=bdd --output=bdd.o%j --nodelist=harpo --partition=kilian --requeue --gres=gpu:2 --mem=64G fine_tune.sh

sbatch --job-name=bdd --output=bdd.o%j --nodelist=nikola-compute02 --partition=default_gpu --requeue --cpus-per-task=2 --gres=gpu:2 --mem=64G fine_tune.sh