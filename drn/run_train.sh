python3 segment.py train -d ../bdd100k/seg -c 2 -s 896 \
    --arch drn_d_22 --batch-size 32 --epochs 250 --lr 0.01 --momentum 0.9 \
    --step 100