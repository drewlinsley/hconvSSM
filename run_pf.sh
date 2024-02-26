rm -rf logs/mmnist_convs5_drew1/checkpoints/*
rm -rf logs/*
CUDA_VISIBLE_DEVICES=0 python scripts/train.py -d pathfinder -o pf-14 -c configs/PF-14/PF-14.yaml

