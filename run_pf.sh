rm -rf logs/*
# XLA_PYTHON_CLIENT_PREALLOCATE=False CUDA_VISIBLE_DEVICES=0 python scripts/train_pf.py -d pathfinder -o pf-14 -c configs/PF-14/PF-14.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/train_pf.py -d pathfinder -o pf-14 -c configs/PF-14/PF-14.yaml

