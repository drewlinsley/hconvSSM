rm -rf logs/*
python scripts/train_pf.py -d pathfinder -o gs://serrelab/hssm/hpf-14-1 -c configs/PF-14/hPF-14.yaml

