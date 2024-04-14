rm -rf logs/*
python scripts/train_pf.py -d pathfinder -o gs://serrelab/hssm/pf-14-1 -c configs/PF-14/PF-14.yaml

