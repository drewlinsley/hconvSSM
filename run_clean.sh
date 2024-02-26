rm -rf logs/mmnist_convs5_drew1/checkpoints/*
CUDA_VISIBLE_DEVICES=0 python scripts/train.py -d data/moving-mnist-pytorch -o mmnist_convs5_drew1 -c configs/Moving-MNIST/300_train_len/mnist_convS5_novq.yaml

