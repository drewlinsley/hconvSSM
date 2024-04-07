GPU=5
TIMESTEPS=1000
REPEATS=20

CUDA_VISIBLE_DEVICES=$GPU python src/models/convS5/fft_conv_conv.py $TIMESTEPS $REPEATS
CUDA_VISIBLE_DEVICES=$GPU python src/models/convS5/fft_conv_fft.py $TIMESTEPS $REPEATS
CUDA_VISIBLE_DEVICES=$GPU python src/models/convS5/fft_conv_scan.py $TIMESTEPS $REPEATS

python plot_benchmarks.py

