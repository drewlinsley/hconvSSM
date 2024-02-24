conda create -n jax2 python=3.10.11
conda activate jax2
conda install --channel "nvidia/label/cuda-11.8.0" cuda

#I have setup cudnn on p8 so you would just need to do this - (or add to bash profile for it to run everytim)
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

pip install --upgrade jax[cuda]==0.3.21 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

#from inside the convssm folder
pip install -r requirements.txt
pip install -e .

