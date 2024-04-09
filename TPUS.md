TPUNAME=hssm3
ZONE=us-central1-a
# ZONE=us-east1-d

gcloud alpha compute tpus tpu-vm delete $TPUNAME --zone=$ZONE

gcloud compute tpus tpu-vm create $TPUNAME --zone=$ZONE --accelerator-type=v3-8 --preemptible --version=tpu-ubuntu2204-base
# gcloud compute tpus tpu-vm create $TPUNAME --zone=$ZONE --accelerator-type=v3-32 --version=tpu-ubuntu2204-base

gcloud alpha compute tpus tpu-vm attach-disk $TPUNAME --zone=$ZONE --disk pathfinder --mode read-only

gcloud compute tpus tpu-vm ssh $TPUNAME --zone=$ZONE --command="git clone https://github.com/drewlinsley/hconvSSM.git; pip install jax[tpu]>=0.2.21 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html;sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y;cd ~/hconvSSM;pip install -r requirements.txt;pip install -e .;cp tpu_patches/factory.py /home/drew_linsley_brown_edu/.local/lib/python3.10/site-packages/flaim/models/factory.py; sudo mkdir -p /mnt/disks/pathfinder; sudo mount -o discard,defaults,noload  /dev/sdb /mnt/disks/pathfinder;cp .netrc /home/drew_linsley_brown_edu/;tmux;export PYTHONPATH=$PYTHONPATH:$(pwd)"

gcloud compute tpus tpu-vm ssh $TPUNAME --zone=$ZONE

gcloud alpha compute tpus tpu-vm delete $TPUNAME --zone=$ZONE

