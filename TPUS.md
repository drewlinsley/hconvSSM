TPUNAME=hssm3
ZONE=us-central1-a

gcloud compute tpus tpu-vm create $TPUNAME --zone=$ZONE --accelerator-type=v3-8 --preemptible --version=tpu-ubuntu2204-base
gcloud alpha compute tpus tpu-vm attach-disk $TPUNAME --zone=$ZONE --disk pathfinder --mode read-write


gcloud compute tpus tpu-vm ssh $TPUNAME --zone=$ZONE --command="git clone https://github.com/drewlinsley/hconvSSM.git; export PYTHONPATH=$PYTHONPATH:$(pwd); pip install jax[tpu]>=0.2.21 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html;git clone --depth=1 --branch=main https://github.com/google/flax;cd flax; sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y;pip install -e .;cd examples/imagenet;pip install -r requirements.txt;cd ~/hconvSSM;pip install -r requirements.txt; cp /home/drew_linsley_brown_edu/.local/lib/python3.10/site-packages/flaim/models/factory.py tpu_patches/; sudo mkdir -p /mnt/disks/pathfinder; sudo mount -o discard,defaults,noload  /dev/sdb /mnt/disks/pathfinder;cp .netrc /home/drew_linsley_brown_edu/;"

# gcloud compute tpus tpu-vm ssh $TPUNAME --zone=$ZONE --command="git clone https://github.com/drewlinsley/hconvSSM.git; export PYTHONPATH=$PYTHONPATH:$(pwd); pip install jax[tpu]>=0.2.21 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; cd hconvSSM; pip install -r requirements.txt; sudo mkdir -p /mnt/disks/pathfinder; sudo mount -o discard,defaults,noload  /dev/sdb /mnt/disks/pathfinder;cp .netrc /home/drew_linsley_brown_edu/;"

# gcloud compute tpus tpu-vm ssh $TPUNAME --zone=$ZONE --command="git clone https://github.com/drewlinsley/hconvSSM.git; export PYTHONPATH=$PYTHONPATH:$(pwd)"
# gcloud compute tpus tpu-vm ssh $TPUNAME --zone=$ZONE --command="pip install jax[tpu]==0.3.21 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; cd hconvSSM; pip install -r requirements.txt; sudo mkdir -p /mnt/disks/pathfinder; sudo mount -o discard,defaults,noload  /dev/sdb /mnt/disks/pathfinder"

gcloud compute tpus tpu-vm ssh $TPUNAME --zone=$ZONE

gcloud alpha compute tpus tpu-vm delete $TPUNAME --zone=$ZONE

