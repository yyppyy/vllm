USR=pi_ak2579
PART=gpu_h200

srun -A $USR --mem=500g --gpus=h200:1 --partition=$PART --nodes=1 --cpus-per-task=32 --time=3:00:00 ./setup.sh