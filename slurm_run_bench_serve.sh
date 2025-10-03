USR=pi_ak2579
PART=gpu_h200

srun -A $USR --mem=100g --gpus=h200:1 --partition=$PART --nodes=1 --cpus-per-task=8 --time=00:30:00 ./bench_serve.sh