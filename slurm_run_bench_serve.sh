USR=pi_ak2579
PART=gpu_h200

NUM_GPUS=$1
EP_DEGREE=$2
NUM_REPLICAS=$3
BATCH_SIZE=$4

srun -A $USR --mem=100g --gpus=h200:$NUM_GPUS --partition=$PART --nodes=1 --cpus-per-task=8 --time=00:30:00 ./bench_serve.sh $NUM_GPUS $EP_DEGREE $NUM_REPLICAS $BATCH_SIZE