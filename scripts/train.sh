# example for training with 4 GPUs 
# (in our experiment we use only one GPU but with more training steps)

# Default values
baseline="False"
frozen_clip="False"
is_dino="True"


# Determine which training script to run
if [ "$is_dino" = "True" ]; then
    TRAIN_SCRIPT="train_dino_clip.py"
elif [ "$baseline" = "True" ]; then
    TRAIN_SCRIPT="train_baseline.py"
elif [ "$frozen_clip" = "True" ]; then
    TRAIN_SCRIPT="train_frozen_clip.py"
else
    TRAIN_SCRIPT="train.py"
fi

GPU_NUM=4
WORLD_SIZE=1
NUM_WORKERS=1
SEED=42

DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
"

data_root=(
    "data/GenImage" \
)


OUTPUT_PATH='./output_dir'

# test class
EXCLUDE_CLASS="Midjourney"

# cache directory
CACHE_DIR='.dataset_cache_3'

# handle the NCCL timeout problem
#export NCCL_DEBUG=INFO
#export NCCL_ASYNC_ERROR_HANDLING=1
#export NCCL_TIMEOUT=3600  # Increase timeout to 1 hour
#export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120

# execution
OMP_NUM_THREADS=1 torchrun $DISTRIBUTED_ARGS $TRAIN_SCRIPT \
    --data_root "$data_root" \
    --output_dir $OUTPUT_PATH \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --batch_size 8 \
    --lr 1e-4 \
    --exclude_class $EXCLUDE_CLASS \
    --total_training_steps 200000 \
    --accumulation_steps 1 \
    --use_fp16 True \
    --num_support_train 5 \
    --num_support_val 10 \
    --num_query_val 30 \
    --cache_dir $CACHE_DIR \
