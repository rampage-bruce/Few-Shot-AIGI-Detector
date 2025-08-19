# example for training with 4 GPUs 
# (in our experiment we use only one GPU but with more training steps)

GPU_NUM=2
WORLD_SIZE=1
NUM_WORKERS=2
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
EXCLUDE_CLASS="glide"

# execution
OMP_NUM_THREADS=1 torchrun $DISTRIBUTED_ARGS train.py \
    --data_root "$data_root" \
    --output_dir $OUTPUT_PATH \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --batch_size 16 \
    --lr 1e-4 \
    --exclude_class $EXCLUDE_CLASS \
    --total_training_steps 200000 \
    --total_training_steps 20000 \
    --accumulation_steps 1 \
    --use_fp16 True \
    --num_support_train 5 \
    --num_support_val 10 \
    --num_query_val 30
