#!/bin/bash

# 设置环境变量
export WANDB_NAME=test
OUTDIR=./checkpoints/$WANDB_NAME

if [ ! -d "$OUTDIR" ]; then
  mkdir -p "$OUTDIR"
fi

export DEBUG_MODE="true"
export LOG_PATH="./logs/${WANDB_NAME}.log"

# 分布式训练参数
GPUS_PER_NODE=2
NNODES=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001
WORLD_SIZE=1
DISTRIBUTED_ARGS="
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT
"

# 运行训练脚本
torchrun $DISTRIBUTED_ARGS \
    src/open_r1/grpo_caption.py \
    --deepspeed training_scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path  /home/notebook/data/group/group/Zhongchunlin/Qwen2_5_VL-finetune/ckpts/Qwen2.5-VL/70%/checkpoint-3007 \
    --train_data_path /home/notebook/data/group/group/Zhongchunlin/OwlCap/data_json/grpo_data_2k.json \
    --eval_data_path None \
    --eval_strategy "no" \
    --video_folder ./ \
    --dataset_name xxx \
    --max_prompt_length 8192 \
    --reward_funcs "format" "correctness" \
    --max_completion_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --report_to tensorboard \
    --save_steps 50 \
    --save_total_limit 10 \
    --save_only_model False \
    2>&1 | tee -a "${OUTDIR}/training_log.txt"