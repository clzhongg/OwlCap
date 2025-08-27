#!/bin/bash

# 检测系统GPU总数
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "检测到系统共有 $GPU_COUNT 块GPU"

# 分配vLLM使用最后两块GPU
VLLM_GPUS=$(seq $((GPU_COUNT-2)) $((GPU_COUNT-1)) | tr '\n' ',')
VLLM_GPUS=${VLLM_GPUS%,}  # 移除最后一个逗号

# 分配训练脚本使用剩余GPU
TRAIN_GPUS=$(seq 0 $((GPU_COUNT-3)) | tr '\n' ',')
TRAIN_GPUS=${TRAIN_GPUS%,}

echo "vLLM服务将使用GPU: $VLLM_GPUS"
echo "训练脚本将使用GPU: $TRAIN_GPUS"

# 启动vllm服务
echo "开始启动vllm服务..."
CUDA_VISIBLE_DEVICES=$VLLM_GPUS vllm serve /home/notebook/data/group/group/Zhongchunlin/VideoChat-R1-main/Qwen3-32B \
    --port 8000 \
    --host 0.0.0.0 \
    --dtype bfloat16 \
    --tensor-parallel-size 2 > /home/notebook/data/group/group/Zhongchunlin/VideoChat-R1-main/training_scripts/vllm_server.log 2>&1 &

VLLM_PID=$!
echo "vllm服务进程ID: $VLLM_PID"

# 等待vllm服务就绪
echo "等待vllm服务部署完成..."
while true; do
    if ss -tuln | grep -q ":8000"; then
        echo "vllm服务已成功部署！"
        break
    else
        sleep 2
    fi
done

# sleep 240
# 运行主训练脚本
echo "开始执行训练脚本..."
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS sh /home/notebook/data/group/group/Zhongchunlin/OwlCap/training_scripts/GRPO_CSER.sh

# 关闭vllm服务
echo "训练脚本执行完成，准备关闭vllm服务..."
if ps -p $VLLM_PID > /dev/null; then
    echo "正在关闭vllm服务（PID: $VLLM_PID）..."
    kill $VLLM_PID
    wait $VLLM_PID 2>/dev/null
    echo "vllm服务已成功关闭"
else
    echo "vllm服务未运行，无需关闭"
fi

echo "所有任务执行完毕"