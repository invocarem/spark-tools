# On Node 1, enter container and start server
export VLLM_CONTAINER="vllm_node"

# On spark1, when launching vLLM
docker exec -it $VLLM_CONTAINER /bin/bash -c "
  vllm serve Qwen/Qwen3.5-2B \
    --tensor-parallel-size 1 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.4 \
    --distributed-executor-backend ray"

