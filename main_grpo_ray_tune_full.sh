#!/bin/bash
set -e -x

# 基础环境配置
export MODEL_PATH=/home/bml_job/custom_workspace/job-elucdpk8tscl/RL-Factory/MODEL/Qwen3-4B
export REWARD_MODEL_PATH=/home/bml_job/custom_workspace/job-elucdpk8tscl/RL-Factory/MODEL/QwQ-32B
export DATA_PATH=/home/bml_job/custom_workspace/job-elucdpk8tscl/RL-Factory/rag_server/data_process/data
export CONFIG_PATH=/home/bml_job/custom_workspace/job-elucdpk8tscl/RL-Factory/envs/configs
export RESULT_DIR=/home/bml_job/custom_workspace/job-elucdpk8tscl/RL-Factory/results

export WANDB_API_KEY=abd5c50b56e57efb5d00993f7057c4c8458ce946
export HYDRA_FULL_ERROR=1
export HF_ATTN_IMPL="eager"

# Ray配置
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_memory_monitor_refresh_ms=0

# 启动Ray集群
echo "启动Ray集群..."
ray start --head --port=6379 --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265

# 等待Ray启动
sleep 5

# 检查Ray状态
ray status

# 运行Ray Tune超参数调优
echo "开始Ray Tune超参数调优..."
python3 ray_tune_grpo.py

# 停止Ray集群
echo "训练完成，停止Ray集群..."
ray stop