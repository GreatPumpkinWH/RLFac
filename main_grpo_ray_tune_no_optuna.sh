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
export RAY_kill_child_processes_on_worker_exit_with_raylet_subreaper=true

# 安装必要的依赖（无需optuna）
echo "安装Ray Tune依赖..."
pip install ray[tune] wandb bayesian-optimization

# 启动Ray集群
echo "启动Ray集群..."
ray start --head --port=6379 --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265

# 等待Ray启动
sleep 5

# 检查Ray状态
echo "Ray集群状态:"
ray status

# 运行Ray Tune超参数调优
echo "开始Ray Tune超参数调优（Bayesian优化）..."
python3 ray_tune_grpo_no_optuna.py

# 停止Ray集群
echo "训练完成，停止Ray集群..."
ray stop

echo "调优完成！查看结果:"
echo "- Ray Dashboard: http://localhost:8265"
echo "- 最佳配置: best_grpo_config.json"
echo "- 运行最佳配置: ./run_best_config.sh"