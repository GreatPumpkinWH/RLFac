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

# 安装Ray完整版本（带dashboard）
echo "安装Ray完整版本..."
pip install "ray[default]" wandb -i https://pypi.tuna.tsinghua.edu.cn/simple || \
pip install "ray[default]" wandb -i https://pypi.douban.com/simple || \
pip install "ray[default]" wandb

# 启动Ray集群（简化版，不带dashboard）
echo "启动Ray集群..."
ray start --head --port=6379 || ray start --head

# 等待Ray启动
sleep 3

# 检查Ray状态
echo "Ray集群状态:"
ray status

# 运行Ray Tune超参数调优
echo "开始Ray Tune超参数调优..."
python3 ray_tune_grpo_fixed.py

# 停止Ray集群
echo "训练完成，停止Ray集群..."
ray stop

echo "调优完成！查看结果:"
echo "- 最佳配置: best_grpo_config.json"
echo "- 运行最佳配置: ./run_best_config.sh"
echo "- 结果目录: ./ray_results/"

## 修复后的Ray Tune配置（兼容minimal Ray）

### 1. 更新后的主脚本 - `main_grpo_ray_tune_fixed.sh`
```bash
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

# 安装Ray完整版本（带dashboard）
echo "安装Ray完整版本..."
pip install "ray[default]" wandb -i https://pypi.tuna.tsinghua.edu.cn/simple || \
pip install "ray[default]" wandb -i https://pypi.douban.com/simple || \
pip install "ray[default]" wandb

# 启动Ray集群（简化版，不带dashboard）
echo "启动Ray集群..."
ray start --head --port=6379 || ray start --head

# 等待Ray启动
sleep 3

# 检查Ray状态
echo "Ray集群状态:"
ray status

# 运行Ray Tune超参数调优
echo "开始Ray Tune超参数调优..."
python3 ray_tune_grpo_fixed.py

# 停止Ray集群
echo "训练完成，停止Ray集群..."
ray stop

echo "调优完成！查看结果:"
echo "- 最佳配置: best_grpo_config.json"
echo "- 运行最佳配置: ./run_best_config.sh"
echo "- 结果目录: ./ray_results/"
```