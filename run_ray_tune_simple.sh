#!/bin/bash
set -e

echo "=== Ray Tune GRPO 超参数调优（兼容版） ==="

# 检查并设置环境变量
export MODEL_PATH=${MODEL_PATH:-"/home/bml_job/custom_workspace/job-elucdpk8tscl/RL-Factory/MODEL/Qwen3-4B"}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-"/home/bml_job/custom_workspace/job-elucdpk8tscl/RL-Factory/MODEL/QwQ-32B"}
export DATA_PATH=${DATA_PATH:-"/home/bml_job/custom_workspace/job-elucdpk8tscl/RL-Factory/rag_server/data_process/data"}
export CONFIG_PATH=${CONFIG_PATH:-"/home/bml_job/custom_workspace/job-elucdpk8tscl/RL-Factory/envs/configs"}

# 安装完整版Ray
echo "安装Ray完整版本..."
pip install "ray[default]" -i https://pypi.tuna.tsinghua.edu.cn/simple || \
pip install "ray[default]" -i https://pypi.douban.com/simple || \
pip install "ray[default]"

# 启动Ray（无dashboard）
echo "启动Ray集群..."
ray start --head --port=6379 --include-dashboard=False || \
ray start --head --port=6379

# 检查状态
echo "Ray集群状态:"
ray status

# 运行调优
echo "开始超参数调优..."
python3 ray_tune_grpo_fixed.py

# 清理
echo "停止Ray集群..."
ray stop

echo "=== 调优完成 ==="
echo "查看结果:"
echo "- 最佳配置: best_grpo_config.json"
echo "- 运行最佳配置: ./run_best_config.sh"
echo "- 详细结果: ray_results/"