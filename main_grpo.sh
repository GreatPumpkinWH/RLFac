set -e -x

export MODEL_PATH=/home/bml_job/custom_workspace/job-elucdpk8tscl/RL-Factory/MODEL/Qwen3-4B
export REWARD_MODEL_PATH=/home/bml_job/custom_workspace/job-elucdpk8tscl/RL-Factory/MODEL/QwQ-32B
export DATA_PATH=/home/bml_job/custom_workspace/job-elucdpk8tscl/RL-Factory/rag_server/data_process/data
export CONFIG_PATH=/home/bml_job/custom_workspace/job-elucdpk8tscl/RL-Factory/envs/configs
export RESULT_DIR=/home/bml_job/custom_workspace/job-elucdpk8tscl/RL-Factory
# 安装Ray Tune依赖
export WANDB_API_KEY=abd5c50b56e57efb5d00993f7057c4c8458ce946
pip install ray[tune] 2>&1 | tee -a ray_install.log

# 使用Ray进行超参数调优
# 创建日志目录
mkdir -p logs

# 使用Ray进行超参数调优
python3 tune_grpo.py \
    --model-path $MODEL_PATH \
    --reward-model-path $REWARD_MODEL_PATH \
    --result-dir $RESULT_DIR \
    --config-path $CONFIG_PATH \
    $@ 2>&1 | tee logs/grpo_$(date +%Y%m%d_%H%M%S).log
