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

# Ray Tune配置
export RAY_ADDRESS="auto"
export RAY_DISABLE_IMPORT_WARNING=1

# 使用Ray Tune进行超参数调优
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=\${lr:1e-6} \
    actor_rollout_ref.actor.ppo_mini_batch_size=\${ppo_mini_batch_size:32} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=\${ppo_micro_batch_size:16} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=\${kl_loss_coef:0.001} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.state_masking=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=\${n_rollouts:4} \
    actor_rollout_ref.rollout.max_turns=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.env.name=search \
    actor_rollout_ref.env.mcp_mode=stdio \
    actor_rollout_ref.env.tool_manager=qwen3 \
    actor_rollout_ref.env.enable_thinking=False \
    actor_rollout_ref.env.config_path=$CONFIG_PATH/mcp_tools.pydata \
    actor_rollout_ref.env.use_process_reward=False \
    reward_rollout.if_use_reward_rollout=False \
    reward_rollout.rollout.tensor_model_parallel_size=4 \
    reward_rollout.rollout.gpu_memory_utilization=0.65 \
    reward_rollout.rollout.model_name=$REWARD_MODEL_PATH \
    reward_rollout.rollout.free_cache_engine=False \
    reward_rollout.rollout.response_length=2048 \
    reward_model.reward_manager=parallel \
    algorithm.kl_ctrl.kl_coef=\${kl_coef:0.001} \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb', 'ray_tune'] \
    trainer.project_name='GRPO_search' \
    trainer.experiment_name='search_ray_tune' \
    trainer.use_ray_tune=True \
    trainer.ray_tune_config.max_samples=20 \
    trainer.ray_tune_config.num_samples=10 \
    trainer.ray_tune_config.gpus_per_trial=4 \
    trainer.ray_tune_config.search_space.lr=[1e-7, 5e-7, 1e-6, 5e-6] \
    trainer.ray_tune_config.search_space.ppo_mini_batch_size=[16, 32, 64] \
    trainer.ray_tune_config.search_space.ppo_micro_batch_size=[8, 16, 32] \
    trainer.ray_tune_config.search_space.kl_loss_coef=[0.0001, 0.001, 0.01] \
    trainer.ray_tune_config.search_space.n_rollouts=[2, 4, 8] \
    trainer.ray_tune_config.search_space.kl_coef=[0.0001, 0.001, 0.01] \
    trainer.ray_tune_config.scheduler.type="asha" \
    trainer.ray_tune_config.scheduler.max_t=100 \
    trainer.ray_tune_config.scheduler.grace_period=10 \
    trainer.ray_tune_config.reuse_actors=True \
    trainer.ray_tune_config.local_mode=False