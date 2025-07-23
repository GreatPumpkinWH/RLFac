import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import subprocess
import json

def train_grpo(config):
    """GRPO训练函数"""
    
    # 构建训练命令
    cmd = [
        "python3", "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={os.environ['DATA_PATH']}/train.parquet",
        f"data.val_files={os.environ['DATA_PATH']}/test.parquet",
        "data.train_batch_size=128",
        "data.max_prompt_length=4096",
        "data.max_response_length=512",
        f"actor_rollout_ref.model.path={os.environ['MODEL_PATH']}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.actor.optim.lr={config['lr']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={config['ppo_micro_batch_size']}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={config['kl_loss_coef']}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.actor.state_masking=True",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.75",
        f"actor_rollout_ref.rollout.n={config['n_rollouts']}",
        "actor_rollout_ref.rollout.max_turns=2",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.ref.fsdp_config.param_offload=False",
        "actor_rollout_ref.rollout.enforce_eager=False",
        "actor_rollout_ref.rollout.free_cache_engine=False",
        "actor_rollout_ref.env.name=search",
        "actor_rollout_ref.env.mcp_mode=stdio",
        "actor_rollout_ref.env.tool_manager=qwen3",
        "actor_rollout_ref.env.enable_thinking=False",
        f"actor_rollout_ref.env.config_path={os.environ['CONFIG_PATH']}/mcp_tools.pydata",
        "actor_rollout_ref.env.use_process_reward=False",
        "reward_rollout.if_use_reward_rollout=False",
        "reward_rollout.rollout.tensor_model_parallel_size=4",
        "reward_rollout.rollout.gpu_memory_utilization=0.65",
        f"reward_rollout.rollout.model_name={os.environ['REWARD_MODEL_PATH']}",
        "reward_rollout.rollout.free_cache_engine=False",
        "reward_rollout.rollout.response_length=2048",
        "reward_model.reward_manager=parallel",
        f"algorithm.kl_ctrl.kl_coef={config['kl_coef']}",
        "trainer.critic_warmup=0",
        "trainer.logger=['wandb']",
        "trainer.project_name='GRPO_search_ray_tune'",
        f"trainer.experiment_name=ray_tune_trial_{config['trial_id']}",
        "trainer.total_training_steps=1000"
    ]
    
    # 运行训练
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 解析结果（这里需要根据实际情况调整）
    # 假设训练脚本会输出validation reward
    validation_reward = 0.0
    for line in result.stdout.split('\n'):
        if 'validation reward' in line.lower():
            try:
                validation_reward = float(line.split(':')[-1].strip())
                break
            except:
                pass
    
    # 返回结果给Ray Tune
    tune.report(validation_reward=validation_reward)

def main():
    # 初始化Ray
    ray.init(address="auto")
    
    # 定义搜索空间
    search_space = {
        "lr": tune.loguniform(1e-7, 1e-5),
        "ppo_mini_batch_size": tune.choice([16, 32, 64, 128]),
        "ppo_micro_batch_size": tune.choice([8, 16, 32]),
        "kl_loss_coef": tune.loguniform(1e-4, 1e-2),
        "n_rollouts": tune.choice([2, 4, 8, 16]),
        "kl_coef": tune.loguniform(1e-4, 1e-2),
        "trial_id": tune.randint(0, 10000)
    }
    
    # 配置ASHA调度器
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="validation_reward",
        mode="max",
        max_t=1000,
        grace_period=50,
        reduction_factor=2
    )
    
    # 配置Optuna搜索算法
    search_alg = OptunaSearch(
        metric="validation_reward",
        mode="max"
    )
    
    # 运行超参数调优
    analysis = tune.run(
        train_grpo,
        config=search_space,
        num_samples=30,
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial={
            "cpu": 8,
            "gpu": 4
        },
        local_dir="./ray_results",
        name="grpo_ray_tune_search",
        verbose=1,
        fail_fast=False,
        max_concurrent_trials=2,
        keep_checkpoints_num=3,
        checkpoint_score_attr="validation_reward"
    )
    
    # 输出最佳配置
    print("最佳配置:")
    best_config = analysis.get_best_config(metric="validation_reward", mode="max")
    print(json.dumps(best_config, indent=2))
    
    # 保存最佳配置
    with open("best_grpo_config.json", "w") as f:
        json.dump(best_config, f, indent=2)
    
    # 关闭Ray
    ray.shutdown()

if __name__ == "__main__":
    main()