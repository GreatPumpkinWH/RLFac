import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.bayesopt import BayesOptSearch
import subprocess
import json
import logging

def train_grpo(config):
    """GRPO训练函数 - 兼容无optuna环境"""
    
    # 设置环境变量
    os.environ['WANDB_API_KEY'] = "abd5c50b56e57efb5d00993f7057c4c8458ce946"
    os.environ['HYDRA_FULL_ERROR'] = "1"
    os.environ['HF_ATTN_IMPL'] = "eager"
    
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
        f"trainer.experiment_name=ray_tune_trial_{config.get('trial_id', 0)}",
        "trainer.total_training_steps=200",  # 减少训练步数用于调优
        "trainer.save_freq=100",
        "trainer.test_freq=50"
    ]
    
    try:
        # 运行训练
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
        
        # 解析训练结果
        validation_reward = 0.0
        training_loss = float('inf')
        
        for line in result.stdout.split('\n'):
            line = line.lower()
            if 'validation reward' in line or 'eval reward' in line:
                try:
                    val_str = line.split(':')[-1].strip()
                    validation_reward = float(val_str.split()[0])
                except:
                    pass
            elif 'training loss' in line:
                try:
                    training_loss = float(line.split(':')[-1].strip())
                except:
                    pass
        
        # 如果无法解析，使用默认值
        if validation_reward == 0.0:
            validation_reward = -training_loss  # 使用负损失作为替代
            
        tune.report(
            validation_reward=validation_reward,
            training_loss=training_loss,
            trial_id=config.get('trial_id', 0)
        )
        
    except subprocess.TimeoutExpired:
        logging.warning(f"Trial {config.get('trial_id', 0)} timed out")
        tune.report(validation_reward=-float('inf'), training_loss=float('inf'))
    except Exception as e:
        logging.error(f"Trial {config.get('trial_id', 0)} failed: {str(e)}")
        tune.report(validation_reward=-float('inf'), training_loss=float('inf'))

def main():
    # 初始化Ray
    ray.init(
        address="auto",
        ignore_reinit_error=True,
        configure_logging=True,
        logging_level=logging.INFO
    )
    
    # 定义搜索空间
    search_space = {
        "lr": tune.loguniform(1e-7, 5e-6),
        "ppo_mini_batch_size": tune.choice([16, 32, 64]),
        "ppo_micro_batch_size": tune.choice([8, 16, 24]),
        "kl_loss_coef": tune.loguniform(1e-4, 1e-2),
        "n_rollouts": tune.choice([2, 4, 6]),
        "kl_coef": tune.loguniform(1e-4, 5e-3),
        "trial_id": tune.randint(0, 1000)
    }
    
    # 配置ASHA调度器
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="validation_reward",
        mode="max",
        max_t=500,
        grace_period=50,
        reduction_factor=2
    )
    
    # 使用Bayesian优化（不需要optuna）
    search_alg = BayesOptSearch(
        metric="validation_reward",
        mode="max",
        random_search_steps=5
    )
    
    # 运行超参数调优
    analysis = tune.run(
        train_grpo,
        config=search_space,
        num_samples=20,  # 减少样本数量
        scheduler=scheduler,
        search_alg=search_alg,
        resources_per_trial={
            "cpu": 4,
            "gpu": 2
        },
        local_dir="./ray_results",
        name="grpo_ray_tune_bayesian",
        verbose=1,
        fail_fast=False,
        max_concurrent_trials=1,  # 减少并发试验
        keep_checkpoints_num=2,
        checkpoint_score_attr="validation_reward",
        stop={"training_iteration": 500}
    )
    
    # 输出最佳配置
    print("=" * 50)
    print("最佳超参数配置:")
    best_config = analysis.get_best_config(metric="validation_reward", mode="max")
    print(json.dumps(best_config, indent=2, ensure_ascii=False))
    
    # 保存最佳配置
    with open("best_grpo_config.json", "w", encoding='utf-8') as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    
    # 生成使用最佳配置的脚本
    generate_best_config_script(best_config)
    
    # 关闭Ray
    ray.shutdown()

def generate_best_config_script(best_config):
    """生成使用最佳配置的脚本"""
    script_content = f'''#!/bin/bash
# 使用最佳超参数运行GRPO训练
set -e -x

export MODEL_PATH={os.environ.get('MODEL_PATH', '/path/to/model')}
export REWARD_MODEL_PATH={os.environ.get('REWARD_MODEL_PATH', '/path/to/reward_model')}
export DATA_PATH={os.environ.get('DATA_PATH', '/path/to/data')}
export CONFIG_PATH={os.environ.get('CONFIG_PATH', '/path/to/config')}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr={best_config.get('lr', 1e-6)} \
    actor_rollout_ref.actor.ppo_mini_batch_size={best_config.get('ppo_mini_batch_size', 32)} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={best_config.get('ppo_micro_batch_size', 16)} \
    actor_rollout_ref.actor.kl_loss_coef={best_config.get('kl_loss_coef', 0.001)} \
    actor_rollout_ref.rollout.n={best_config.get('n_rollouts', 4)} \
    algorithm.kl_ctrl.kl_coef={best_config.get('kl_coef', 0.001)} \
    trainer.total_training_steps=2000
'''
    
    with open("run_best_config.sh", "w") as f:
        f.write(script_content)
    os.chmod("run_best_config.sh", 0o755)
    print("已生成最佳配置脚本: run_best_config.sh")

if __name__ == "__main__":
    main()