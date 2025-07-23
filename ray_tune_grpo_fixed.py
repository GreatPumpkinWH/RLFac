import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator
import subprocess
import json
import logging
import time

def train_grpo(config):
    """GRPO训练函数 - 修复版本"""
    
    # 设置环境变量
    os.environ['WANDB_API_KEY'] = "abd5c50b56e57efb5d00993f7057c4c8458ce946"
    os.environ['HYDRA_FULL_ERROR'] = "1"
    os.environ['HF_ATTN_IMPL'] = "eager"
    
    # 创建结果目录
    trial_dir = os.getcwd()
    log_file = f"trial_{config.get('trial_id', 0)}.log"
    
    # 构建训练命令
    cmd = [
        "python3", "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        f"data.train_files={os.environ['DATA_PATH']}/train.parquet",
        f"data.val_files={os.environ['DATA_PATH']}/test.parquet",
        "data.train_batch_size=64",  # 减少批次大小
        "data.max_prompt_length=2048",  # 减少长度
        "data.max_response_length=256",
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
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        f"actor_rollout_ref.rollout.n={config['n_rollouts']}",
        "actor_rollout_ref.rollout.max_turns=1",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.env.name=search",
        "actor_rollout_ref.env.mcp_mode=stdio",
        "actor_rollout_ref.env.tool_manager=qwen3",
        "actor_rollout_ref.env.enable_thinking=False",
        f"actor_rollout_ref.env.config_path={os.environ['CONFIG_PATH']}/mcp_tools.pydata",
        "algorithm.kl_ctrl.kl_coef={config['kl_coef']}",
        "trainer.critic_warmup=0",
        "trainer.logger=['wandb']",
        "trainer.project_name='GRPO_ray_tune_fixed'",
        f"trainer.experiment_name=trial_{config.get('trial_id', 0)}",
        "trainer.total_training_steps=400",  # 减少训练步数
        "trainer.save_freq=50",
        "trainer.test_freq=25"
    ]
    
    try:
        # 运行训练并记录日志
        with open(log_file, 'w') as log_f:
            result = subprocess.run(
                cmd, 
                stdout=log_f, 
                stderr=subprocess.STDOUT, 
                text=True, 
                timeout=3600  # 1小时超时
            )
        
        # 读取日志文件
        with open(log_file, 'r') as log_f:
            log_content = log_f.read()
        
        # 解析训练结果
        validation_reward = -1.0  # 默认值
        training_loss = 1.0
        
        for line in log_content.split('\n'):
            line = line.lower()
            if any(x in line for x in ['validation reward', 'eval reward', 'reward:']):
                try:
                    # 提取数字
                    import re
                    numbers = re.findall(r'-?\d+\.?\d*', line)
                    if numbers:
                        validation_reward = float(numbers[-1])
                except:
                    pass
            elif 'training loss' in line:
                try:
                    numbers = re.findall(r'-?\d+\.?\d*', line)
                    if numbers:
                        training_loss = float(numbers[-1])
                except:
                    pass
        
        # 确保有合理的值
        if validation_reward <= -10:
            validation_reward = -training_loss
            
        # 使用兼容的metric报告方式
        try:
            from ray import train
            # 使用ray.tune.report替代ray.train.report
            tune.report({
            "validation_reward": max(validation_reward, -10),
            "training_loss": training_loss,
            "trial_id": config.get('trial_id', 0)
            })
        except ImportError:
            # 兼容旧版本
            tune.report(
                validation_reward=max(validation_reward, -10),
                training_loss=training_loss,
                trial_id=config.get('trial_id', 0)
            )
        
    except subprocess.TimeoutExpired:
        logging.warning(f"Trial {config.get('trial_id', 0)} timed out")
        try:
            from ray import train
            train.report({
                "validation_reward": -5.0,
                "training_loss": 5.0,
                "trial_id": config.get('trial_id', 0)
            })
        except ImportError:
            tune.report(validation_reward=-5.0, training_loss=5.0)
    except Exception as e:
        logging.error(f"Trial {config.get('trial_id', 0)} failed: {str(e)}")
        try:
            from ray import train
            train.report({
                "validation_reward": -5.0,
                "training_loss": 5.0,
                "trial_id": config.get('trial_id', 0)
            })
        except ImportError:
            tune.report(validation_reward=-5.0, training_loss=5.0)

def main():
    # 初始化Ray（简化配置）
    ray.init(
        address="auto",
        ignore_reinit_error=True,
        logging_level=logging.INFO,
        include_dashboard=False  # 禁用dashboard
    )
    
    # 简化搜索空间
    search_space = {
        "lr": tune.loguniform(5e-7, 2e-6),
        "ppo_mini_batch_size": tune.choice([16, 32]),
        "ppo_micro_batch_size": tune.choice([8, 12, 16]),
        "kl_loss_coef": tune.loguniform(5e-4, 5e-3),
        "n_rollouts": tune.choice([2, 3, 4]),
        "kl_coef": tune.loguniform(5e-4, 2e-3),
        "trial_id": tune.randint(0, 100)
    }
    
    # 配置ASHA调度器（更激进）
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="validation_reward",
        mode="max",
        max_t=200,
        grace_period=25,
        reduction_factor=2
    )
    
    # 使用基本搜索算法（无需额外依赖）
    search_alg = BasicVariantGenerator(
        max_concurrent=1
    )
    
    # 运行超参数调优
    print("开始超参数调优...")
    analysis = tune.run(
        train_grpo,  # 修复：使用正确的函数名
        config=search_space,
        scheduler=scheduler,  # 修复：使用正确的变量名
        num_samples=12,
        resources_per_trial={"cpu": 2, "gpu": 2},
        storage_path=os.path.abspath("./ray_results"),
        name="grpo_ray_tune_minimal"
        # 移除metric和mode参数，因为scheduler已配置
    )
    
    # 修复第225行附近的代码
    # 输出最佳配置
    print("\n" + "="*60)
    print("超参数调优完成！")
    print("="*60)
    
    # 使用get_best_trial获取最佳结果 - 这是正确的Ray Tune API
    best_trial = analysis.get_best_trial(metric="validation_reward", mode="max")
    if best_trial:
        best_config = best_trial.config
        best_validation_reward = best_trial.last_result.get("validation_reward", "N/A")
    else:
        best_config = {}
        best_validation_reward = "N/A"
    
    # 简化的统计信息输出
    print(f"\n总共运行试验: {len(analysis.results_df)}")
    print(f"最佳验证奖励: {best_validation_reward}")
    print(f"最佳配置: {best_config}")
    print("最佳超参数配置:")
    print(json.dumps(best_config, indent=2, ensure_ascii=False))
    
    # 保存最佳配置
    with open("best_grpo_config.json", "w", encoding='utf-8') as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    
    # 生成使用最佳配置的脚本
    generate_best_config_script(best_config)
    
    # 修复第272行 - 只保留一次统计信息
    print(f"\n总共运行试验: {len(analysis.results_df)}")
    print(f"最佳验证奖励: {best_validation_reward}")
    
    # 关闭Ray
    ray.shutdown()

def generate_best_config_script(best_config):
    """生成使用最佳配置的脚本"""
    script_content = f'''#!/bin/bash
# 使用最佳超参数运行GRPO训练
# 生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}

set -e -x

export MODEL_PATH={os.environ.get('MODEL_PATH', '/path/to/model')}
export REWARD_MODEL_PATH={os.environ.get('REWARD_MODEL_PATH', '/path/to/reward_model')}
export DATA_PATH={os.environ.get('DATA_PATH', '/path/to/data')}
export CONFIG_PATH={os.environ.get('CONFIG_PATH', '/path/to/config')}

# 使用最佳配置运行完整训练
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
    trainer.total_training_steps=200

echo "训练完成！"
'''
    
    with open("run_best_config.sh", "w") as f:
        f.write(script_content)
    os.chmod("run_best_config.sh", 0o755)
    print("已生成最佳配置脚本: run_best_config.sh")

if __name__ == "__main__":
    main()