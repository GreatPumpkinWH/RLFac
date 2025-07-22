import argparse
import subprocess
import ray
import wandb
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.wandb import wandb_mixin
from ray.tune import CLIReporter
import os
import re


def extract_metrics(log_file):
    """从训练日志中提取关键指标"""
    metrics = {
        'validation_loss': None,
        'reward_mean': None,
        'accuracy': None
    }
    
    if not os.path.exists(log_file):
        return metrics
    
    with open(log_file, 'r') as f:
        content = f.read()
        
        # 提取验证损失
        loss_match = re.search(r'validation_loss[:=]\s*([\d\.]+)', content)
        if loss_match:
            metrics['validation_loss'] = float(loss_match.group(1))
        
        # 提取平均奖励
        reward_match = re.search(r'reward_mean[:=]\s*([\d\.]+)', content)
        if reward_match:
            metrics['reward_mean'] = float(reward_match.group(1))
        
        # 提取准确率
        acc_match = re.search(r'accuracy[:=]\s*([\d\.]+)', content)
        if acc_match:
            metrics['accuracy'] = float(acc_match.group(1))
    
    return metrics


def train_func(config, args):
    # 创建唯一的实验目录
    exp_name = f"grpo_lr{config['lr']}_kl{config['kl_loss_coef']}_bs{config['train_batch_size']}"
    exp_dir = os.path.join(args.result_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    log_file = os.path.join(exp_dir, 'train.log')
    
    # 构建训练命令
    cmd = [
        "python3", "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        "data.train_files=data/nq_search/train.parquet",
        "data.val_files=data/nq_search/test.parquet",
        f"data.train_batch_size={config['train_batch_size']}",
        "data.max_prompt_length=4096",
        "data.max_response_length=512",
        f"actor_rollout_ref.model.path={args.model_path}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.actor.optim.lr={config['lr']}",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16",
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
        "actor_rollout_ref.rollout.n=4",
        "actor_rollout_ref.rollout.max_turns=2",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16",
        "actor_rollout_ref.ref.fsdp_config.param_offload=False",
        "actor_rollout_ref.rollout.enforce_eager=False",
        "actor_rollout_ref.rollout.free_cache_engine=False",
        "actor_rollout_ref.env.name=search",
        "actor_rollout_ref.env.mcp_mode=stdio",
        "actor_rollout_ref.env.tool_manager=qwen3",
        "actor_rollout_ref.env.enable_thinking=False",
        f"actor_rollout_ref.env.config_path={args.config_path}/mcp_tools.pydata",
        "actor_rollout_ref.env.use_process_reward=False",
        "reward_rollout.if_use_reward_rollout=False",
        "reward_rollout.rollout.tensor_model_parallel_size=4",
        "reward_rollout.rollout.gpu_memory_utilization=0.65",
        f"reward_rollout.rollout.model_name={args.reward_model_path}",
        "reward_rollout.rollout.free_cache_engine=False",
        "reward_rollout.rollout.response_length=2048",
        "reward_model.reward_manager=parallel",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "trainer.critic_warmup=0",
        "trainer.logger=['wandb']",
        "trainer.project_name='GRPO_search'",
        f"trainer.experiment_name={exp_name}",
        "trainer.n_gpus_per_node=8",
        "trainer.nnodes=1",
        "trainer.val_before_train=False",
        f"trainer.default_local_dir={exp_dir}",
        "trainer.default_hdfs_dir=null",
        "trainer.save_freq=20",
        "trainer.test_freq=10",
        "trainer.total_epochs=5"
    ]

    # 执行训练命令并记录日志
    with open(log_file, 'w') as f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            f.write(line)
            f.flush()
            # 实时解析并报告指标
            if 'validation_loss' in line:
                try:
                    loss = float(line.split(':')[-1].strip())
                    tune.report(validation_loss=loss)
                except ValueError:
                    pass
        process.wait()
        
        # 检查命令是否成功执行
        if process.returncode != 0:
            raise RuntimeError(f"训练命令执行失败，返回码: {process.returncode}")

    # 提取并报告最终指标
    metrics = extract_metrics(log_file)
    if metrics['validation_loss'] is not None:
        tune.report(
            validation_loss=metrics['validation_loss'],
            reward_mean=metrics['reward_mean'],
            accuracy=metrics['accuracy'],
            done=True
        )


def main(args):
    # 验证WandB API密钥
    if not os.getenv("WANDB_API_KEY"):
        raise ValueError("WANDB_API_KEY环境变量未设置，请在main_grpo.sh中配置")
    
    # 初始化WandB
    wandb.init(project="GRPO_search", name="hp_tuning")
    
    # 初始化Ray
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    
    # 定义超参数搜索空间
    search_space = {
        "lr": tune.loguniform(1e-7, 1e-5),  # 学习率搜索范围
        "kl_loss_coef": tune.uniform(0.0001, 0.01),  # KL损失系数范围
        "train_batch_size": tune.choice([64, 128, 256]),  # 批大小选项
    }
    
    # 设置早停调度器
    scheduler = ASHAScheduler(
        metric="validation_loss",
        mode="min",
        max_t=5,  # 最大训练轮数
        grace_period=1,  # 至少训练轮数
        reduction_factor=2,  # 每次减少一半试验
    )
    
    # 设置结果报告器
    reporter = CLIReporter(
        metric_columns=["validation_loss", "reward_mean", "accuracy", "training_iteration"]
    )
    
    # 创建结果目录
    os.makedirs(args.result_dir, exist_ok=True)
    
    # 运行超参数搜索
    analysis = tune.run(
        lambda config: train_func(config, args),
        config=search_space,
        scheduler=scheduler,
        num_samples=10,  # 尝试10组超参数组合
        resources_per_trial={"gpu": 8},  # 每组试验使用8个GPU
        local_dir=args.result_dir,
        name="grpo_hp_tuning",
        progress_reporter=reporter,
        keep_checkpoints_num=1,
        checkpoint_score_attr="validation_loss",
    )
    
    # 输出最佳结果
    print("最佳超参数组合:", analysis.best_config)
    print("最佳验证损失:", analysis.best_result["validation_loss"])
    
    # 生成最佳配置文件
    best_config = analysis.best_config
    with open(os.path.join(args.result_dir, "best_config.txt"), "w") as f:
        f.write("最佳超参数组合:\n")
        for key, value in best_config.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\n最佳验证损失: {analysis.best_result['validation_loss']}\n")
    
    wandb.finish()
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用Ray进行GRPO超参数调优')
    parser.add_argument('--model-path', required=True, help='模型路径')
    parser.add_argument('--reward-model-path', required=True, help='奖励模型路径')
    parser.add_argument('--result-dir', required=True, help='结果保存目录')
    parser.add_argument('--config-path', required=True, help='配置文件路径')
    args = parser.parse_args()
    main(args)