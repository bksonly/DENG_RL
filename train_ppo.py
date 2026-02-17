# nohup python train_ppo.py > train_ppo.log 2>&1 & echo $! > train_ppo.pid && echo "PID: $(cat train_ppo.pid)"

# tail -f train_ppo.log
# kill $(cat train_ppo.pid)
import os
import random
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from rl.ppo_agent import PPOAgent
from rl.vector_env import VectorEnv
from rl.rollout import collect_rollout


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO 超参数（可根据实际情况调整）
NUM_ENVS = 20  # 改为20，除以5能除尽，每种对手策略正好4个env
ROLLOUT_STEPS = 1024  # 增加 rollout steps：从 256 提升到 1024，每个 episode 收集更多数据
GAMMA = 0.99
GAE_LAMBDA = 0.99
PPO_EPOCHS = 4
PPO_BATCH_SIZE = 512  # 相应增加 batch size：从 256 提升到 512
CLIP_EPS = 0.1
ENT_COEF = 0.05  # 增加探索：从 0.01 提升到 0.05
VF_COEF = 0.5
LR = 1e-5  # 降低学习率：从 3e-4 降到 1e-4，提高稳定性
MAX_GRAD_NORM = 0.5

# Checkpoint 配置
CHECKPOINT_DIR = "checkpoints"
RESUME_FROM_CHECKPOINT = None  # 设置为 checkpoint 文件路径以恢复训练，例如 "checkpoints/latest.pt"
CHECKPOINT_INTERVAL = 50  # 每50个episode保存一个checkpoint

# 训练配置
MAX_EPISODES = 1000  # 最大训练 episode 数

# 对手策略配置（5种策略，每种20%）
# 20%贪心，20%当前，20%第100个epoch，20%当前前100个，20%随机历史


def save_training_plot(
    episodes: List[int],
    eval_rewards_greedy: List[Optional[float]],
    eval_rewards_ep100: List[Optional[float]],
    eval_rewards_ep100_before: List[Optional[float]],
    save_path: str,
):
    """
    保存训练进度图：显示三条评估曲线
    - vs贪心策略
    - vs第100个epoch checkpoint
    - vs当前前100个checkpoint
    """
    plt.figure(figsize=(12, 6))
    
    # vs贪心策略
    greedy_episodes = [ep for ep, r in zip(episodes, eval_rewards_greedy) if r is not None]
    greedy_rewards = [r for r in eval_rewards_greedy if r is not None]
    if len(greedy_episodes) > 0:
        plt.plot(greedy_episodes, greedy_rewards, linewidth=1.5, alpha=0.7, color="blue", label="vs Greedy")
    
    # vs第100个epoch checkpoint
    ep100_episodes = [ep for ep, r in zip(episodes, eval_rewards_ep100) if r is not None]
    ep100_rewards = [r for r in eval_rewards_ep100 if r is not None]
    if len(ep100_episodes) > 0:
        plt.plot(ep100_episodes, ep100_rewards, linewidth=1.5, alpha=0.7, color="red", label="vs Ep100 Checkpoint")
    
    # vs当前前100个checkpoint
    ep100_before_episodes = [ep for ep, r in zip(episodes, eval_rewards_ep100_before) if r is not None]
    ep100_before_rewards = [r for r in eval_rewards_ep100_before if r is not None]
    if len(ep100_before_episodes) > 0:
        plt.plot(ep100_before_episodes, ep100_before_rewards, linewidth=1.5, alpha=0.7, color="green", label="vs Ep100 Before Checkpoint")
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Mean Reward", fontsize=12)
    plt.title("Training Progress - Evaluation Metrics", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def get_checkpoint_path(episode: int) -> str:
    """获取指定episode的checkpoint路径"""
    return os.path.join(CHECKPOINT_DIR, f"checkpoint_ep{episode}.pt")


def load_checkpoint_agent(checkpoint_path: str, num_actions: int, device: torch.device) -> PPOAgent:
    """从checkpoint加载一个PPOAgent"""
    agent = PPOAgent(
        num_actions=num_actions,
        lr=LR,
        clip_eps=CLIP_EPS,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        ppo_epochs=PPO_EPOCHS,
        batch_size=PPO_BATCH_SIZE,
        device=device,
    )
    agent.load_checkpoint(checkpoint_path, device)
    return agent


def setup_mixed_opponents(
    episode: int,
    current_agent: PPOAgent,
    num_envs: int,
    num_actions: int,
    checkpoint_dir: str,
    device: torch.device,
) -> Tuple[List, List[str]]:
    """
    设置混合对手策略（5种策略，每种20%）
    返回：(opponent_agents, opponent_strategy_types)
    """
    opponent_agents = []
    opponent_strategy_types = []
    
    # 获取所有可用的checkpoint
    available_checkpoints = []
    if os.path.exists(checkpoint_dir):
        for fname in os.listdir(checkpoint_dir):
            if fname.startswith("checkpoint_ep") and fname.endswith(".pt"):
                try:
                    ep_num = int(fname.replace("checkpoint_ep", "").replace(".pt", ""))
                    if ep_num % CHECKPOINT_INTERVAL == 0:  # 只考虑每50个episode保存的checkpoint
                        available_checkpoints.append((ep_num, os.path.join(checkpoint_dir, fname)))
                except ValueError:
                    continue
    available_checkpoints.sort(key=lambda x: x[0])
    
    # 计算需要的checkpoint
    checkpoint_ep_100 = None
    checkpoint_ep_100_before = None
    
    # 第100个epoch的checkpoint
    if episode >= 100:
        checkpoint_ep_100_path = get_checkpoint_path(100)
        if os.path.exists(checkpoint_ep_100_path):
            checkpoint_ep_100 = checkpoint_ep_100_path
    
    # 当前进度前100个（向下取整）
    ep_before = (episode // 100) * 100  # 向下取整到100的倍数
    if ep_before > 0 and ep_before != episode:
        checkpoint_ep_100_before_path = get_checkpoint_path(ep_before)
        if os.path.exists(checkpoint_ep_100_before_path):
            checkpoint_ep_100_before = checkpoint_ep_100_before_path
    
    # 随机历史checkpoint（从所有可用checkpoint中随机选）
    random_checkpoint = None
    if len(available_checkpoints) > 0:
        # 排除当前episode和已经选中的checkpoint
        candidates = [
            (ep, path) for ep, path in available_checkpoints
            if ep < episode and path != checkpoint_ep_100 and path != checkpoint_ep_100_before
        ]
        if len(candidates) > 0:
            _, random_checkpoint = random.choice(candidates)
    
    # 为每个环境分配对手策略（20个env，每种策略4个）
    strategies = []
    strategies.extend(["greedy"] * 4)  # 20%贪心
    strategies.extend(["current"] * 4)  # 20%当前
    strategies.extend(["ep100"] * 4)  # 20%第100个epoch
    strategies.extend(["ep100_before"] * 4)  # 20%当前前100个
    strategies.extend(["random"] * 4)  # 20%随机历史
    random.shuffle(strategies)  # 随机打乱
    
    for strategy in strategies:
        if strategy == "greedy":
            opponent_agents.append(None)
            opponent_strategy_types.append("greedy")
        elif strategy == "current":
            opponent_agents.append(current_agent)
            opponent_strategy_types.append("current")
        elif strategy == "ep100":
            if checkpoint_ep_100 is not None:
                try:
                    agent = load_checkpoint_agent(checkpoint_ep_100, num_actions, device)
                    opponent_agents.append(agent)
                    opponent_strategy_types.append("ep100")
                except Exception as e:
                    print(f"Warning: Failed to load checkpoint_ep_100: {e}, using greedy")
                    opponent_agents.append(None)
                    opponent_strategy_types.append("greedy")
            else:
                # 如果不存在，使用贪心策略
                opponent_agents.append(None)
                opponent_strategy_types.append("greedy")
        elif strategy == "ep100_before":
            if checkpoint_ep_100_before is not None:
                try:
                    agent = load_checkpoint_agent(checkpoint_ep_100_before, num_actions, device)
                    opponent_agents.append(agent)
                    opponent_strategy_types.append("ep100_before")
                except Exception as e:
                    print(f"Warning: Failed to load checkpoint_ep_100_before: {e}, using greedy")
                    opponent_agents.append(None)
                    opponent_strategy_types.append("greedy")
            else:
                # 如果不存在，使用贪心策略
                opponent_agents.append(None)
                opponent_strategy_types.append("greedy")
        elif strategy == "random":
            if random_checkpoint is not None:
                try:
                    agent = load_checkpoint_agent(random_checkpoint, num_actions, device)
                    opponent_agents.append(agent)
                    opponent_strategy_types.append("random")
                except Exception as e:
                    print(f"Warning: Failed to load random checkpoint: {e}, using greedy")
                    opponent_agents.append(None)
                    opponent_strategy_types.append("greedy")
            else:
                # 如果不存在，使用贪心策略
                opponent_agents.append(None)
                opponent_strategy_types.append("greedy")
    
    return opponent_agents, opponent_strategy_types


# evaluate_vs_greedy函数已不再需要，因为评估指标直接从训练结果中提取


def train():
    # 创建 checkpoint 目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    vec_env = VectorEnv(NUM_ENVS)
    dummy_env = vec_env.envs[0]
    num_actions = dummy_env.num_actions

    agent = PPOAgent(
        num_actions=num_actions,
        lr=LR,
        clip_eps=CLIP_EPS,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        ppo_epochs=PPO_EPOCHS,
        batch_size=PPO_BATCH_SIZE,
        device=DEVICE,
    )

    # 每个环境维护一个 LSTM hidden_state (现在是4个tensor: h_actor, c_actor, h_critic, c_critic)
    hidden_states: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] or None] = [
        None for _ in range(NUM_ENVS)
    ]

    # 训练进度记录
    episodes_history = []
    training_rewards_history = []  # 训练时的reward（自对弈或贪心）
    # 三种评估指标的reward历史
    eval_rewards_greedy_history = []  # vs贪心策略的reward
    eval_rewards_ep100_history = []  # vs第100个epoch checkpoint的reward
    eval_rewards_ep100_before_history = []  # vs当前前100个checkpoint的reward
    episode = 0
    best_reward = float("-inf")

    # 从 checkpoint 恢复训练
    start_episode = 0
    if RESUME_FROM_CHECKPOINT is not None and os.path.exists(RESUME_FROM_CHECKPOINT):
        print(f"Loading checkpoint from {RESUME_FROM_CHECKPOINT}")
        start_episode, best_reward = agent.load_checkpoint(RESUME_FROM_CHECKPOINT, DEVICE)
        print(f"Resumed from episode {start_episode}, best reward: {best_reward}")
        # 加载训练历史（如果存在）
        if RESUME_FROM_CHECKPOINT.endswith("latest.pt"):
            history_path = os.path.join(CHECKPOINT_DIR, "latest_history.npz")
        elif RESUME_FROM_CHECKPOINT.endswith("best_model.pt"):
            history_path = os.path.join(CHECKPOINT_DIR, "latest_history.npz")
        else:
            history_path = RESUME_FROM_CHECKPOINT.replace(".pt", "_history.npz")
        if os.path.exists(history_path):
            history = np.load(history_path)
            episodes_history = history["episodes"].tolist()
            training_rewards_history = history.get("training_rewards", history.get("rewards", [])).tolist()
            eval_rewards_greedy_history = history.get("eval_rewards_greedy", []).tolist()
            eval_rewards_ep100_history = history.get("eval_rewards_ep100", []).tolist()
            eval_rewards_ep100_before_history = history.get("eval_rewards_ep100_before", []).tolist()

    # 创建进度条
    pbar = tqdm(
        range(start_episode, MAX_EPISODES),
        desc=f"Training",
        unit="episode",
        initial=start_episode,
        total=MAX_EPISODES,
    )

    for episode in pbar:
        # episode 在循环中从 start_episode 到 MAX_EPISODES-1，转换为从 1 开始的计数
        current_episode = episode + 1
        
        # 设置混合对手策略
        opponent_agents, opponent_strategy_types = setup_mixed_opponents(
            current_episode, agent, NUM_ENVS, num_actions, CHECKPOINT_DIR, DEVICE
        )
        vec_env.set_opponent_agents(opponent_agents, opponent_strategy_types)
        
        batch = collect_rollout(
            vec_env,
            agent,
            hidden_states,
            rollout_steps=ROLLOUT_STEPS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
        )
        training_reward = batch["mean_reward"]
        num_games = batch["num_games_completed"]
        total_steps = ROLLOUT_STEPS * NUM_ENVS
        agent.update(batch)

        episodes_history.append(current_episode)
        training_rewards_history.append(training_reward)
        
        # 从训练结果中统计不同对手策略下的reward
        rewards = batch["rewards"]  # [T, E]
        dones = batch["dones"]  # [T, E]
        opponent_strategy_types = batch["opponent_strategy_types"]  # List[str] of length E
        
        # 分别统计三种评估指标的reward
        # vs贪心策略
        greedy_rewards = []
        # vs第100个epoch checkpoint
        ep100_rewards = []
        # vs当前前100个checkpoint
        ep100_before_rewards = []
        
        T, E = rewards.shape
        for t in range(T):
            for e in range(E):
                if dones[t, e]:  # 只统计终局时的reward
                    reward = rewards[t, e]
                    strategy_type = opponent_strategy_types[e]
                    if strategy_type == "greedy":
                        greedy_rewards.append(reward)
                    elif strategy_type == "ep100":
                        ep100_rewards.append(reward)
                    elif strategy_type == "ep100_before":
                        ep100_before_rewards.append(reward)
        
        eval_reward_greedy = float(np.mean(greedy_rewards)) if len(greedy_rewards) > 0 else None
        eval_reward_ep100 = float(np.mean(ep100_rewards)) if len(ep100_rewards) > 0 else None
        eval_reward_ep100_before = float(np.mean(ep100_before_rewards)) if len(ep100_before_rewards) > 0 else None
        
        eval_rewards_greedy_history.append(eval_reward_greedy)
        eval_rewards_ep100_history.append(eval_reward_ep100)
        eval_rewards_ep100_before_history.append(eval_reward_ep100_before)

        # 计算移动平均 reward（用于更稳定的评估）
        window = min(10, len(training_rewards_history))
        recent_avg = np.mean(training_rewards_history[-window:]) if len(training_rewards_history) > 0 else training_reward

        # 更新最佳 reward（基于移动平均）
        if recent_avg > best_reward:
            best_reward = recent_avg
            # 保存最佳模型
            best_checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            agent.save_checkpoint(best_checkpoint_path, current_episode, best_reward)
        
        # 更新进度条信息
        pbar.set_postfix({
            "Reward": f"{training_reward:.3f}",
            "Games": f"{num_games}/{total_steps}"
        })


        # 每50个episode保存一个checkpoint
        if current_episode % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = get_checkpoint_path(current_episode)
            agent.save_checkpoint(checkpoint_path, current_episode, best_reward)
            pbar.write(f"Saved checkpoint: {checkpoint_path}")
        
        # 每次更新 latest checkpoint
        latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
        agent.save_checkpoint(latest_checkpoint_path, current_episode, best_reward)
        
        # 保存训练历史
        latest_history_path = os.path.join(CHECKPOINT_DIR, "latest_history.npz")
        np.savez(
            latest_history_path,
            episodes=np.array(episodes_history),
            training_rewards=np.array(training_rewards_history),
            eval_rewards_greedy=np.array([r if r is not None else np.nan for r in eval_rewards_greedy_history]),
            eval_rewards_ep100=np.array([r if r is not None else np.nan for r in eval_rewards_ep100_history]),
            eval_rewards_ep100_before=np.array([r if r is not None else np.nan for r in eval_rewards_ep100_before_history]),
        )
        
        # 更新最新的训练图（每10个episode更新一次，减少IO开销）
        if current_episode % 10 == 0 or current_episode == MAX_EPISODES:
            latest_plot_path = os.path.join(CHECKPOINT_DIR, "training_progress_latest.png")
            save_training_plot(
                episodes_history,
                eval_rewards_greedy_history,
                eval_rewards_ep100_history,
                eval_rewards_ep100_before_history,
                latest_plot_path,
            )
    
    # 训练完成
    pbar.close()
    print(f"\nTraining completed! Total episodes: {MAX_EPISODES}")
    print(f"Best reward: {best_reward:.4f}")


if __name__ == "__main__":
    train()

