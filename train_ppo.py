# nohup python train_ppo.py > train_ppo.log 2>&1 & echo $! > train_ppo.pid && echo "PID: $(cat train_ppo.pid)"

# tail -f train_ppo.log
# kill $(cat train_ppo.pid)
# ps aux | grep "train_ppo.py" | grep -v grep
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
# NUM_ENVS 表示用于训练更新的环境数量
NUM_ENVS = 100  
ROLLOUT_STEPS = 512
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
PPO_BATCH_SIZE = 4096  
CLIP_EPS = 0.1
ENT_COEF = 0.01  # 增加探索：从 0.01 提升到 0.05
VF_COEF = 0.5
LR = 1e-5  # 降低学习率：从 3e-4 降到 1e-4，提高稳定性
MAX_GRAD_NORM = 0.5

# Checkpoint 配置
CHECKPOINT_DIR = "checkpoints"
RESUME_FROM_CHECKPOINT = None  # 设置为 checkpoint 文件路径以恢复训练，例如 "checkpoints/latest.pt"
CHECKPOINT_INTERVAL = 50  # 每50个episode保存一个checkpoint

# 训练配置
MAX_EPISODES = 1000  # 最大训练 episode 数

# 额外用于 vs idiot 评估的环境数量（约占训练环境的 10%）
NUM_ENVS_TRAIN = NUM_ENVS
NUM_ENVS_IDIOT = max(1, int(NUM_ENVS_TRAIN * 0.1))
TOTAL_ENVS = NUM_ENVS_TRAIN + NUM_ENVS_IDIOT

# 对手策略配置（训练用 4 种混合对手 + 额外 idiot 评估环境）
# 训练更新只使用前 NUM_ENVS_TRAIN 个环境（不含 idiot）；
# 额外 NUM_ENVS_IDIOT 个环境固定使用 idiot，对应数据只用于评估曲线。


def save_training_plot(
    episodes: List[int],
    eval_rewards_greedy: List[Optional[float]],
    eval_rewards_idiot: List[Optional[float]],
    eval_rewards_ep100: List[Optional[float]],
    eval_rewards_ep50_before: List[Optional[float]],
    save_path: str,
):
    """
    保存训练进度图：显示多条评估曲线
    - vs 贪心策略（greedy）
    - vs idiot 策略（白痴贪心）
    - vs 第100个epoch checkpoint
    - vs 当前前一个“50 区间”的 checkpoint（从 150 开始）
    """
    plt.figure(figsize=(12, 6))
    
    # vs 贪心策略
    greedy_episodes = [ep for ep, r in zip(episodes, eval_rewards_greedy) if r is not None]
    greedy_rewards = [r for r in eval_rewards_greedy if r is not None]
    if len(greedy_episodes) > 0:
        plt.plot(greedy_episodes, greedy_rewards, linewidth=1.5, alpha=0.7, color="blue", label="vs Greedy")

    # vs idiot 策略
    idiot_episodes = [ep for ep, r in zip(episodes, eval_rewards_idiot) if r is not None]
    idiot_rewards = [r for r in eval_rewards_idiot if r is not None]
    if len(idiot_episodes) > 0:
        plt.plot(idiot_episodes, idiot_rewards, linewidth=1.5, alpha=0.7, color="orange", label="vs Idiot")

    # vs 第100个epoch checkpoint
    ep100_episodes = [ep for ep, r in zip(episodes, eval_rewards_ep100) if r is not None]
    ep100_rewards = [r for r in eval_rewards_ep100 if r is not None]
    if len(ep100_episodes) > 0:
        plt.plot(ep100_episodes, ep100_rewards, linewidth=1.5, alpha=0.7, color="red", label="vs Ep100 Checkpoint")

    # vs 当前前一个“50 区间”的 checkpoint（ep50_before，从 150 开始）
    ep50_before_episodes = [ep for ep, r in zip(episodes, eval_rewards_ep50_before) if r is not None]
    ep50_before_rewards = [r for r in eval_rewards_ep50_before if r is not None]
    if len(ep50_before_episodes) > 0:
        plt.plot(ep50_before_episodes, ep50_before_rewards, linewidth=1.5, alpha=0.7, color="green", label="vs Ep50 Before Checkpoint")
    
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
    num_envs_train: int,
    num_envs_idiot: int,
    num_actions: int,
    checkpoint_dir: str,
    device: torch.device,
) -> Tuple[List, List[str]]:
    """
    设置对手策略：
    - 前 num_envs_train 个环境使用 4 种混合对手（greedy / ep100 / ep50_before / random），用于训练更新；
    - 后 num_envs_idiot 个环境固定使用 idiot，对应数据仅用于评估曲线。
    返回：(opponent_agents, opponent_strategy_types)，长度为 num_envs_train + num_envs_idiot。
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
    checkpoint_ep50_before = None

    # 第100个epoch的checkpoint（ep100）
    if episode >= 100:
        checkpoint_ep_100_path = get_checkpoint_path(100)
        if os.path.exists(checkpoint_ep_100_path):
            checkpoint_ep_100 = checkpoint_ep_100_path

    # ep50_before：使用“往前一个 50 区间”的 checkpoint（即“当前进度前 50 个 episode 的 checkpoint”）
    # 例如：
    #   - 150～199 epoch 使用 ep100
    #   - 200～249 epoch 使用 ep150
    #   - 小于 100 的阶段不使用任何历史模型（退化为 greedy）
    if episode >= 150:
        ep_before = ((episode // 50) - 1) * 50
        checkpoint_ep50_before_path = get_checkpoint_path(ep_before)
        if os.path.exists(checkpoint_ep50_before_path):
            checkpoint_ep50_before = checkpoint_ep50_before_path
    
    # 随机历史checkpoint（从所有可用checkpoint中随机选）
    random_checkpoint = None
    if len(available_checkpoints) > 0:
        # 排除当前episode和已经选中的checkpoint
        candidates = [
            (ep, path)
            for ep, path in available_checkpoints
            if ep < episode and path != checkpoint_ep_100 and path != checkpoint_ep50_before
        ]
        if len(candidates) > 0:
            _, random_checkpoint = random.choice(candidates)
    
    # 为训练环境分配混合对手（不包含 idiot）：
    # - greedy 约 40%
    # - ep100 / ep50_before / random 平分剩余的 60%
    strategies = []
    num_greedy = int(num_envs_train * 0.4)
    remaining = max(0, num_envs_train - num_greedy)
    per_rest = remaining // 3
    # 将任何余数都放到 random 上
    num_ep100 = per_rest
    num_ep50_before = per_rest
    num_random = remaining - num_ep100 - num_ep50_before

    strategies.extend(["greedy"] * num_greedy)
    strategies.extend(["ep100"] * num_ep100)
    strategies.extend(["ep50_before"] * num_ep50_before)
    strategies.extend(["random"] * num_random)
    random.shuffle(strategies)  # 随机打乱
    
    # 先为训练环境设置混合对手
    for strategy in strategies:
        if strategy == "greedy":
            opponent_agents.append(None)
            opponent_strategy_types.append("greedy")
        elif strategy == "idiot":
            opponent_agents.append(None)
            opponent_strategy_types.append("idiot")
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
        elif strategy == "ep50_before":
            if checkpoint_ep50_before is not None:
                try:
                    agent = load_checkpoint_agent(checkpoint_ep50_before, num_actions, device)
                    opponent_agents.append(agent)
                    opponent_strategy_types.append("ep50_before")
                except Exception as e:
                    print(f"Warning: Failed to load checkpoint_ep50_before: {e}, using greedy")
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
    
    # 再为额外评估环境设置固定 idiot 对手（不参与训练，只用于评估）
    for _ in range(num_envs_idiot):
        opponent_agents.append(None)
        opponent_strategy_types.append("idiot")

    return opponent_agents, opponent_strategy_types


# evaluate_vs_greedy函数已不再需要，因为评估指标直接从训练结果中提取


def train():
    # 创建 checkpoint 目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    vec_env = VectorEnv(TOTAL_ENVS)
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

    # 每个环境维护一个 Actor 的 LSTM hidden_state: (h_actor, c_actor) or None
    hidden_states: List[Tuple[torch.Tensor, torch.Tensor] or None] = [
        None for _ in range(TOTAL_ENVS)
    ]

    # 训练进度记录
    episodes_history = []
    training_rewards_history = []  # 训练时的reward
    # 多种评估指标的reward历史
    eval_rewards_greedy_history = []         # vs 贪心策略
    eval_rewards_idiot_history = []          # vs idiot 白痴策略
    eval_rewards_ep100_history = []          # vs 第100个epoch checkpoint
    eval_rewards_ep50_before_history = []    # vs ep50_before（当前进度前 50 个 episode 的 checkpoint，从150开始）
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
            # 当前仅支持最新格式的历史文件，不再兼容旧字段
            history = np.load(history_path)
            episodes_history = history["episodes"].tolist()
            training_rewards_history = history["training_rewards"].tolist()
            eval_rewards_greedy_history = history["eval_rewards_greedy"].tolist()
            eval_rewards_idiot_history = history["eval_rewards_idiot"].tolist()
            eval_rewards_ep100_history = history["eval_rewards_ep100"].tolist()
            eval_rewards_ep50_before_history = history["eval_rewards_ep50_before"].tolist()

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
        
        # 设置混合对手策略（前 NUM_ENVS_TRAIN 个用于训练，后 NUM_ENVS_IDIOT 个只用于 vs idiot 评估）
        opponent_agents, opponent_strategy_types = setup_mixed_opponents(
            current_episode,
            agent,
            NUM_ENVS_TRAIN,
            NUM_ENVS_IDIOT,
            num_actions,
            CHECKPOINT_DIR,
            DEVICE,
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
        num_aborted_games = batch.get("num_aborted_games", 0)
        aborted_ratio = batch.get("aborted_game_ratio", 0.0)
        total_steps = ROLLOUT_STEPS * TOTAL_ENVS
        agent.update(batch)

        episodes_history.append(current_episode)
        training_rewards_history.append(training_reward)
        
        # 从训练结果中统计不同对手策略下的reward
        rewards = batch["rewards"]  # [T, E]
        dones = batch["dones"]  # [T, E]
        aborted_terminals = batch.get("aborted_terminals", None)  # [T,E] or None
        opponent_strategy_types = batch["opponent_strategy_types"]  # List[str] of length E
        
        # 分别统计多种评估指标的reward
        # vs 贪心策略
        greedy_rewards = []
        # vs idiot 策略
        idiot_rewards = []
        # vs 第100个epoch checkpoint（ep100）
        ep100_rewards = []
        # vs ep50_before：当前进度前 50 个 episode 的 checkpoint（从150开始）
        ep50_before_rewards = []
        
        T, E = rewards.shape
        for t in range(T):
            for e in range(E):
                if dones[t, e]:  # 只统计终局时的reward
                    if aborted_terminals is not None and aborted_terminals[t, e]:
                        continue
                    reward = rewards[t, e]
                    strategy_type = opponent_strategy_types[e]
                    if strategy_type == "greedy":
                        greedy_rewards.append(reward)
                    elif strategy_type == "idiot":
                        idiot_rewards.append(reward)
                    elif strategy_type == "ep100":
                        ep100_rewards.append(reward)
                    elif strategy_type == "ep50_before":
                        ep50_before_rewards.append(reward)

        eval_reward_greedy = float(np.mean(greedy_rewards)) if len(greedy_rewards) > 0 else None
        eval_reward_idiot = float(np.mean(idiot_rewards)) if len(idiot_rewards) > 0 else None
        eval_reward_ep100 = float(np.mean(ep100_rewards)) if len(ep100_rewards) > 0 else None
        eval_reward_ep50_before = float(np.mean(ep50_before_rewards)) if len(ep50_before_rewards) > 0 else None

        eval_rewards_greedy_history.append(eval_reward_greedy)
        eval_rewards_idiot_history.append(eval_reward_idiot)
        eval_rewards_ep100_history.append(eval_reward_ep100)
        eval_rewards_ep50_before_history.append(eval_reward_ep50_before)

        # 更新进度条信息
        pbar.set_postfix({
            "Reward": f"{training_reward:.3f}",
            "Games": f"{num_games}/{total_steps}",
            "Abort": f"{num_aborted_games}({aborted_ratio:.2%})",
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
            eval_rewards_idiot=np.array([r if r is not None else np.nan for r in eval_rewards_idiot_history]),
            eval_rewards_ep100=np.array([r if r is not None else np.nan for r in eval_rewards_ep100_history]),
            eval_rewards_ep50_before=np.array([r if r is not None else np.nan for r in eval_rewards_ep50_before_history]),
        )
        
        # 更新最新的训练图（每10个episode更新一次，减少IO开销）
        if current_episode % 10 == 0 or current_episode == MAX_EPISODES:
            latest_plot_path = os.path.join(CHECKPOINT_DIR, "training_progress_latest.png")
            save_training_plot(
                episodes_history,
                eval_rewards_greedy_history,
                eval_rewards_idiot_history,
                eval_rewards_ep100_history,
                eval_rewards_ep50_before_history,
                latest_plot_path,
            )
    
    # 训练完成
    pbar.close()
    print(f"\nTraining completed! Total episodes: {MAX_EPISODES}")
    print(f"Best reward: {best_reward:.4f}")


if __name__ == "__main__":
    train()

