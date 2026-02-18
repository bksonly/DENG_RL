import argparse
import multiprocessing
import os
import random
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from env.core_env import GanDengYanEnv
from rl.ppo_agent import PPOAgent
from test_gdy import greedy_action_for_player


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 进程局部存储：每个 worker 进程只加载一次模型
_worker_ppo_agent = None
_worker_checkpoint_path = None
_worker_device_str = None


def _init_worker(checkpoint_path: str, device_str: str):
    """
    进程池初始化函数：在每个 worker 进程启动时调用一次，加载模型。
    这样可以避免每个 trial 都重复加载模型。
    """
    global _worker_ppo_agent, _worker_checkpoint_path, _worker_device_str
    _worker_checkpoint_path = checkpoint_path
    _worker_device_str = device_str
    device = torch.device(device_str)
    _worker_ppo_agent = load_ppo_agent(checkpoint_path, device)


def load_ppo_agent(checkpoint_path: str, device: torch.device) -> PPOAgent:
    """
    加载对手用的 PPOAgent（只用于推理，不再训练）。
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 用一个临时环境拿到 num_actions
    tmp_env = GanDengYanEnv()
    num_actions = tmp_env.num_actions

    # 这些超参数只影响训练阶段，此处仅用于构建网络结构，可以与训练脚本保持一致
    agent = PPOAgent(
        num_actions=num_actions,
        lr=1e-5,
        clip_eps=0.1,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=4,
        batch_size=512,
        device=device,
    )
    agent.load_checkpoint(checkpoint_path, device)
    agent.actor.eval()
    agent.critic.eval()
    return agent


def reset_opponent_hidden_states(env: GanDengYanEnv) -> None:
    """
    清空对手 LSTM hidden state，防止跨局泄漏记忆。
    """
    if getattr(env, "opponent_hidden_states", None) is not None:
        for i in range(len(env.opponent_hidden_states)):
            env.opponent_hidden_states[i] = None


def run_single_game(env: GanDengYanEnv) -> np.ndarray:
    """
    在给定环境上跑完一整局：
    - 玩家0（agent_id）使用贪心策略；
    - 其它三名玩家使用 PPO 策略（由 env.opponent_agent 控制）。
    返回：本局4个玩家各自的最终 reward，shape = [4]。
    """
    # 每局开始前清理对手 hidden state
    reset_opponent_hidden_states(env)
    state = env.reset()
    done = False
    # 本局结束时各玩家的最终 reward（从当前 env 终局状态计算）
    final_rewards: np.ndarray = np.zeros(4, dtype=np.float32)

    while not done:
        # 理论上此时应当轮到 agent_id 出牌
        if env.current_player != env.agent_id:
            # 若出现异常状态，直接标记为废局以免死循环
            # （正常情况下不应发生）
            _, _, done, _ = env.step(0)
            if done:
                break

        action_idx = greedy_action_for_player(env, env.agent_id)
        _, _, done, _ = env.step(action_idx)

        if done:
            # 终局时，根据终局状态为每个玩家单独计算 reward
            orig_agent_id = env.agent_id
            num_players = env.num_players
            for pid in range(num_players):
                env.agent_id = pid
                final_rewards[pid] = env._compute_final_reward()
            env.agent_id = orig_agent_id
            break

    return final_rewards


def run_single_trial(args: Tuple[int, int, int]) -> np.ndarray:
    """
    Worker 函数：在单个进程中运行一个 trial。
    
    参数：
        args: (trial_idx, games_per_trial, base_seed)
    注意：模型通过进程局部变量 _worker_ppo_agent 共享，避免重复加载。
    
    返回：
        该 trial 的 total_reward，shape = [4]
    """
    global _worker_ppo_agent, _worker_device_str
    trial_idx, games_per_trial, base_seed = args
    
    # 为每个 trial 设置独立的随机种子
    trial_seed = base_seed + trial_idx
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    if _worker_device_str == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial_seed)
    
    # 使用进程局部存储的 PPO agent（在进程初始化时已加载）
    if _worker_ppo_agent is None:
        raise RuntimeError("Worker PPO agent not initialized. This should not happen.")
    
    # 复用环境对象，避免重复创建开销
    # 注意：每个 trial 使用同一个环境对象，但每局游戏前会 reset，所以是安全的
    env = GanDengYanEnv(opponent_agent=_worker_ppo_agent, opponent_hidden_states=[None, None, None])
    total_reward = np.zeros(4, dtype=np.float32)
    
    for _ in range(games_per_trial):
        rewards = run_single_game(env)  # shape [4]
        total_reward += rewards
    
    return total_reward


def run_trial_batch(args: Tuple[List[int], int, int]) -> List[np.ndarray]:
    """
    Worker 函数：在单个进程中批量运行多个 trials，减少进程间通信开销。
    在 worker 内复用环境对象，避免重复创建开销。
    
    参数：
        args: (trial_indices, games_per_trial, base_seed)
    
    返回：
        该批次所有 trials 的 total_reward 列表，每个元素 shape = [4]
    """
    global _worker_ppo_agent, _worker_device_str
    trial_indices, games_per_trial, base_seed = args
    
    # 使用进程局部存储的 PPO agent（在进程初始化时已加载）
    if _worker_ppo_agent is None:
        raise RuntimeError("Worker PPO agent not initialized. This should not happen.")
    
    # 在 worker 内复用环境对象，避免重复创建开销
    env = GanDengYanEnv(opponent_agent=_worker_ppo_agent, opponent_hidden_states=[None, None, None])
    
    results = []
    for trial_idx in trial_indices:
        # 为每个 trial 设置独立的随机种子
        trial_seed = base_seed + trial_idx
        random.seed(trial_seed)
        np.random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        if _worker_device_str == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(trial_seed)
        
        # 运行 games_per_trial 局游戏（复用环境对象）
        total_reward = np.zeros(4, dtype=np.float32)
        for _ in range(games_per_trial):
            rewards = run_single_game(env)  # shape [4]
            total_reward += rewards
        
        results.append(total_reward)
    
    return results


def run_trials(
    num_trials: int,
    games_per_trial: int,
    checkpoint_path: str,
    seed: int,
    num_workers: int = 1,
) -> np.ndarray:
    """
    重复 num_trials 次实验，每次实验连续打 games_per_trial 局：
    - 玩家0使用贪心策略；
    - 其余三家使用 PPO 对手策略（latest.pt）。
    返回：shape = [num_trials, 4] 的 total_reward 数组，
    其中 axis=1 依次对应玩家0/1/2/3在该次试验中的总reward。
    
    参数：
        num_trials: 实验次数
        games_per_trial: 每次实验的游戏局数
        checkpoint_path: PPO checkpoint 路径
        seed: 基础随机种子
        num_workers: 并行 worker 数量，1 表示串行执行
    """
    # 设置基础随机种子（用于主进程）
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 确定设备（多进程时使用 CPU 更稳定，单进程可以使用 CUDA）
    if num_workers == 1:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        # 多进程时使用 CPU，避免 CUDA 多进程问题
        device_str = "cpu"
    
    # 串行执行（向后兼容）
    if num_workers == 1:
        ppo_agent = load_ppo_agent(checkpoint_path, DEVICE)
        totals: List[np.ndarray] = []
        
        # 复用环境对象，避免重复创建开销
        # 一个环境对象中，赢家连庄逻辑通过 next_dealer 自动处理
        env = GanDengYanEnv(opponent_agent=ppo_agent, opponent_hidden_states=[None, None, None])
        for _ in tqdm(range(num_trials), desc="Simulating trials"):
            total_reward = np.zeros(4, dtype=np.float32)
            for _ in range(games_per_trial):
                rewards = run_single_game(env)  # shape [4]
                total_reward += rewards
            totals.append(total_reward)
        
        return np.stack(totals, axis=0).astype(np.float32)
    
    # 多进程并行执行 - 使用真正的批量处理
    # 将 trials 分成批次，每个 worker 一次处理一个批次（包含多个 trials）
    # 这样可以减少进程间通信次数，并在 worker 内复用环境对象
    
    # 计算批量大小：每个 worker 处理约 4-8 批，平衡通信开销和负载均衡
    batch_size = max(10, num_trials // (num_workers * 20))  # 每个 worker 处理约 6 批
    
    # 将 trials 分成批次
    trial_batches = []
    for i in range(0, num_trials, batch_size):
        batch_indices = list(range(i, min(i + batch_size, num_trials)))
        trial_batches.append((batch_indices, games_per_trial, seed))
    
    print(f"Using batch processing: {len(trial_batches)} batches, {batch_size} trials per batch")
    
    # 使用进程池并行执行，initializer 确保每个进程只加载一次模型
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(checkpoint_path, device_str)
    ) as pool:
        # 使用 imap 以便 tqdm 能够实时显示进度
        batch_results = list(tqdm(
            pool.imap(run_trial_batch, trial_batches),
            total=len(trial_batches),
            desc="Simulating trials"
        ))
    
    # 展平批次结果
    totals = []
    for batch_result in batch_results:
        totals.extend(batch_result)
    
    return np.stack(totals, axis=0).astype(np.float32)


def plot_distribution(
    totals: np.ndarray,
    loss_threshold: float,
    out_path: str,
) -> None:
    """
    绘制4个玩家的 total_reward 分布直方图（2x2 子图）并标注：
    - 每个玩家的均值、方差
    - 对于玩家0（“我”），额外标注 P(total_reward <= -loss_threshold)
    """
    # totals: [num_trials, 4]
    assert totals.ndim == 2 and totals.shape[1] == 4, "totals must be [num_trials, 4]"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=False)
    axes = axes.flatten()

    player_labels = [
        "Player 0 (Me, greedy)",
        "Player 1 (PPO)",
        "Player 2 (PPO)",
        "Player 3 (PPO)",
    ]

    for pid in range(4):
        ax = axes[pid]
        data = totals[:, pid]
        mean = float(np.mean(data))
        var = float(np.var(data))

        ax.hist(data, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
        ax.axvline(mean, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean:.2f}")

        text = f"mean = {mean:.3f}\nvar = {var:.3f}"

        # 对于玩家0（“我”），额外标注 P(total_reward <= -loss_threshold)
        if pid == 0:
            prob_loss = float(np.mean(data <= -loss_threshold))
            text += f"\nP(total <= -{loss_threshold:.0f}) = {prob_loss * 100:.2f}%"

        ax.text(
            0.97,
            0.97,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_title(player_labels[pid])
        ax.set_xlabel("Total reward over games_per_trial")
        ax.set_ylabel("Frequency")
        ax.legend()

    plt.suptitle("Distribution of total reward per player (greedy vs PPO opponents)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate many 10-game sessions: player 0 uses greedy strategy, "
            "three opponents use PPO policy from checkpoints/latest.pt, and "
            "plot the distribution of total rewards."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/server/Desktop/DENG_RL/checkpoints/latest.pt",
        help="Path to PPO checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=10000,
        help="Number of trials (each trial is games-per-trial games, default: %(default)s)",
    )
    parser.add_argument(
        "--games-per-trial",
        type=int,
        default=10,
        help="Number of games per trial (default: %(default)s)",
    )
    parser.add_argument(
        "--loss-threshold",
        type=float,
        default=80.0,
        help="Loss threshold for P(total_reward <= -threshold), default: %(default)s",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--out-plot",
        type=str,
        default="/home/server/Desktop/DENG_RL/eval1/sim_total_reward_hist.png",
        help="Output path for histogram figure (default: %(default)s)",
    )
    parser.add_argument(
        "--out-data",
        type=str,
        default="/home/server/Desktop/DENG_RL/eval1/sim_total_reward.npy",
        help="Output path for raw total_reward numpy array (default: %(default)s)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: CPU count, use 1 for serial execution)",
    )

    args = parser.parse_args()

    totals = run_trials(
        num_trials=args.num_trials,
        games_per_trial=args.games_per_trial,
        checkpoint_path=args.model,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    # 保存原始数据
    os.makedirs(os.path.dirname(args.out_data), exist_ok=True)
    np.save(args.out_data, totals)

    # 绘图
    plot_distribution(
        totals=totals,
        loss_threshold=args.loss_threshold,
        out_path=args.out_plot,
    )

    # 控制台打印统计信息（按玩家）
    num_trials = totals.shape[0]
    print(f"Num trials: {num_trials}")
    print(f"Games per trial: {args.games_per_trial}")
    for pid in range(4):
        data = totals[:, pid]
        mean = float(np.mean(data))
        var = float(np.var(data))
        line = f"Player {pid} - Mean total reward: {mean:.4f}, Variance: {var:.4f}"
        if pid == 0:
            prob_loss = float(np.mean(data <= -args.loss_threshold))
            line += f", P(total_reward <= -{args.loss_threshold:.0f}): {prob_loss * 100:.4f}%"
        print(line)


if __name__ == "__main__":
    main()

