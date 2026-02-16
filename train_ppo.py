import os
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from rl.ppo_agent import PPOAgent
from rl.vector_env import VectorEnv
from rl.rollout import collect_rollout


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO 超参数（可根据实际情况调整）
NUM_ENVS = 8
ROLLOUT_STEPS = 1024  # 增加 rollout steps：从 256 提升到 1024，每个 episode 收集更多数据
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
PPO_BATCH_SIZE = 512  # 相应增加 batch size：从 256 提升到 512
CLIP_EPS = 0.2
ENT_COEF = 0.05  # 增加探索：从 0.01 提升到 0.05
VF_COEF = 0.5
LR = 1e-4  # 降低学习率：从 3e-4 降到 1e-4，提高稳定性
MAX_GRAD_NORM = 0.5

# Checkpoint 配置
CHECKPOINT_DIR = "checkpoints"
RESUME_FROM_CHECKPOINT = None  # 设置为 checkpoint 文件路径以恢复训练，例如 "checkpoints/latest.pt"

# 自对弈配置
SELF_PLAY_START_EPISODE = 100  # 从第几个 episode 开始自对弈
EVAL_STEPS = 256  # 评估时使用的 rollout steps
EVAL_INTERVAL = 10  # 每 N 个 episode 评估一次与贪心策略的对弈表现


def save_training_plot(
    episodes: List[int],
    training_rewards: List[float],
    eval_rewards: List[float],
    self_play_start: int,
    save_path: str,
):
    """
    保存训练进度图：显示训练reward和评估reward两条曲线
    """
    plt.figure(figsize=(12, 6))
    
    # 训练 reward（自对弈或贪心）
    plt.plot(episodes, training_rewards, linewidth=1.5, alpha=0.7, color="blue", label="Training Reward")
    
    # 评估 reward（与贪心对弈）
    if len(eval_rewards) > 0:
        eval_episodes = [ep for ep, r in zip(episodes, eval_rewards) if r is not None]
        eval_rewards_filtered = [r for r in eval_rewards if r is not None]
        if len(eval_episodes) > 0:
            plt.plot(eval_episodes, eval_rewards_filtered, linewidth=1.5, alpha=0.7, color="red", label="Eval vs Greedy")
    
    # 标记自对弈开始点
    if self_play_start > 0 and len(episodes) >= self_play_start:
        plt.axvline(x=self_play_start, color="green", linestyle="--", alpha=0.5, label="Self-play Start")
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Mean Reward", fontsize=12)
    plt.title("Training Progress", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_vs_greedy(agent: PPOAgent, num_envs: int, eval_steps: int) -> float:
    """
    评估 agent 与贪心策略对弈的平均 reward
    """
    eval_env = VectorEnv(num_envs, opponent_agent=None)  # 贪心策略
    hidden_states = [None for _ in range(num_envs)]
    
    rewards_list = []
    dones_list = []
    
    states = eval_env.reset()
    for _ in range(eval_steps):
        batch_actions = []
        for env_idx, env in enumerate(eval_env.envs):
            state = states[env_idx]
            hand = env._hand_counts(env.agent_id)
            legal_mask = env.get_legal_actions(
                hand, env.last_move_pattern, env.last_move_cards, must_play=env.must_play
            )
            
            from env.encoding import build_global_features
            has_active_last_move = env.last_move_pattern is not None
            global_feats = build_global_features(
                agent_id=env.agent_id,
                hands=env.hands,
                num_players=env.num_players,
                current_player=env.current_player,
                must_play=env.must_play,
                has_active_last_move=has_active_last_move,
            )
            
            action, _, _, new_hidden = agent.select_action(
                state, global_feats, legal_mask, hidden_states[env_idx]
            )
            hidden_states[env_idx] = new_hidden
            batch_actions.append(action)
        
        next_states, rewards, dones, _ = eval_env.step(batch_actions)
        rewards_list.append(rewards.copy())
        dones_list.append(dones.copy())
        states = next_states
    
    # 计算终局时的平均 reward
    rewards_arr = np.stack(rewards_list, axis=0)  # [T, E]
    dones_arr = np.stack(dones_list, axis=0)  # [T, E]
    terminal_rewards = rewards_arr[dones_arr]
    if len(terminal_rewards) > 0:
        return float(terminal_rewards.mean())
    else:
        return 0.0


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

    # 每个环境维护一个 LSTM hidden_state
    hidden_states: List[Tuple[torch.Tensor, torch.Tensor] or None] = [
        None for _ in range(NUM_ENVS)
    ]

    # 训练进度记录
    episodes_history = []
    training_rewards_history = []  # 训练时的reward（自对弈或贪心）
    eval_rewards_history = []  # 评估时的reward（与贪心对弈）
    episode = 0
    best_reward = float("-inf")
    use_self_play = False

    # 从 checkpoint 恢复训练
    if RESUME_FROM_CHECKPOINT is not None and os.path.exists(RESUME_FROM_CHECKPOINT):
        print(f"Loading checkpoint from {RESUME_FROM_CHECKPOINT}")
        episode, best_reward = agent.load_checkpoint(RESUME_FROM_CHECKPOINT, DEVICE)
        print(f"Resumed from episode {episode}, best reward: {best_reward}")
        # 加载训练历史（如果存在）
        # 支持两种格式：latest.pt -> latest_history.npz，或 best_model.pt -> latest_history.npz
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
            eval_rewards_history = history.get("eval_rewards", [None] * len(episodes_history)).tolist()
            # 检查是否应该启用自对弈
            if episode >= SELF_PLAY_START_EPISODE:
                use_self_play = True
                vec_env.set_opponent_agent(agent)
                print(f"Self-play enabled (resumed from episode {episode})")

    while True:
        # 检查是否需要切换到自对弈
        if episode == SELF_PLAY_START_EPISODE and not use_self_play:
            use_self_play = True
            vec_env.set_opponent_agent(agent)
            print(f"\n{'='*60}")
            print(f"Switching to self-play at episode {episode}")
            print(f"{'='*60}\n")
        
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

        episode += 1
        episodes_history.append(episode)
        training_rewards_history.append(training_reward)
        
        # 评估：与贪心策略对弈（每 EVAL_INTERVAL 个 episode 评估一次）
        eval_reward = None
        if episode >= SELF_PLAY_START_EPISODE and episode % EVAL_INTERVAL == 0:
            # 自对弈后，定期评估与贪心对弈的表现
            eval_reward = evaluate_vs_greedy(agent, NUM_ENVS, EVAL_STEPS)
        eval_rewards_history.append(eval_reward)

        # 计算移动平均 reward（用于更稳定的评估）
        window = min(10, len(training_rewards_history))
        recent_avg = np.mean(training_rewards_history[-window:]) if len(training_rewards_history) > 0 else training_reward

        # 更新最佳 reward（基于移动平均）
        if recent_avg > best_reward:
            best_reward = recent_avg
            # 保存最佳模型
            best_checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            agent.save_checkpoint(best_checkpoint_path, episode, best_reward)
            eval_str = f", Eval vs Greedy = {eval_reward:.4f}" if eval_reward is not None else ""
            print(
                f"Episode {episode}: Training reward = {training_reward:.4f}, Recent avg = {recent_avg:.4f}{eval_str}, "
                f"Games = {num_games}/{total_steps} steps (NEW BEST!)"
            )
        else:
            eval_str = f", Eval vs Greedy = {eval_reward:.4f}" if eval_reward is not None else ""
            print(
                f"Episode {episode}: Training reward = {training_reward:.4f}, Recent avg = {recent_avg:.4f}{eval_str}, "
                f"Games = {num_games}/{total_steps} steps"
            )

        # 每次更新 latest checkpoint 和训练图
        latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
        agent.save_checkpoint(latest_checkpoint_path, episode, best_reward)
        
        # 保存训练历史
        latest_history_path = os.path.join(CHECKPOINT_DIR, "latest_history.npz")
        np.savez(
            latest_history_path,
            episodes=np.array(episodes_history),
            training_rewards=np.array(training_rewards_history),
            eval_rewards=np.array([r if r is not None else np.nan for r in eval_rewards_history]),
        )
        
        # 更新最新的训练图
        latest_plot_path = os.path.join(CHECKPOINT_DIR, "training_progress_latest.png")
        save_training_plot(
            episodes_history,
            training_rewards_history,
            eval_rewards_history,
            SELF_PLAY_START_EPISODE,
            latest_plot_path,
        )


if __name__ == "__main__":
    train()

