"""
采样轨迹（rollout）相关逻辑。
"""

from typing import List, Dict, Any, Tuple

import numpy as np
import torch

from env.encoding import build_global_features, build_state, build_state_for_critic
from .ppo_agent import PPOAgent, compute_gae
from .vector_env import VectorEnv


def collect_rollout(
    vec_env: VectorEnv,
    agent: PPOAgent,
    hidden_states: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] or None],
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
) -> Dict[str, Any]:
    """
    收集一段长度为 rollout_steps 的轨迹。
    返回打平后的训练 batch 所需的所有字段。
    """
    states = vec_env.reset()

    actor_states_buf = []
    critic_states_buf = []
    actions_buf = []
    log_probs_buf = []
    rewards_buf = []
    dones_buf = []
    values_buf = []
    legal_masks_buf = []
    global_feats_buf = []

    for _ in range(rollout_steps):
        batch_actions: List[int] = []
        batch_log_probs: List[float] = []
        batch_values: List[float] = []
        batch_legal_masks: List[np.ndarray] = []
        batch_global_feats: List[np.ndarray] = []
        batch_actor_states: List[np.ndarray] = []
        batch_critic_states: List[np.ndarray] = []

        for env_idx, env in enumerate(vec_env.envs):
            state = states[env_idx]  # 这是从env返回的状态，目前是3通道的actor状态
            
            # 构建actor状态（3通道：自己的手牌、墓地、目标牌）
            actor_state = build_state(
                agent_id=env.agent_id,
                hands=env.hands,
                num_players=env.num_players,
                graveyard=env.graveyard,
                last_move_cards=env.last_move_cards,
                last_move_player=env.last_move_player,
                current_player=env.current_player,
                must_play=env.must_play,
                has_active_last_move=env.last_move_pattern is not None,
            )
            
            # 构建critic状态（6通道：自己的手牌、3个对手手牌、墓地、目标牌）
            critic_state = build_state_for_critic(
                agent_id=env.agent_id,
                hands=env.hands,
                num_players=env.num_players,
                graveyard=env.graveyard,
                last_move_cards=env.last_move_cards,
                last_move_player=env.last_move_player,
                current_player=env.current_player,
                must_play=env.must_play,
                has_active_last_move=env.last_move_pattern is not None,
            )
            
            # 计算当前合法动作 mask
            hand = env._hand_counts(env.agent_id)
            legal_mask = env.get_legal_actions(
                hand, env.last_move_pattern, env.last_move_cards, must_play=env.must_play
            )

            has_active_last_move = env.last_move_pattern is not None
            global_feats = build_global_features(
                agent_id=env.agent_id,
                hands=env.hands,
                num_players=env.num_players,
                current_player=env.current_player,
                must_play=env.must_play,
                has_active_last_move=has_active_last_move,
            )

            action, log_prob, value, new_hidden = agent.select_action(
                actor_state, critic_state, global_feats, legal_mask, hidden_states[env_idx]
            )
            hidden_states[env_idx] = new_hidden

            batch_actions.append(action)
            batch_log_probs.append(log_prob)
            batch_values.append(value)
            batch_legal_masks.append(legal_mask)
            batch_global_feats.append(global_feats)
            batch_actor_states.append(actor_state)
            batch_critic_states.append(critic_state)

        next_states, rewards, dones, infos = vec_env.step(batch_actions)

        actor_states_buf.append(np.stack(batch_actor_states, axis=0))
        critic_states_buf.append(np.stack(batch_critic_states, axis=0))
        actions_buf.append(np.array(batch_actions, dtype=np.int64))
        log_probs_buf.append(np.array(batch_log_probs, dtype=np.float32))
        rewards_buf.append(rewards.copy())
        dones_buf.append(dones.copy())
        values_buf.append(np.array(batch_values, dtype=np.float32))
        legal_masks_buf.append(np.stack(batch_legal_masks, axis=0))
        global_feats_buf.append(np.stack(batch_global_feats, axis=0))

        states = next_states

    # 最后一个 step 的 value 用于 GAE
    last_values: List[float] = []
    for env_idx, env in enumerate(vec_env.envs):
        # 构建最后一个step的critic状态（只需要critic，不需要actor）
        critic_state = build_state_for_critic(
            agent_id=env.agent_id,
            hands=env.hands,
            num_players=env.num_players,
            graveyard=env.graveyard,
            last_move_cards=env.last_move_cards,
            last_move_player=env.last_move_player,
            current_player=env.current_player,
            must_play=env.must_play,
            has_active_last_move=env.last_move_pattern is not None,
        )
        has_active_last_move = env.last_move_pattern is not None
        global_feats = build_global_features(
            agent_id=env.agent_id,
            hands=env.hands,
            num_players=env.num_players,
            current_player=env.current_player,
            must_play=env.must_play,
            has_active_last_move=has_active_last_move,
        )
        critic_state_t = torch.from_numpy(critic_state).float().unsqueeze(0).to(agent.device)
        global_feats_t = (
            torch.from_numpy(global_feats).float().unsqueeze(0).to(agent.device)
        )
        
        # 只调用 critic，提取 critic 的 hidden state
        hidden = hidden_states[env_idx]
        critic_hidden = None
        if hidden is not None:
            # hidden 格式是 (h_actor, c_actor, h_critic, c_critic)
            _, _, h_critic, c_critic = hidden
            critic_hidden = (h_critic, c_critic)
        
        value_t, _ = agent.critic(critic_state_t, global_feats_t, critic_hidden)
        last_values.append(float(value_t.item()))
    last_values_arr = np.array(last_values, dtype=np.float32)

    actor_states = np.stack(actor_states_buf, axis=0)  # [T, E, 3, 4, 15]
    critic_states = np.stack(critic_states_buf, axis=0)  # [T, E, 6, 4, 15]
    actions = np.stack(actions_buf, axis=0)  # [T, E]
    log_probs = np.stack(log_probs_buf, axis=0)  # [T, E]
    rewards = np.stack(rewards_buf, axis=0)  # [T, E]
    dones = np.stack(dones_buf, axis=0)  # [T, E]
    values = np.stack(values_buf, axis=0)  # [T, E]
    legal_masks = np.stack(legal_masks_buf, axis=0)  # [T, E, A]
    global_feats = np.stack(global_feats_buf, axis=0)  # [T, E, 5]

    returns, advantages = compute_gae(rewards, values, dones, last_values_arr, gamma, gae_lambda)

    # 打平成 [T*E, ...]
    T, E = rewards.shape
    N = T * E
    actor_states_flat = actor_states.reshape(T * E, *actor_states.shape[2:])
    critic_states_flat = critic_states.reshape(T * E, *critic_states.shape[2:])
    actions_flat = actions.reshape(N)
    log_probs_flat = log_probs.reshape(N)
    returns_flat = returns.reshape(N)
    advantages_flat = advantages.reshape(N)
    legal_masks_flat = legal_masks.reshape(N, legal_masks.shape[-1])
    global_feats_flat = global_feats.reshape(N, global_feats.shape[-1])

    advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

    # 统计完成的游戏局数（dones 为 True 的次数）
    num_games_completed = int(dones.sum())
    
    # 计算平均 reward：只统计终局时的 reward（dones=True 时的 reward 的平均值）
    # 在稀疏奖励设置下，只有终局才有非零 reward
    terminal_rewards = rewards[dones]
    if len(terminal_rewards) > 0:
        mean_reward = float(terminal_rewards.mean())
    else:
        # 如果没有完成的游戏，返回 0
        mean_reward = 0.0

    # 记录每个环境的对手策略类型（用于后续统计不同对手策略下的reward）
    opponent_strategy_types = vec_env.opponent_strategy_types if hasattr(vec_env, 'opponent_strategy_types') else None

    batch = {
        "actor_states": torch.from_numpy(actor_states_flat).float(),
        "critic_states": torch.from_numpy(critic_states_flat).float(),
        "actions": torch.from_numpy(actions_flat).long(),
        "log_probs": torch.from_numpy(log_probs_flat).float(),
        "returns": torch.from_numpy(returns_flat).float(),
        "advantages": torch.from_numpy(advantages_flat).float(),
        "legal_masks": torch.from_numpy(legal_masks_flat.astype(np.bool_)),
        "global_feats": torch.from_numpy(global_feats_flat).float(),
        "mean_reward": mean_reward,
        "num_games_completed": num_games_completed,
        "rewards": rewards,  # 保留原始rewards用于按对手策略类型统计
        "dones": dones,  # 保留原始dones用于按对手策略类型统计
        "opponent_strategy_types": opponent_strategy_types,  # 每个环境的对手策略类型
    }

    return batch

