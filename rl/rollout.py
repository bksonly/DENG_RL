"""
采样轨迹（rollout）相关逻辑。
"""

from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch

from env.encoding import build_global_features, build_state, build_state_for_critic
from .ppo_agent import PPOAgent, compute_gae
from .vector_env import VectorEnv


def collect_rollout(
    vec_env: VectorEnv,
    agent: PPOAgent,
    hidden_states: List[Tuple[torch.Tensor, torch.Tensor] or None],
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
    actor_h0_buf = []
    actor_c0_buf = []
    episode_id_buf = []
    aborted_terminal_buf = []

    # 每个 env 的牌局编号（VectorEnv 在 done 时会自动 reset，因此这里用计数器区分“同一 rollout 内的不同局”）
    num_envs = len(vec_env.envs)
    episode_ids = np.zeros((num_envs,), dtype=np.int32)
    aborted_episodes = [set() for _ in range(num_envs)]

    for _ in range(rollout_steps):
        batch_actor_states: List[np.ndarray] = []
        batch_critic_states: List[np.ndarray] = []
        batch_legal_masks: List[np.ndarray] = []
        batch_global_feats: List[np.ndarray] = []
        batch_h0: List[np.ndarray] = []
        batch_c0: List[np.ndarray] = []

        # 记录本 step 对应的 episode_id（按 env）
        episode_id_buf.append(episode_ids.copy())

        # 为所有环境构建批量状态和特征
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

            # 记录本 step 输入给 Actor 的 hidden state（用于 RNN-PPO 训练重放）
            hidden = hidden_states[env_idx]
            if hidden is None:
                H = agent.actor.actor_lstm.hidden_size
                h0 = np.zeros((H,), dtype=np.float32)
                c0 = np.zeros((H,), dtype=np.float32)
            else:
                h0 = hidden[0].detach().cpu().numpy().reshape(-1).astype(np.float32)
                c0 = hidden[1].detach().cpu().numpy().reshape(-1).astype(np.float32)

            batch_actor_states.append(actor_state)
            batch_critic_states.append(critic_state)
            batch_legal_masks.append(legal_mask)
            batch_global_feats.append(global_feats)
            batch_h0.append(h0)
            batch_c0.append(c0)

        # 将当前步的所有环境状态打包成批量张量
        actor_states_step = np.stack(batch_actor_states, axis=0)   # [E,3,4,15]
        critic_states_step = np.stack(batch_critic_states, axis=0) # [E,6,4,15]
        legal_masks_step = np.stack(batch_legal_masks, axis=0)     # [E,A]
        global_feats_step = np.stack(batch_global_feats, axis=0)   # [E,5]

        # 使用 batched 接口一次性选择所有环境的动作
        actions_step, log_probs_step, values_step, new_hiddens = agent.select_action_batch(
            actor_states_step,
            critic_states_step,
            global_feats_step,
            legal_masks_step,
            tuple(hidden_states),
            compute_value=True,
        )

        # 更新 hidden_states
        for env_idx in range(num_envs):
            hidden_states[env_idx] = new_hiddens[env_idx]

        next_states, rewards, dones, infos = vec_env.step(list(actions_step.astype(int)))

        # 记录本 step 是否为废局终局（仅在 done=True 时有意义）
        aborted_terminal = np.array(
            [bool(infos[i].get("aborted", False)) if dones[i] else False for i in range(num_envs)],
            dtype=bool,
        )
        aborted_terminal_buf.append(aborted_terminal)

        # 将废局的整局标记下来（后续整段过滤），并在 done 时推进 episode_id
        for env_idx, d in enumerate(dones):
            if d:
                if aborted_terminal[env_idx]:
                    aborted_episodes[env_idx].add(int(episode_ids[env_idx]))
                episode_ids[env_idx] += 1
        
        # 终局后 env 已被 VectorEnv.reset()，这里同步清理我方 actor hidden，避免跨局泄漏
        for env_idx, d in enumerate(dones):
            if d:
                hidden_states[env_idx] = None

        actor_states_buf.append(actor_states_step)
        critic_states_buf.append(critic_states_step)
        actions_buf.append(actions_step.astype(np.int64))
        log_probs_buf.append(log_probs_step.astype(np.float32))
        rewards_buf.append(rewards.copy())
        dones_buf.append(dones.copy())
        values_buf.append(values_step.astype(np.float32))
        legal_masks_buf.append(legal_masks_step)
        global_feats_buf.append(global_feats_step)
        actor_h0_buf.append(np.stack(batch_h0, axis=0))
        actor_c0_buf.append(np.stack(batch_c0, axis=0))

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

        value_t = agent.critic(critic_state_t, global_feats_t)
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
    actor_h0 = np.stack(actor_h0_buf, axis=0)  # [T, E, H]
    actor_c0 = np.stack(actor_c0_buf, axis=0)  # [T, E, H]
    episode_id = np.stack(episode_id_buf, axis=0)  # [T, E]
    aborted_terminals = np.stack(aborted_terminal_buf, axis=0)  # [T, E]

    # 生成每步是否属于废局的 mask（整段牌局过滤）
    T, E = dones.shape
    aborted_steps = np.zeros((T, E), dtype=bool)
    for e in range(E):
        if len(aborted_episodes[e]) == 0:
            continue
        aborted_steps[:, e] = np.isin(
            episode_id[:, e], np.array(sorted(aborted_episodes[e]), dtype=np.int32)
        )
    valid_mask = ~aborted_steps

    returns, advantages = compute_gae(rewards, values, dones, last_values_arr, gamma, gae_lambda)

    # 统计完成的游戏局数（dones 为 True 的次数）
    num_games_completed = int(dones.sum())
    num_aborted_games = int(aborted_terminals.sum())
    
    # 计算平均 reward：只统计终局时的 reward（dones=True 时的 reward 的平均值）
    # 在稀疏奖励设置下，只有终局才有非零 reward
    terminal_rewards = rewards[dones & (~aborted_terminals)]
    if len(terminal_rewards) > 0:
        mean_reward = float(terminal_rewards.mean())
    else:
        # 如果没有完成的游戏，返回 0
        mean_reward = 0.0

    # 记录每个环境的对手策略类型（用于后续统计不同对手策略下的reward）
    opponent_strategy_types = vec_env.opponent_strategy_types if hasattr(vec_env, 'opponent_strategy_types') else None

    batch = {
        # 序列数据（RNN-PPO 训练使用）
        "actor_states": torch.from_numpy(actor_states).float(),          # [T,E,3,4,15]
        "critic_states": torch.from_numpy(critic_states).float(),        # [T,E,6,4,15]
        "actions": torch.from_numpy(actions).long(),                     # [T,E]
        "log_probs": torch.from_numpy(log_probs).float(),                # [T,E]
        "returns": torch.from_numpy(returns).float(),                    # [T,E]
        "advantages": torch.from_numpy(advantages).float(),              # [T,E]
        "legal_masks": torch.from_numpy(legal_masks.astype(np.bool_)),   # [T,E,A]
        "global_feats": torch.from_numpy(global_feats).float(),          # [T,E,5]
        "dones_t": torch.from_numpy(dones.astype(np.bool_)),             # [T,E]
        "actor_h0": torch.from_numpy(actor_h0).float(),                  # [T,E,H]
        "actor_c0": torch.from_numpy(actor_c0).float(),                  # [T,E,H]
        "valid_mask": torch.from_numpy(valid_mask.astype(np.bool_)),     # [T,E]
        "aborted_terminals": aborted_terminals,                          # [T,E] numpy bool
        "mean_reward": mean_reward,
        "num_games_completed": num_games_completed,
        "num_aborted_games": num_aborted_games,
        "aborted_game_ratio": float(num_aborted_games / num_games_completed) if num_games_completed > 0 else 0.0,
        "aborted_steps_filtered": int(aborted_steps.sum()),
        "rewards": rewards,  # 保留原始rewards用于按对手策略类型统计
        "dones": dones,  # 保留原始dones用于按对手策略类型统计
        "opponent_strategy_types": opponent_strategy_types,  # 每个环境的对手策略类型
    }

    return batch

