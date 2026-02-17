"""
状态编码相关工具。

拆分为两部分：
- 空间状态：将牌数量等信息编码为 [C_s, 4, 15] 的张量，只包含适合卷积的牌面矩阵；
- 全局特征：与牌面布局无关的一些标量（例如各家剩余手牌数、是否为摸牌后首出）。
"""

from typing import Optional

import numpy as np

from .actions import NUM_RANKS


def encode_counts_to_matrix(counts: np.ndarray) -> np.ndarray:
    """
    将长度为 NUM_RANKS 的数量向量编码为 [4, 15]：
      row0: >=1, row1: >=2, row2: >=3, row3: >=4
    """
    mat = np.zeros((4, NUM_RANKS), dtype=np.float32)
    for r in range(NUM_RANKS):
        c = int(counts[r])
        if c >= 1:
            mat[0, r] = 1.0
        if c >= 2:
            mat[1, r] = 1.0
        if c >= 3:
            mat[2, r] = 1.0
        if c >= 4:
            mat[3, r] = 1.0
    return mat


def build_state(
    agent_id: int,
    hands: np.ndarray,
    num_players: int,
    graveyard: np.ndarray,
    last_move_cards: Optional[np.ndarray],
    last_move_player: Optional[int],
    current_player: int,
    must_play: bool,
    has_active_last_move: bool,
) -> np.ndarray:
    """
    根据环境中的公开信息构造 [C_s, 4, 15] 空间状态张量。

    仅包含与牌面布局直接相关、适合卷积的矩阵信息：
      0: 我的手牌（数量编码矩阵）
      1: 墓地（全局已出牌，数量编码矩阵）
      2: 当前必须管的牌（上一轮/上一家的牌型，数量编码矩阵）
    其他全局标量信息（各家剩余手牌数、是否为摸牌后首出等）单独通过
    build_global_features 进行编码。
    """
    C_s = 3
    state = np.zeros((C_s, 4, NUM_RANKS), dtype=np.float32)

    # Channel 0: 我的手牌
    my_hand = hands[agent_id]
    state[0] = encode_counts_to_matrix(my_hand)

    # Channel 1: 墓地（全局已出牌）
    state[1] = encode_counts_to_matrix(graveyard)

    # Channel 2: 当前必须管的牌（target）
    if last_move_cards is not None and current_player != last_move_player:
        state[2] = encode_counts_to_matrix(last_move_cards)
    else:
        state[2] = 0.0

    return state


def build_state_for_critic(
    agent_id: int,
    hands: np.ndarray,
    num_players: int,
    graveyard: np.ndarray,
    last_move_cards: Optional[np.ndarray],
    last_move_player: Optional[int],
    current_player: int,
    must_play: bool,
    has_active_last_move: bool,
) -> np.ndarray:
    """
    为Critic构建包含所有玩家手牌的状态 [6, 4, 15]。
    
    通道顺序：
      0: 我的手牌
      1-3: 三个对手的手牌（按相对位置：左家、对家、右家）
      4: 墓地（全局已出牌）
      5: 当前必须管的牌（target）
    """
    C_s = 6
    state = np.zeros((C_s, 4, NUM_RANKS), dtype=np.float32)
    
    # Channel 0: 我的手牌
    my_hand = hands[agent_id]
    state[0] = encode_counts_to_matrix(my_hand)
    
    # Channel 1-3: 三个对手的手牌（按相对位置：左家、对家、右家）
    # 相对位置映射：1 -> 左家, 2 -> 对家, 3 -> 右家
    rel_pos_to_channel = {1: 1, 2: 2, 3: 3}
    for pid in range(num_players):
        if pid == agent_id:
            continue
        rel = (pid - agent_id) % num_players
        channel = rel_pos_to_channel.get(rel, None)
        if channel is not None:
            opp_hand = hands[pid]
            state[channel] = encode_counts_to_matrix(opp_hand)
    
    # Channel 4: 墓地（全局已出牌）
    state[4] = encode_counts_to_matrix(graveyard)
    
    # Channel 5: 当前必须管的牌（target）
    if last_move_cards is not None and current_player != last_move_player:
        state[5] = encode_counts_to_matrix(last_move_cards)
    else:
        state[5] = 0.0
    
    return state


def build_global_features(
    agent_id: int,
    hands: np.ndarray,
    num_players: int,
    current_player: int,
    must_play: bool,
    has_active_last_move: bool,
) -> np.ndarray:
    """
    构造长度为 5 的全局标量特征向量：
      0: 我自己剩余手牌数（归一化到 [0, 1]）
      1: 左家剩余手牌数（归一化）
      2: 对家剩余手牌数（归一化）
      3: 右家剩余手牌数（归一化）
      4: is_first_move_after_draw：是否为“摸牌后首出”（0/1）

    这里假设总玩家数固定为 4，且 agent_id=0，
    因此“左家/对家/右家”的顺序可以通过 (pid - agent_id) % num_players 推导。
    """
    feats = np.zeros(5, dtype=np.float32)

    # 0: 自己剩余手牌数
    my_remain = float(hands[agent_id].sum())
    feats[0] = min(my_remain / 20.0, 1.0)

    # 1-3: 三个对手的剩余手牌数（按相对位置：左家、对家、右家）
    # 相对位置映射表：1 -> 左家, 2 -> 对家, 3 -> 右家
    rel_pos_to_idx = {1: 1, 2: 2, 3: 3}
    for pid in range(num_players):
        if pid == agent_id:
            continue
        rel = (pid - agent_id) % num_players
        idx = rel_pos_to_idx.get(rel, None)
        if idx is None:
            continue
        remain = float(hands[pid].sum())
        feats[idx] = min(remain / 20.0, 1.0)

    # 4: 是否为“摸牌后首出”
    if must_play and current_player == agent_id and not has_active_last_move:
        feats[4] = 1.0
    else:
        feats[4] = 0.0

    return feats

