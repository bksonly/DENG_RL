"""
干瞪眼环境核心实现：发牌、出牌规则、奖励与 step 流程。
"""

import random
from typing import List, Dict, Tuple, Optional, Any

import numpy as np

from .actions import (
    NUM_RANKS,
    RANK_3,
    RANK_2,
    RANK_JOKER_SMALL,
    RANK_JOKER_BIG,
    ActionPattern,
    build_action_space,
    can_pay_pattern,
    beats,
    enumerate_legal_first_moves,
    enumerate_legal_responses,
)
from .encoding import build_state


class GanDengYanEnv:
    """
    干瞪眼环境（单智能体视角）。
    - 手牌、墓地等使用数量统计，不区分花色。
    - step: 从“我出牌”到下一次轮到“我出牌”或本局结束。
    """

    def __init__(
        self,
        num_players: int = 4,
        seed: Optional[int] = None,
        opponent_agent=None,
        opponent_hidden_states=None,
        opponent_strategy_type: str = "greedy",
    ):
        assert num_players == 4, "当前实现仅支持 4 人局"
        self.num_players = num_players
        self.rng = random.Random(seed)
        
        # 对手策略：
        # - opponent_agent 为 None 时，根据 opponent_strategy_type 选择贪心/idiot 等固定策略；
        # - opponent_agent 不为 None 时，使用 PPO policy。
        self.opponent_agent = opponent_agent
        self.opponent_hidden_states = opponent_hidden_states  # List[hidden_state or None] for each opponent
        self.opponent_strategy_type = opponent_strategy_type

        # 下一局的庄家（用于实现赢家连庄规则）
        self.next_dealer: Optional[int] = None

        # 预生成整副牌（只看点数）
        self.full_deck: List[int] = self._build_full_deck()

        # 静态动作空间
        self.actions: List[ActionPattern] = build_action_space()
        self.num_actions = len(self.actions)

        # 运行时状态
        self.reset()

    # ===== 环境基础接口 =====

    def reset(self) -> np.ndarray:
        """洗牌、发牌并初始化整局状态，返回初始观测。"""
        self.deck = list(self.full_deck)
        self.rng.shuffle(self.deck)
        self.deck_pos = 0

        # 确定庄家：如果上一局有赢家，则连庄；否则随机选择
        if self.next_dealer is not None:
            self.dealer = self.next_dealer
        else:
            self.dealer = self.rng.randrange(self.num_players)
        
        # 重置 next_dealer，等待本局结束后的更新
        self.next_dealer = None

        # 手牌计数：players x NUM_RANKS
        self.hands = np.zeros((self.num_players, NUM_RANKS), dtype=np.int32)

        # 发牌：其他 3 家 5 张，庄家 6 张
        for pid in range(self.num_players):
            num_cards = 6 if pid == self.dealer else 5
            for _ in range(num_cards):
                self._draw_card(pid)

        # 墓地（全局已出牌）
        self.graveyard = np.zeros(NUM_RANKS, dtype=np.int32)

        # 当前桌面牌型与拥有者
        self.last_move_pattern: Optional[ActionPattern] = None
        self.last_move_cards: Optional[np.ndarray] = None  # NUM_RANKS 向量
        self.last_move_player: Optional[int] = None

        # 当前玩家（从庄家开始）
        self.current_player = self.dealer

        # 统计每个玩家是否有“实际出牌”
        self.played_any = [False for _ in range(self.num_players)]
        # 专门标记“庄家是否除了开局首手外还出过牌”
        self.dealer_extra_play = False
        # 本局中炸弹被打出的总次数（用于全场翻倍）
        self.bomb_times = 0

        # pass 计数：用于判断是否一圈都没人接
        self.passes_since_last_move = 0

        # 我方视角：固定玩家 0 作为训练的 Agent
        self.agent_id = 0

        # 标记“当前玩家是否处于摸牌后首出（必须出牌）”
        self.must_play = False

        # 牌局是否已经废局
        self.aborted = False

        # 如果当前不是 agent 的回合，需要模拟对手直到轮到 agent
        # 这样 reset() 后环境就处于"轮到 agent 出牌"的状态
        max_iterations = self.num_players  # 最多模拟一圈就会轮到agent
        iterations = 0
        while self.current_player != self.agent_id and not self.aborted and iterations < max_iterations:
            self._opponent_move(self.current_player)
            iterations += 1
            # 检查是否有人出完牌（终局）
            if self._remaining_cards(self.current_player) == 0:
                # 如果对手在轮到agent之前就出完牌了，记录赢家并重新reset一遍
                # 这种情况虽然少见，但在极端牌型下是可能发生的
                winner = self.current_player
                if not self.aborted:
                    # 如果不是废局，记录赢家作为下一局的庄家
                    self.next_dealer = winner
                else:
                    # 如果是废局，不连庄
                    self.next_dealer = None
                return self.reset()
            self.current_player = (self.current_player + 1) % self.num_players
        
        # 如果经过最大迭代次数仍未轮到 agent，标记为废局
        if self.current_player != self.agent_id:
            self.aborted = True

        state = self._get_obs()
        return state

    # ===== 内部基础操作 =====

    def _build_full_deck(self) -> List[int]:
        """
        构造一副完整的牌堆，只包含点数信息，不区分花色。
        """
        deck: List[int] = [r for r in range(RANK_3, RANK_2 + 1) for _ in range(4)]
        deck.extend([RANK_JOKER_SMALL, RANK_JOKER_BIG])
        return deck

    def _draw_card(self, pid: int) -> bool:
        """
        给玩家摸一张牌。若牌堆已空，返回 False。
        
        参数:
            pid: player_id，玩家ID（0, 1, 2, 3），其中0是训练的agent
        返回:
            bool: 是否成功摸牌
        """
        if self.deck_pos >= len(self.deck):
            return False
        rank = self.deck[self.deck_pos]
        self.deck_pos += 1
        self.hands[pid, rank] += 1
        return True

    def _hand_counts(self, pid: int) -> np.ndarray:
        """
        获取指定玩家的手牌计数。
        
        参数:
            pid: player_id，玩家ID（0, 1, 2, 3）
        返回:
            np.ndarray: 长度为NUM_RANKS的手牌计数向量
        """
        return self.hands[pid]

    def _remaining_cards(self, pid: int) -> int:
        """
        获取指定玩家的剩余手牌数。
        
        参数:
            pid: player_id，玩家ID（0, 1, 2, 3）
        返回:
            int: 剩余手牌数
        """
        return int(self.hands[pid].sum())

    # ===== 合法动作计算 =====

    def _find_action_index(self, pattern: ActionPattern) -> Optional[int]:
        """在预定义动作空间中查找指定动作的索引。"""
        for idx, pat in enumerate(self.actions):
            if (pat.type == pattern.type and
                pat.rank == pattern.rank and
                pat.length == pattern.length and
                pat.start_rank == pattern.start_rank):
                return idx
        return None

    def get_legal_actions(
        self,
        hand: np.ndarray,
        last_move_pattern: Optional[ActionPattern],
        last_move_used: Optional[np.ndarray],
        must_play: bool = False,
    ) -> np.ndarray:
        """
        返回一个 bool mask，长度为 self.num_actions。
        优化版本：基于手牌和接牌规则高效枚举合法动作，然后在总表中查找索引。
        """
        mask = np.zeros(self.num_actions, dtype=bool)
        
        # 处理 pass 动作
        if not must_play and last_move_pattern is not None:
            pass_idx = self._find_action_index(ActionPattern(type="pass"))
            if pass_idx is not None:
                mask[pass_idx] = True

        # 枚举合法动作
        if last_move_pattern is not None:
            # 接牌：只枚举能管住的情况
            legal_patterns = enumerate_legal_responses(hand, last_move_pattern, last_move_used)
        else:
            # 首出：基于手牌枚举所有可能的动作
            legal_patterns = enumerate_legal_first_moves(hand)
        
        # 将枚举出的 ActionPattern 映射到总表中的索引
        for pat in legal_patterns:
            idx = self._find_action_index(pat)
            if idx is not None:
                mask[idx] = True
        
        return mask

    # ===== 贪心策略（供对手与调试脚本复用） =====

    def _select_greedy_action_index(self, pid: int) -> int:
        """
        为指定玩家 pid 选择一个“菜鸟式”贪心动作：
        - 接牌时：优先打小牌，尽量不用炸弹（只有没有其它选择时才考虑炸弹）；
          例如上家打4，我更倾向于打5，而不是直接掏2或炸弹。
        - 首出时：在所有合法动作中，忽略炸弹，取“最后一个”非炸弹动作，尽量不先出2或2对。
         （结合当前枚举顺序，相当于优先出最长的顺子，其次对子，最后单牌）。
        - 若没有任何可出的非炸弹动作，则在炸弹中选择“代价最小”的一个；
          若连炸弹也没有，则只能 PASS（如果规则允许）。
        """
        hand = self._hand_counts(pid)
        legal_mask = self.get_legal_actions(
            hand, self.last_move_pattern, self.last_move_cards, must_play=self.must_play
        )

        # 找到 pass 的索引（若存在）
        pass_idx = self._find_action_index(ActionPattern(type="pass"))
        can_pass = (
            pass_idx is not None
            and pass_idx < len(legal_mask)
            and bool(legal_mask[pass_idx])
        )

        # 根据是否有上家动作，枚举候选牌型
        if self.last_move_pattern is not None:
            # 接牌：只枚举能管住的动作
            legal_patterns = enumerate_legal_responses(
                hand, self.last_move_pattern, self.last_move_cards
            )
        else:
            # 首出：基于手牌枚举所有可能动作
            legal_patterns = enumerate_legal_first_moves(hand)

        candidates = []
        for pat in legal_patterns:
            idx = self._find_action_index(pat)
            if idx is None:
                continue
            if idx >= len(legal_mask) or not legal_mask[idx]:
                continue
            candidates.append((pat, idx))

        # 如果没有任何可出的动作（理论上不太可能），则 PASS 或 0 号动作兜底
        if not candidates:
            if can_pass:
                return pass_idx  # type: ignore[arg-type]
            return 0

        # ===== 情况一：接牌，优先打小牌，尽量不用炸弹 =====
        if self.last_move_pattern is not None:
            non_bombs = [(pat, idx) for pat, idx in candidates if pat.type != "bomb"]

            # 有非炸弹解时，只在这些解里选“最小”的一手
            if non_bombs:
                def key_non_bomb(item):
                    pat, _ = item
                    if pat.type in ("single", "pair", "bomb"):
                        rank = pat.rank if pat.rank is not None else 0
                        length = pat.length if pat.length is not None else 0
                        # 单张 / 对子：以点数为主，长度次之
                        return (rank, length)
                    elif pat.type == "straight":
                        start = pat.start_rank if pat.start_rank is not None else 0
                        length = pat.length if pat.length is not None else 0
                        # 顺子：先比较起始点数，再比较长度
                        return (start, length)
                    # 其它类型暂不使用
                    return (0, 0)

                _, best_idx = min(non_bombs, key=key_non_bomb)
                return best_idx

            # 只有炸弹可用时，在炸弹中选择“最轻微”的一个
            bombs = [(pat, idx) for pat, idx in candidates if pat.type == "bomb"]
            if bombs:
                def key_bomb(item):
                    pat, _ = item
                    length = pat.length if pat.length is not None else 0
                    rank = pat.rank if pat.rank is not None else 0
                    # 先比较长度，再比较点数：越短、点数越小越优先
                    return (length, rank)

                _, best_idx = min(bombs, key=key_bomb)
                return best_idx

            # 理论上不会走到这里（candidates 非空且不是炸弹非炸弹的组合），兜底 PASS
            if can_pass:
                return pass_idx  # type: ignore[arg-type]
            return 0

        # ===== 情况二：首出，在合法非炸弹动作中取“最后一个” =====
        non_bombs = [(pat, idx) for pat, idx in candidates if pat.type != "bomb"]
        if non_bombs:
            # 先尝试在非炸弹动作中排除“2”和“对2”，这样不会无脑先掏2或2对。
            filtered = [
                (pat, idx)
                for pat, idx in non_bombs
                if not (pat.type in ("single", "pair") and pat.rank == RANK_2)
            ]

            if filtered:
                # 若排除 2/对2 后仍有候选，按当前枚举顺序取最后一个：
                #   - 顺子中最长的组合
                #   - 若没有顺子，则是较大的对子
                #   - 最后才是较大的单张
                _, best_idx = filtered[-1]
            else:
                # 如果把 2 和对2 排掉之后一个都不剩，就退回到原非炸弹列表，
                # 仍然取最后一个（哪怕是 2 或对2），避免无牌可出。
                _, best_idx = non_bombs[-1]
            return best_idx

        # 只有炸弹：选一个“最轻微”的炸弹（长度短、点数小）
        bombs = [(pat, idx) for pat, idx in candidates if pat.type == "bomb"]
        if bombs:
            def key_bomb_first(item):
                pat, _ = item
                length = pat.length if pat.length is not None else 0
                rank = pat.rank if pat.rank is not None else 0
                return (length, rank)

            _, best_idx = min(bombs, key=key_bomb_first)
            return best_idx

        # 理论兜底：没有候选但规则允许 PASS
        if can_pass:
            return pass_idx  # type: ignore[arg-type]
        return 0

    def _select_idiot_action_index(self, pid: int) -> int:
        """
        为指定玩家 pid 选择动作（\"idiot\" 白痴策略）：
        - 行为等同于旧版简单贪心：
          - 在合法动作中按枚举顺序找到第一个非 PASS 动作；
          - 若没有非 PASS 动作且允许 PASS，则 PASS；
          - 否则退化为动作 0。
        - 这样一般会优先出最小的单牌，其次最小的对子/顺子。
        """
        hand = self._hand_counts(pid)
        legal_mask = self.get_legal_actions(
            hand, self.last_move_pattern, self.last_move_cards, must_play=self.must_play
        )

        # PASS 的索引
        pass_idx = self._find_action_index(ActionPattern(type="pass"))
        can_pass = (
            pass_idx is not None
            and pass_idx < len(legal_mask)
            and bool(legal_mask[pass_idx])
        )

        # 按之前贪心逻辑的方式枚举候选牌型
        if self.last_move_pattern is not None:
            legal_patterns = enumerate_legal_responses(
                hand, self.last_move_pattern, self.last_move_cards
            )
        else:
            legal_patterns = enumerate_legal_first_moves(hand)

        # 找到第一个合法的非 PASS 动作
        for pat in legal_patterns:
            if pat.type == "pass":
                continue
            idx = self._find_action_index(pat)
            if idx is None:
                continue
            if idx >= len(legal_mask) or not legal_mask[idx]:
                continue
            return idx

        # 没有非 PASS，尝试 PASS
        if can_pass:
            return pass_idx  # type: ignore[arg-type]

        # 兜底：直接返回0号动作
        return 0

    # ===== 观测编码 =====

    def _get_obs(self) -> np.ndarray:
        """构造 [C, 4, 15] 状态张量。"""
        has_active_last_move = self.last_move_pattern is not None
        return build_state(
            agent_id=self.agent_id,
            hands=self.hands,
            num_players=self.num_players,
            graveyard=self.graveyard,
            last_move_cards=self.last_move_cards,
            last_move_player=self.last_move_player,
            current_player=self.current_player,
            must_play=self.must_play,
            has_active_last_move=has_active_last_move,
        )

    # ===== step 流程 =====

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        从 Agent 的视角执行一步：
        - 假定当前轮到 agent_id 出牌。
        - 执行给定动作，然后让环境内部模拟其他玩家，直到再次轮到 agent 或本局结束。
        """
        if self.aborted:
            return self._get_obs(), 0.0, True, {"aborted": True}

        assert self.current_player == self.agent_id, "step 只能在轮到 agent 时调用"

        hand = self._hand_counts(self.agent_id)
        legal_mask = self.get_legal_actions(
            hand, self.last_move_pattern, self.last_move_cards, must_play=self.must_play
        )
        if not legal_mask[action_idx]:
            # 非法动作：简单地当成 pass 处理，以免训练崩溃
            action_idx = 0  # 0 号约定为 pass

        pat = self.actions[action_idx]
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        # 执行 agent 动作
        self._apply_action(self.agent_id, pat)

        # 终局检测
        if self._remaining_cards(self.agent_id) == 0:
            reward = self._compute_final_reward()
            done = True
            info["winner"] = self.agent_id
            info["aborted"] = self.aborted
            # 记录赢家：如果不是废局，下一局由赢家连庄
            if not self.aborted:
                self.next_dealer = self.agent_id
            else:
                self.next_dealer = None
            return self._get_obs(), reward, done, info

        # 轮到下一个玩家
        self.current_player = (self.current_player + 1) % self.num_players

        # 环境内部模拟对手直到再次轮到 agent 或终局/废局
        while True:
            if self.aborted:
                done = True
                info["aborted"] = True
                break

            if self.current_player == self.agent_id:
                break

            self._opponent_move(self.current_player)

            # 检查是否有人出完牌
            if self._remaining_cards(self.current_player) == 0:
                winner = self.current_player
                reward = self._compute_final_reward()
                done = True
                info["winner"] = winner
                info["aborted"] = self.aborted
                # 记录赢家：如果不是废局，下一局由赢家连庄
                if not self.aborted:
                    self.next_dealer = winner
                else:
                    self.next_dealer = None
                break

            self.current_player = (self.current_player + 1) % self.num_players

        next_state = self._get_obs()
        return next_state, reward, done, info

    def _apply_action(self, pid: int, pat: ActionPattern):
        """
        对某个玩家应用给定动作（不含轮转逻辑）。
        
        参数:
            pid: player_id，玩家ID（0, 1, 2, 3）
            pat: 要执行的动作模式
        """
        hand = self._hand_counts(pid)

        if pat.type == "pass":
            self.passes_since_last_move += 1
            # 检查是否一圈都没人接
            if (
                self.last_move_pattern is not None
                and self.passes_since_last_move >= self.num_players - 1
            ):
                # 这一轮结束，last_move_player 获得新一轮首出权
                winner = self.last_move_player
                self.last_move_pattern = None
                self.last_move_cards = None
                self.last_move_player = None
                self.passes_since_last_move = 0

                if winner is not None:
                    # 赢家（无论是否为 agent）从牌堆摸一张牌（若有）
                    if not self._draw_card(winner):
                        # 牌堆耗尽，直接废局
                        self.aborted = True
                    # 新一轮由赢家先出
                    self.current_player = winner

                # must_play 对所有玩家生效：
                # 只要有玩家压住一圈并摸牌，则其首手属于“摸牌后首出”，不能过牌。
                self.must_play = (winner is not None and not self.aborted)
            return

        # 非 pass：真正出牌
        # 若当前动作为炸弹，且桌面上已有牌，则视为“炸弹炸场一次”，用于全场翻倍计数
        if pat.type == "bomb" and self.last_move_pattern is not None:
            self.bomb_times += 1

        can_play, used, _ = can_pay_pattern(hand, pat)
        if not can_play or used is None:
            # 理论上不应出现，防御性编程：当成 pass
            self._apply_action(pid, ActionPattern(type="pass"))
            return

        # 从手牌中扣除
        self.hands[pid] -= used
        # 墓地增加
        self.graveyard += used

        # 更新 last_move
        self.last_move_pattern = pat
        self.last_move_cards = used
        self.last_move_player = pid
        self.passes_since_last_move = 0

        # 记录“是否有实际出牌”
        if pid == self.dealer:
            # 庄家：首手不计入 dealer_extra_play，后续才算
            if self.played_any[pid]:
                self.dealer_extra_play = True
            else:
                # 首次出牌（可能是开局那一手）
                self.played_any[pid] = True
        else:
            self.played_any[pid] = True

        # 一旦有人真正出牌，就不是“摸牌后首出”的限制状态了
        self.must_play = False

    def _opponent_move(self, pid: int):
        """
        对手策略：如果设置了 opponent_agent 则使用 PPO policy，否则使用贪心策略。
        
        参数:
            pid: player_id，玩家ID（1, 2, 3 中的一个，因为0是agent）
        """
        if self.opponent_agent is not None:
            # 使用 PPO policy
            # 需要从对手视角构建状态（临时切换 agent_id）
            original_agent_id = self.agent_id
            self.agent_id = pid
            
            # 构建actor状态（3通道：自己的手牌、墓地、目标牌）
            from .encoding import build_state
            actor_state = build_state(
                agent_id=pid,
                hands=self.hands,
                num_players=self.num_players,
                graveyard=self.graveyard,
                last_move_cards=self.last_move_cards,
                last_move_player=self.last_move_player,
                current_player=self.current_player,
                must_play=self.must_play,
                has_active_last_move=self.last_move_pattern is not None,
            )
            
            # 对手只需要actor来选择动作，不需要critic（value不会被使用）
            # 传入一个空的critic_state（不会被使用，但保持接口兼容）
            critic_state = np.zeros((6, 4, 15), dtype=np.float32)
            
            self.agent_id = original_agent_id  # 恢复
            
            hand = self._hand_counts(pid)
            legal_mask = self.get_legal_actions(
                hand, self.last_move_pattern, self.last_move_cards, must_play=self.must_play
            )
            
            # 构建全局特征（从对手视角）
            from .encoding import build_global_features
            has_active_last_move = self.last_move_pattern is not None
            global_feats = build_global_features(
                agent_id=pid,
                hands=self.hands,
                num_players=self.num_players,
                current_player=self.current_player,
                must_play=self.must_play,
                has_active_last_move=has_active_last_move,
            )
            
            # 获取对手的 hidden state（如果有）
            # 注意：hidden state 格式现在是 (h_actor, c_actor) 或 None（当只调用 actor 时）
            opp_idx = pid - 1  # 对手索引（pid 1,2,3 对应索引 0,1,2）
            hidden = None
            if self.opponent_hidden_states is not None and opp_idx < len(self.opponent_hidden_states):
                old_hidden = self.opponent_hidden_states[opp_idx]
                # 如果 hidden state 是旧格式（4个tensor），只提取 actor 部分
                if old_hidden is not None and len(old_hidden) == 4:
                    h_actor, c_actor, _, _ = old_hidden
                    hidden = (h_actor, c_actor)
                else:
                    hidden = old_hidden
            
            # 只调用 actor，不计算 value
            action_idx, _, _, new_hidden = self.opponent_agent.select_action(
                actor_state, critic_state, global_feats, legal_mask, hidden, compute_value=False
            )
            
            # 更新 hidden state（只包含 actor 的 hidden state）
            if self.opponent_hidden_states is not None and opp_idx < len(self.opponent_hidden_states):
                self.opponent_hidden_states[opp_idx] = new_hidden
            
            pat = self.actions[action_idx]
            self._apply_action(pid, pat)
        else:
            # 固定策略：根据 opponent_strategy_type 选择具体规则
            strategy = getattr(self, "opponent_strategy_type", "greedy")
            if strategy == "idiot":
                action_idx = self._select_idiot_action_index(pid)
            else:
                # 默认使用改进后的“菜鸟式”贪心
                action_idx = self._select_greedy_action_index(pid)
            pat = self.actions[action_idx]
            self._apply_action(pid, pat)

    # ===== 终局奖励计算 =====

    def _compute_final_reward(self) -> float:
        """
        按大纲中的结算逻辑计算我方的最终 reward。
        - 每个输家单独计算 loss_i 并封顶 20；
        - 通关（整局未出牌；庄家若只出过首手也算通关） loss_i * 2；
        - 若我是赢家：reward = sum_i loss_i；
        - 若我是输家：reward = -loss_me。
        """
        # 废局不参与训练
        if self.aborted:
            return 0.0

        # 找赢家
        remaining = [self._remaining_cards(pid) for pid in range(self.num_players)]
        if 0 not in remaining:
            # 理论上不应发生
            return 0.0
        winner = remaining.index(0)

        losses = [0.0 for _ in range(self.num_players)]

        for pid in range(self.num_players):
            if pid == winner:
                continue
            loss = float(remaining[pid])

            # 通关判定
            if pid == self.dealer:
                # 庄家若只出过开局首手，则 dealer_extra_play 为 False
                is_tongguan = not self.dealer_extra_play
            else:
                is_tongguan = not self.played_any[pid]

            if is_tongguan:
                loss *= 2.0

            # 终局时剩余手牌中若仍包含任意大小王（小王或大王），在上述基础上再额外翻倍一次
            jokers_left = int(self.hands[pid, RANK_JOKER_SMALL] + self.hands[pid, RANK_JOKER_BIG])
            if jokers_left > 0 and loss > 0.0:
                loss *= 2.0

            # 本局中炸弹被打出的总次数，按 2^k 对全场输家再翻倍（k 为炸弹次数）
            k = int(getattr(self, "bomb_times", 0))
            if k > 0 and loss > 0.0:
                loss *= (2.0 ** k)

            loss = min(loss, 20.0)
            losses[pid] = loss

        if self.agent_id == winner:
            return float(sum(losses))
        else:
            return -losses[self.agent_id]


if __name__ == "__main__":
    # 简单自检：跑几步看环境是否能正常执行
    env = GanDengYanEnv()
    state = env.reset()
    done = False
    while not done:
        hand = env._hand_counts(env.agent_id)
        legal = env.get_legal_actions(
            hand, env.last_move_pattern, env.last_move_cards, must_play=env.must_play
        )
        legal_idxs = np.nonzero(legal)[0]
        if len(legal_idxs) == 0:
            a = 0
        else:
            a = int(random.choice(list(legal_idxs)))
        state, reward, done, info = env.step(a)
        if done:
            break

