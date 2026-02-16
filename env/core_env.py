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
)
from .encoding import build_state


class GanDengYanEnv:
    """
    干瞪眼环境（单智能体视角）。
    - 手牌、墓地等使用数量统计，不区分花色。
    - step: 从“我出牌”到下一次轮到“我出牌”或本局结束。
    """

    def __init__(self, num_players: int = 4, seed: Optional[int] = None, opponent_agent=None, opponent_hidden_states=None):
        assert num_players == 4, "当前实现仅支持 4 人局"
        self.num_players = num_players
        self.rng = random.Random(seed)
        
        # 对手策略：None 表示贪心，否则使用 PPO agent
        self.opponent_agent = opponent_agent
        self.opponent_hidden_states = opponent_hidden_states  # List[hidden_state or None] for each opponent

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

        # 每局随机庄家
        self.dealer = self.rng.randrange(self.num_players)

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
        max_iterations = self.num_players * 2  # 最多模拟两圈
        iterations = 0
        while self.current_player != self.agent_id and not self.aborted and iterations < max_iterations:
            self._opponent_move(self.current_player)
            iterations += 1
            # 检查是否有人出完牌（终局）
            if self._remaining_cards(self.current_player) == 0:
                # 终局，但 reset() 不应该返回终局状态，所以这里应该不会发生
                # 如果发生了，说明牌局异常，标记为废局
                self.aborted = True
                break
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
        """给玩家摸一张牌。若牌堆已空，返回 False。"""
        if self.deck_pos >= len(self.deck):
            return False
        rank = self.deck[self.deck_pos]
        self.deck_pos += 1
        self.hands[pid, rank] += 1
        return True

    def _hand_counts(self, pid: int) -> np.ndarray:
        return self.hands[pid]

    def _remaining_cards(self, pid: int) -> int:
        return int(self.hands[pid].sum())

    # ===== 合法动作计算 =====

    def get_legal_actions(
        self,
        hand: np.ndarray,
        last_move_pattern: Optional[ActionPattern],
        last_move_used: Optional[np.ndarray],
        must_play: bool = False,
    ) -> np.ndarray:
        """返回一个 bool mask，长度为 self.num_actions。"""
        mask = np.zeros(self.num_actions, dtype=bool)

        for idx, pat in enumerate(self.actions):
            if pat.type == "pass":
                # pass 的合法性单独判断
                if not must_play:
                    # 只有当前有 last_move 时 pass 才有意义；没牌型时首出不能 pass
                    if last_move_pattern is not None:
                        mask[idx] = True
                continue

            can_play, used, _ = can_pay_pattern(hand, pat)
            if not can_play or used is None:
                continue

            if last_move_pattern is None:
                # 首出：任意可出的非 pass 牌型都合法
                mask[idx] = True
            else:
                if beats(last_move_pattern, last_move_used, pat, used):
                    mask[idx] = True

        return mask

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
                break

            self.current_player = (self.current_player + 1) % self.num_players

        next_state = self._get_obs()
        return next_state, reward, done, info

    def _apply_action(self, pid: int, pat: ActionPattern):
        """对某个玩家应用给定动作（不含轮转逻辑）。"""
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
        """
        if self.opponent_agent is not None:
            # 使用 PPO policy
            # 需要从对手视角构建状态（临时切换 agent_id）
            original_agent_id = self.agent_id
            self.agent_id = pid
            state = self._get_obs()
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
            opp_idx = pid - 1  # 对手索引（pid 1,2,3 对应索引 0,1,2）
            hidden = None
            if self.opponent_hidden_states is not None and opp_idx < len(self.opponent_hidden_states):
                hidden = self.opponent_hidden_states[opp_idx]
            
            action_idx, _, _, new_hidden = self.opponent_agent.select_action(
                state, global_feats, legal_mask, hidden
            )
            
            # 更新 hidden state
            if self.opponent_hidden_states is not None and opp_idx < len(self.opponent_hidden_states):
                self.opponent_hidden_states[opp_idx] = new_hidden
            
            pat = self.actions[action_idx]
            self._apply_action(pid, pat)
        else:
            # 贪心策略（原有逻辑）
            hand = self._hand_counts(pid)
            legal = self.get_legal_actions(
                hand, self.last_move_pattern, self.last_move_cards, must_play=self.must_play
            )
            # 策略：找到第一个合法的非 pass 动作；若没有，则 pass
            action_idx = 0  # 默认 pass
            for idx, pat in enumerate(self.actions):
                if not legal[idx]:
                    continue
                if pat.type != "pass":
                    action_idx = idx
                    break
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

