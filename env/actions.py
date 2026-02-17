"""
动作空间与牌型规则相关的工具。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# 牌点数编码：0-12: 3-4-...-K-A-2, 13: 小王, 14: 大王
NUM_RANKS = 15
RANK_3 = 0
RANK_2 = 12
RANK_JOKER_SMALL = 13
RANK_JOKER_BIG = 14


@dataclass
class ActionPattern:
    """
    静态动作描述：不区分具体花色，只描述牌型结构。
    """

    type: str  # 'pass' | 'single' | 'pair' | 'straight' | 'bomb'
    # 对于 single/pair/bomb：使用主点数和长度
    rank: Optional[int] = None
    length: Optional[int] = None
    # 对于 straight：起始点数和长度
    start_rank: Optional[int] = None


def build_action_space() -> List[ActionPattern]:
    """
    生成静态动作空间：pass、单张、对子、顺子、炸弹。
    """
    actions: List[ActionPattern] = []

    # pass 动作
    actions.append(ActionPattern(type="pass"))

    # 单张（不能单出大小王）
    for r in range(RANK_3, RANK_2 + 1):
        actions.append(ActionPattern(type="single", rank=r, length=1))

    # 对子（主点数 3~2）
    for r in range(RANK_3, RANK_2 + 1):
        actions.append(ActionPattern(type="pair", rank=r, length=2))

    # 顺子：长度 >=3，不能包含 2、大小王
    max_rank_for_straight = RANK_2 - 1  # 到 A
    for length in range(3, 13):  # 3 到 12 张顺子
        for start in range(RANK_3, max_rank_for_straight - length + 2):
            actions.append(
                ActionPattern(type="straight", start_rank=start, length=length)
            )

    # 炸弹：同点数，多张（3~6 张），主点数 3~2
    for r in range(RANK_3, RANK_2 + 1):
        for length in range(3, 7):  # 最多允许 6 张（利用大小王补牌）
            actions.append(ActionPattern(type="bomb", rank=r, length=length))

    return actions


def num_jokers(counts: np.ndarray) -> int:
    """计算手牌中大小王总数。"""
    return int(counts[RANK_JOKER_SMALL] + counts[RANK_JOKER_BIG])


def can_pay_pattern(
    counts: np.ndarray, pattern: ActionPattern
) -> Tuple[bool, Optional[np.ndarray], bool]:
    """
    判断在给定手牌 counts 下，是否能打出 pattern。
    返回：
        (是否可出, 实际使用的牌向量 used_counts, 是否为纯牌型不含王)。
    """
    used = np.zeros(NUM_RANKS, dtype=np.int32)
    jokers_available = num_jokers(counts)
    uses_joker = False

    if pattern.type == "pass":
        return True, used, True

    if pattern.type == "single":
        r = pattern.rank
        if r is not None and counts[r] > 0:
            used[r] = 1
            return True, used, True
        # 单张不能用王
        return False, None, False

    if pattern.type == "pair":
        r = pattern.rank
        if r is None:
            return False, None, False
        need = 2
        have = int(counts[r])
        if have >= need:
            used[r] = need
            return True, used, True
        # 尝试用王补成对子（比如 3+王）
        deficit = need - have
        if 0 < deficit <= jokers_available:
            used[r] = have
            # 尽量先用小王再用大王
            take_small = min(deficit, counts[RANK_JOKER_SMALL])
            take_big = deficit - take_small
            used[RANK_JOKER_SMALL] = take_small
            used[RANK_JOKER_BIG] = take_big
            uses_joker = True
            return True, used, not uses_joker
        return False, None, False

    if pattern.type == "straight":
        start = pattern.start_rank
        length = pattern.length
        if start is None or length is None:
            return False, None, False
        need_per_rank = 1
        tmp_used = np.zeros(NUM_RANKS, dtype=np.int32)
        jokers = jokers_available
        for offset in range(length):
            r = start + offset
            have = int(counts[r])
            if have >= need_per_rank:
                tmp_used[r] = need_per_rank
            else:
                deficit = need_per_rank - have
                if deficit <= jokers:
                    tmp_used[r] = have
                    jokers -= deficit
                else:
                    return False, None, False
        # 分配实际用到的王
        used = tmp_used.copy()
        used[RANK_JOKER_SMALL] = jokers_available - jokers
        uses_joker = used[RANK_JOKER_SMALL] > 0 or used[RANK_JOKER_BIG] > 0
        return True, used, not uses_joker

    if pattern.type == "bomb":
        r = pattern.rank
        length = pattern.length
        if r is None or length is None:
            return False, None, False
        need = length
        have = int(counts[r])
        if need <= 0:
            return False, None, False
        if have >= need:
            used[r] = need
            return True, used, True
        deficit = need - have
        if deficit <= jokers_available and need >= 3:
            used[r] = have
            take_small = min(deficit, counts[RANK_JOKER_SMALL])
            take_big = deficit - take_small
            used[RANK_JOKER_SMALL] = take_small
            used[RANK_JOKER_BIG] = take_big
            uses_joker = True
            return True, used, not uses_joker
        return False, None, False

    return False, None, False


def bomb_strength(pattern: ActionPattern, used_counts: np.ndarray) -> Tuple[int, int, int]:
    """
    炸弹比较用 key：
      1) 长度（越长越大）
      2) 是否纯炸弹（1: 纯，0: 含王）
      3) 主点数大小
    """
    length = pattern.length or 0
    is_pure = 1 if used_counts[RANK_JOKER_SMALL] == 0 and used_counts[RANK_JOKER_BIG] == 0 else 0
    rank = pattern.rank or 0
    return length, is_pure, rank


def beats(
    last_pat: ActionPattern,
    last_used: np.ndarray,
    new_pat: ActionPattern,
    new_used: np.ndarray,
) -> bool:
    """
    判定 new_pat 是否能在规则下管住 last_pat。
    """
    # 炸弹可以管住任何非炸弹
    if new_pat.type == "bomb" and last_pat.type != "bomb":
        return True

    # 只有炸弹可以管炸弹
    if last_pat.type == "bomb":
        if new_pat.type != "bomb":
            return False
        # 比较炸弹大小
        return bomb_strength(new_pat, new_used) > bomb_strength(last_pat, last_used)

    # 普通牌型之间
    if last_pat.type == "single" and new_pat.type == "single":
        # 2 可以管住任何普通单张（除了王和炸弹），无视 +1
        if new_pat.rank == RANK_2 and last_pat.rank != RANK_2:
            return True
        # 否则必须点数刚好 +1
        return new_pat.rank == last_pat.rank + 1

    if last_pat.type == "pair" and new_pat.type == "pair":
        # 对 2 可以管住任何普通对子
        if new_pat.rank == RANK_2 and last_pat.rank != RANK_2:
            return True
        return new_pat.rank == last_pat.rank + 1

    if last_pat.type == "straight" and new_pat.type == "straight":
        # 顺子：长度必须相同，起始点数严格 +1
        return (
            new_pat.length == last_pat.length
            and new_pat.start_rank == (last_pat.start_rank + 1)
        )

    # 其它类型组合不能互相管（例如对子不能管顺子），除非是炸弹，已在前面处理
    return False


def enumerate_legal_first_moves(hand: np.ndarray) -> List[ActionPattern]:
    """
    首出时，基于手牌枚举所有可能的动作。
    返回合法的 ActionPattern 列表。
    """
    legal_patterns: List[ActionPattern] = []
    
    # 单张：手牌里有哪些牌（除了大小王不能单出）
    for r in range(RANK_3, RANK_2 + 1):
        if hand[r] > 0:
            pat = ActionPattern(type="single", rank=r, length=1)
            can_play, used, _ = can_pay_pattern(hand, pat)
            if can_play and used is not None:
                legal_patterns.append(pat)

    # 对子：手牌里有哪些牌有2张或以上，或者1张+王
    for r in range(RANK_3, RANK_2 + 1):
        have = int(hand[r])
        jokers = int(hand[RANK_JOKER_SMALL] + hand[RANK_JOKER_BIG])
        if have >= 2 or (have >= 1 and jokers >= 1):
            pat = ActionPattern(type="pair", rank=r, length=2)
            can_play, used, _ = can_pay_pattern(hand, pat)
            if can_play and used is not None:
                legal_patterns.append(pat)

    # 顺子：从手牌里最小的牌开始，从小到大枚举
    # 手牌最多6张，所以顺子长度上限是6
    max_straight_length = min(6, int(hand.sum()))
    if max_straight_length >= 3:
        # 找到手牌中最小和最大的非王牌
        min_rank = None
        for r in range(RANK_3, RANK_2 + 1):
            if hand[r] > 0:
                if min_rank is None:
                    min_rank = r
                break
        
        if min_rank is not None:
            # 从每个可能的起始点数开始，从小到大尝试长度
            # 如果从某个起始点开始的短顺子不行，就不用考虑长顺子
            for start in range(min_rank, RANK_2):  # 顺子不能包含2，所以起始点最多到A
                # 从长度3开始，逐步增加长度
                for length in range(3, max_straight_length + 1):
                    # 检查这个顺子是否有效（不能包含2）
                    if start + length - 1 > RANK_2 - 1:  # 不能包含2
                        break
                    pat = ActionPattern(type="straight", start_rank=start, length=length)
                    can_play, used, _ = can_pay_pattern(hand, pat)
                    if can_play and used is not None:
                        legal_patterns.append(pat)
                    else:
                        # 如果短的顺子都不行，就不用考虑从这个起始点开始的长顺子
                        break

    # 炸弹：手牌里有哪些牌有3张或以上，或者可以用王补
    for r in range(RANK_3, RANK_2 + 1):
        have = int(hand[r])
        jokers = int(hand[RANK_JOKER_SMALL] + hand[RANK_JOKER_BIG])
        # 尝试长度3-6的炸弹
        for length in range(3, min(7, have + jokers + 1)):
            if have + jokers >= length:
                pat = ActionPattern(type="bomb", rank=r, length=length)
                can_play, used, _ = can_pay_pattern(hand, pat)
                if can_play and used is not None:
                    legal_patterns.append(pat)

    return legal_patterns


def enumerate_legal_responses(
    hand: np.ndarray,
    last_move_pattern: ActionPattern,
    last_move_used: np.ndarray,
) -> List[ActionPattern]:
    """
    接牌时，只枚举能管住上一个动作的有限情况。
    返回合法的 ActionPattern 列表。
    """
    legal_patterns: List[ActionPattern] = []
    last_type = last_move_pattern.type
    
    # 炸弹可以管住任何非炸弹
    if last_type != "bomb":
        # 枚举所有可能的炸弹
        for r in range(RANK_3, RANK_2 + 1):
            have = int(hand[r])
            jokers = int(hand[RANK_JOKER_SMALL] + hand[RANK_JOKER_BIG])
            for length in range(3, min(7, have + jokers + 1)):
                if have + jokers >= length:
                    pat = ActionPattern(type="bomb", rank=r, length=length)
                    can_play, used, _ = can_pay_pattern(hand, pat)
                    if can_play and used is not None:
                        if beats(last_move_pattern, last_move_used, pat, used):
                            legal_patterns.append(pat)

    # 同类型接牌
    if last_type == "single":
        # 单张：只能是 last_rank + 1，或者2（如果last_rank != 2）
        last_rank = last_move_pattern.rank
        if last_rank is not None:
            # 尝试 +1
            if last_rank + 1 <= RANK_2:
                pat = ActionPattern(type="single", rank=last_rank + 1, length=1)
                can_play, used, _ = can_pay_pattern(hand, pat)
                if can_play and used is not None:
                    legal_patterns.append(pat)
            
            # 尝试2（如果last_rank != 2）
            if last_rank != RANK_2:
                pat = ActionPattern(type="single", rank=RANK_2, length=1)
                can_play, used, _ = can_pay_pattern(hand, pat)
                if can_play and used is not None:
                    legal_patterns.append(pat)

    elif last_type == "pair":
        # 对子：只能是 last_rank + 1，或者2（如果last_rank != 2）
        last_rank = last_move_pattern.rank
        if last_rank is not None:
            # 尝试 +1
            if last_rank + 1 <= RANK_2:
                pat = ActionPattern(type="pair", rank=last_rank + 1, length=2)
                can_play, used, _ = can_pay_pattern(hand, pat)
                if can_play and used is not None:
                    legal_patterns.append(pat)
            
            # 尝试2（如果last_rank != 2）
            if last_rank != RANK_2:
                pat = ActionPattern(type="pair", rank=RANK_2, length=2)
                can_play, used, _ = can_pay_pattern(hand, pat)
                if can_play and used is not None:
                    legal_patterns.append(pat)

    elif last_type == "straight":
        # 顺子：长度必须相同，起始点数 +1
        last_start = last_move_pattern.start_rank
        last_length = last_move_pattern.length
        if last_start is not None and last_length is not None:
            new_start = last_start + 1
            # 检查新起始点数是否有效（不能包含2）
            if new_start + last_length - 1 <= RANK_2 - 1:
                pat = ActionPattern(type="straight", start_rank=new_start, length=last_length)
                can_play, used, _ = can_pay_pattern(hand, pat)
                if can_play and used is not None:
                    if beats(last_move_pattern, last_move_used, pat, used):
                        legal_patterns.append(pat)

    elif last_type == "bomb":
        # 只有更大的炸弹可以管炸弹
        last_rank = last_move_pattern.rank
        last_length = last_move_pattern.length
        if last_rank is not None and last_length is not None:
            # 枚举所有可能的炸弹
            for r in range(RANK_3, RANK_2 + 1):
                have = int(hand[r])
                jokers = int(hand[RANK_JOKER_SMALL] + hand[RANK_JOKER_BIG])
                for length in range(3, min(7, have + jokers + 1)):
                    if have + jokers >= length:
                        pat = ActionPattern(type="bomb", rank=r, length=length)
                        can_play, used, _ = can_pay_pattern(hand, pat)
                        if can_play and used is not None:
                            if beats(last_move_pattern, last_move_used, pat, used):
                                legal_patterns.append(pat)

    return legal_patterns

