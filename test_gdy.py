"""
手工检查干瞪眼规则的测试脚本：
 - 自动跑完一整局；
 - 每一手牌（包含 PASS）都打印出来；
 - 打印出牌前后手牌、当前 round 状态，方便人眼验证规则是否正确。
"""

from typing import List, Tuple, Optional

import json
import time

import numpy as np

from env.core_env import GanDengYanEnv
from env.actions import (
    ActionPattern,
    NUM_RANKS,
    RANK_3,
    RANK_2,
    RANK_JOKER_SMALL,
    RANK_JOKER_BIG,
)


RANK_STRS = {
    RANK_3 + i: s
    for i, s in enumerate(["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"])
}
RANK_STRS[RANK_JOKER_SMALL] = "SJ"  # small joker
RANK_STRS[RANK_JOKER_BIG] = "BJ"    # big joker


# region agent log
DEBUG_LOG_PATH = r"c:\Users\QGC\Desktop\deng_rl\.cursor\debug.log"


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "run1") -> None:
    """将调试信息以 NDJSON 形式追加写入 debug.log。"""
    ts = int(time.time() * 1000)
    entry = {
        "id": f"log_{ts}",
        "timestamp": ts,
        "location": location,
        "message": message,
        "data": data,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        # 调试日志失败不影响主流程
        pass


# endregion


def rank_to_str(r: int) -> str:
    return RANK_STRS.get(r, f"R{r}")


def hand_to_str(counts: np.ndarray) -> str:
    parts: List[str] = []
    for r in range(NUM_RANKS):
        c = int(counts[r])
        if c > 0:
            parts.append(f"{rank_to_str(r)}x{c}")
    return " ".join(parts) if parts else "(empty)"


def pattern_to_str(pat: ActionPattern) -> str:
    if pat.type == "pass":
        return "过牌"
    if pat.type == "single":
        return f"单张 {rank_to_str(pat.rank)}"
    if pat.type == "pair":
        return f"对子 {rank_to_str(pat.rank)}"
    if pat.type == "straight":
        start = pat.start_rank
        length = pat.length
        seq = [rank_to_str(start + i) for i in range(length)]
        return "顺子 " + "-".join(seq)
    if pat.type == "bomb":
        return f"炸弹 长度={pat.length} 点数={rank_to_str(pat.rank)}"
    return f"未知牌型({pat.type})"


class DebugEnv(GanDengYanEnv):
    """在 GanDengYanEnv 基础上加打印，方便人工检查规则。"""

    def reset(self):
        state = super().reset()
        _agent_log(
            hypothesis_id="H1",
            location="test_gdy.py:DebugEnv.reset",
            message="after reset",
            data={
                "dealer": self.dealer,
                "agent_id": self.agent_id,
                "current_player": self.current_player,
            },
        )
        print("========== 新的一局 ==========")
        print(f"庄家: 玩家 {self.dealer}，牌堆剩余张数: {len(self.deck) - self.deck_pos}")
        for pid in range(self.num_players):
            print(f"玩家 {pid} 起始手牌: {hand_to_str(self._hand_counts(pid))}")
        print("================================\n")
        return state

    def _apply_action(self, pid: int, pat: ActionPattern):
        # 出牌前每个玩家的总手牌数，用于检测是否有人摸牌
        before_all = [int(self._remaining_cards(p)) for p in range(self.num_players)]
        # 出牌前当前玩家的手牌
        before = self._hand_counts(pid).copy()

        if pat.type == "pass":
            print(f"玩家 {pid} 选择：过牌")
        else:
            print(f"玩家 {pid} 出牌：{pattern_to_str(pat)}")
            print(f"  出牌前手牌: {hand_to_str(before)}")

        # 调用原始逻辑
        super()._apply_action(pid, pat)

        # 出牌后当前玩家的手牌
        after = self._hand_counts(pid).copy()

        if pat.type != "pass":
            print(f"  出牌后手牌: {hand_to_str(after)}")

        # 检测是否有人摸牌：总手牌数 +1，且能找到唯一一个玩家手牌数 +1
        after_all = [int(self._remaining_cards(p)) for p in range(self.num_players)]
        total_before = sum(before_all)
        total_after = sum(after_all)
        if total_after == total_before + 1:
            drew_players = [
                p for p in range(self.num_players)
                if after_all[p] == before_all[p] + 1
            ]
            if len(drew_players) == 1:
                dp = drew_players[0]
                print(
                    f"  玩家 {dp} 从牌堆摸了一张牌，"
                    f"摸牌后手牌: {hand_to_str(self._hand_counts(dp))}，"
                    f"牌堆剩余张数: {len(self.deck) - self.deck_pos}"
                )

        print(
            f"  状态: 最后出牌玩家={self.last_move_player}, "
            f"连续过牌数={self.passes_since_last_move}, "
            f"是否摸牌后首出={self.must_play}, "
            f"是否废局={self.aborted}，"
            f"牌堆剩余张数={len(self.deck) - self.deck_pos}"
        )
        print("")


def greedy_action_for_player(env: GanDengYanEnv, pid: int) -> int:
    """
    为指定玩家 pid 选择动作（“菜鸟式”贪心策略）：
      - 接牌时优先打小牌，尽量不用炸弹；
      - 首出时在合法动作里（排除炸弹）取“最后一个”动作，
        也就是更倾向于先打顺子、再打对子，最后才是单牌。
      - 若没有任何非炸弹解，则在炸弹中选一个代价最小的炸弹；
      - 若连炸弹也没有，则 PASS（如果规则允许）。
    """
    # 直接复用环境内部的贪心逻辑，保证与训练时的一致性
    return env._select_greedy_action_index(pid)


def run_one_game(seed: Optional[int] = 1233):
    """
    跑一整局干瞪眼，并打印每一手牌：
      - seed: 控制随机性；传固定 seed 方便复现，也可以传 None 完全随机。
    """
    env = DebugEnv(seed=seed)
    env.reset()
    step_idx = 0

    while True:
        pid = env.current_player
        print(f"==== 回合 {step_idx}，当前玩家: 玩家 {pid} ====")

        if pid == 0:
            # 我（玩家 0）出牌，使用简单贪心策略
            action_idx = greedy_action_for_player(env, pid)
            pat = env.actions[action_idx]
            env._apply_action(pid, pat)
        else:
            # 其它玩家使用环境内置贪心策略
            env._opponent_move(pid)

        # 检查是否有人出完牌
        winner: Optional[int] = None
        for p in range(env.num_players):
            if env._remaining_cards(p) == 0:
                winner = p
                break

        if winner is not None:
            reward = env._compute_final_reward()
            print("========== 本局结束 ==========")
            print(
                f"赢家: 玩家 {winner}，"
                f"我方奖励: {reward}，是否废局: {env.aborted}，"
                f"牌堆剩余张数: {len(env.deck) - env.deck_pos}，"
                f"本局炸弹次数: {getattr(env, 'bomb_times', 0)}"
            )
            print("---- 结算详情（每位玩家剩余手牌） ----")
            for p in range(env.num_players):
                remain = env._remaining_cards(p)
                print(f"玩家 {p} 剩余 {remain} 张: {hand_to_str(env._hand_counts(p))}")
            print("================================")
            break

        # 轮到下一个玩家：
        # 普通情况下按座位顺序轮转；
        # 若在 _apply_action 中结束了一轮并开启新一轮，env.current_player
        # 已经被设置为赢家，此时不再额外 +1。
        if not env.aborted and not (
            env.last_move_pattern is None and env.passes_since_last_move == 0
        ):
            env.current_player = (env.current_player + 1) % env.num_players
        step_idx += 1


if __name__ == "__main__":
    # 你可以修改 seed 或改成 None 看更多不同局面
    run_one_game(seed=4)

