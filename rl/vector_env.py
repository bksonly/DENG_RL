"""
简单的向量环境封装：串行运行多个 GanDengYanEnv 实例。
"""

from typing import List, Dict, Any, Tuple

import numpy as np

from env.core_env import GanDengYanEnv


class VectorEnv:
    """
    串行 VectorEnv，实现统一的 reset / step 接口。
    支持混合对手策略：每个环境可以有不同的对手策略。
    """

    def __init__(self, num_envs: int, opponent_agents=None, opponent_strategy_types=None):
        """
        opponent_agents: List[Optional[PPOAgent]] - 每个环境的对手agent，None表示贪心策略
        opponent_strategy_types: List[str] - 每个环境的对手策略类型（用于记录）
        """
        self.num_envs = num_envs
        self.opponent_agents = opponent_agents if opponent_agents is not None else [None] * num_envs
        self.opponent_strategy_types = opponent_strategy_types if opponent_strategy_types is not None else ["greedy"] * num_envs
        
        # 为每个环境的每个对手维护 hidden state
        self.opponent_hidden_states = None
        if any(agent is not None for agent in self.opponent_agents):
            self.opponent_hidden_states = [[None for _ in range(3)] for _ in range(num_envs)]  # 3个对手
        
        self.envs = []
        for env_idx in range(num_envs):
            opp_agent = self.opponent_agents[env_idx]
            opp_hidden = self.opponent_hidden_states[env_idx] if self.opponent_hidden_states is not None else None
            strategy_type = self.opponent_strategy_types[env_idx]
            env = GanDengYanEnv(
                opponent_agent=opp_agent,
                opponent_hidden_states=opp_hidden,
                opponent_strategy_type=strategy_type,
            )
            self.envs.append(env)
    
    def set_opponent_agents(self, opponent_agents: List, opponent_strategy_types: List[str]):
        """设置每个环境的对手策略"""
        self.opponent_agents = opponent_agents
        self.opponent_strategy_types = opponent_strategy_types
        for env_idx, env in enumerate(self.envs):
            env.opponent_agent = opponent_agents[env_idx]
        if any(agent is not None for agent in opponent_agents) and self.opponent_hidden_states is None:
            self.opponent_hidden_states = [[None for _ in range(3)] for _ in range(self.num_envs)]
        for env_idx, env in enumerate(self.envs):
            if self.opponent_hidden_states is not None:
                env.opponent_hidden_states = self.opponent_hidden_states[env_idx]
            # 更新每个环境的对手策略类型（仅在使用固定策略时生效）
            env.opponent_strategy_type = self.opponent_strategy_types[env_idx]

    def reset(self) -> np.ndarray:
        states = [env.reset() for env in self.envs]
        return np.stack(states, axis=0)  # [E, C, 4, 15]

    def step(
        self, actions: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        next_states = []
        rewards = []
        dones = []
        infos = []
        for env_idx, (env, a) in enumerate(zip(self.envs, actions)):
            s, r, d, info = env.step(a)
            if d:
                # 若终局（包括废局），立即 reset
                # 同时清理对手的 RNN hidden state，避免跨局泄漏记忆
                if self.opponent_hidden_states is not None:
                    for i in range(len(self.opponent_hidden_states[env_idx])):
                        self.opponent_hidden_states[env_idx][i] = None
                s = env.reset()
            next_states.append(s)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        return (
            np.stack(next_states, axis=0),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

