"""
简单的向量环境封装：串行运行多个 GanDengYanEnv 实例。
"""

from typing import List, Dict, Any, Tuple

import numpy as np

from env.core_env import GanDengYanEnv


class VectorEnv:
    """
    串行 VectorEnv，实现统一的 reset / step 接口。
    """

    def __init__(self, num_envs: int, opponent_agent=None):
        self.num_envs = num_envs
        self.opponent_agent = opponent_agent
        # 为每个环境的每个对手维护 hidden state
        self.opponent_hidden_states = None
        if opponent_agent is not None:
            self.opponent_hidden_states = [[None for _ in range(3)] for _ in range(num_envs)]  # 3个对手
        
        self.envs = []
        for env_idx in range(num_envs):
            opp_hidden = self.opponent_hidden_states[env_idx] if self.opponent_hidden_states is not None else None
            env = GanDengYanEnv(opponent_agent=opponent_agent, opponent_hidden_states=opp_hidden)
            self.envs.append(env)
    
    def set_opponent_agent(self, opponent_agent):
        """设置对手策略（用于切换自对弈）"""
        self.opponent_agent = opponent_agent
        for env in self.envs:
            env.opponent_agent = opponent_agent
        if opponent_agent is not None and self.opponent_hidden_states is None:
            self.opponent_hidden_states = [[None for _ in range(3)] for _ in range(self.num_envs)]
            for env_idx, env in enumerate(self.envs):
                env.opponent_hidden_states = self.opponent_hidden_states[env_idx]

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

