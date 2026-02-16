"""
PPO Agent 定义及 GAE 等相关工具函数。
"""

from typing import Dict, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import ActorCritic


class PPOAgent:
    """
    PPO Agent，封装策略网络与更新逻辑。
    训练超参数通过构造函数显式传入，方便在入口脚本集中管理。
    """

    def __init__(
        self,
        num_actions: int,
        lr: float,
        clip_eps: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        ppo_epochs: int,
        batch_size: int,
        device: torch.device,
    ):
        self.num_actions = num_actions
        self.device = device
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # ActorCritic 同时接收空间状态和 5 维全局特征
        self.net = ActorCritic(num_actions=num_actions, global_dim=5).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,
        global_feats: np.ndarray,
        legal_mask: np.ndarray,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[int, float, float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        根据当前状态、全局特征和合法动作 mask 选择动作。
        返回：action, log_prob, value, new_hidden
        """
        state_t = (
            torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        )  # [1,C,4,15]
        global_feats_t = (
            torch.from_numpy(global_feats).float().unsqueeze(0).to(self.device)
        )  # [1,5]
        legal_mask_t = (
            torch.from_numpy(legal_mask.astype(np.bool_))
            .unsqueeze(0)
            .to(self.device)
        )  # [1,A]

        logits, value, new_hidden = self.net(state_t, global_feats_t, hidden)

        # 将非法动作 logits 置为 -1e9
        logits_masked = logits.clone()
        logits_masked[~legal_mask_t] = -1e9

        probs = torch.softmax(logits_masked, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
            new_hidden,
        )

    def update(self, batch: Dict[str, torch.Tensor]):
        """
        使用 PPO-Clip 算法更新策略与价值函数。
        batch 中的张量在调用前应已搬到 CPU，由本函数负责搬到 device。
        """
        states = batch["states"]  # [N, C_s, 4, 15]
        actions = batch["actions"]  # [N]
        old_log_probs = batch["log_probs"]  # [N]
        returns = batch["returns"]  # [N]
        advantages = batch["advantages"]  # [N]
        legal_masks = batch["legal_masks"]  # [N, A]
        global_feats = batch["global_feats"]  # [N, 5]

        N = states.size(0)
        inds = np.arange(N)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(inds)
            for start in range(0, N, self.batch_size):
                end = start + self.batch_size
                mb_inds = inds[start:end]

                mb_states = states[mb_inds].to(self.device)
                mb_actions = actions[mb_inds].to(self.device)
                mb_old_log_probs = old_log_probs[mb_inds].to(self.device)
                mb_returns = returns[mb_inds].to(self.device)
                mb_advantages = advantages[mb_inds].to(self.device)
                mb_legal_masks = legal_masks[mb_inds].to(self.device)
                mb_global_feats = global_feats[mb_inds].to(self.device)

                logits, values, _ = self.net(mb_states, mb_global_feats, None)

                # mask 非法动作
                logits_masked = logits.clone()
                logits_masked[~mb_legal_masks] = -1e9
                probs = torch.softmax(logits_masked, dim=-1)
                dist = torch.distributions.Categorical(probs)

                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                ) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = (mb_returns - values).pow(2).mean()

                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def save_checkpoint(self, filepath: str, episode: int, best_reward: float = None):
        """
        保存 checkpoint，包含模型权重、optimizer 状态和训练进度。
        """
        checkpoint = {
            "episode": episode,
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_reward": best_reward,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str, device: torch.device):
        """
        从 checkpoint 加载模型和训练进度。
        返回：episode, best_reward
        """
        checkpoint = torch.load(filepath, map_location=device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        episode = checkpoint.get("episode", 0)
        best_reward = checkpoint.get("best_reward", None)
        return episode, best_reward


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于 GAE(lambda) 计算 returns 和 advantages。
    输入 shape 均为 [T, E]。
    """
    T, E = rewards.shape
    advantages = np.zeros((T, E), dtype=np.float32)
    last_gae = np.zeros(E, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_values = last_values
            next_non_terminal = 1.0 - dones[t].astype(np.float32)
        else:
            next_values = values[t + 1]
            next_non_terminal = 1.0 - dones[t + 1].astype(np.float32)

        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return returns, advantages

