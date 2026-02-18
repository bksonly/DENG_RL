"""
PPO Agent 定义及 GAE 等相关工具函数。
"""

from typing import Dict, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import Actor, Critic


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
        seq_len: int = 32,
    ):
        self.num_actions = num_actions
        self.device = device
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.seq_len = seq_len

        # 独立的 Actor 和 Critic 网络
        self.actor = Actor(num_actions=num_actions, global_dim=5).to(self.device)
        self.critic = Critic(global_dim=5).to(self.device)
        
        # 共享一个 optimizer 同时优化两个网络
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

    @torch.no_grad()
    def select_action(
        self,
        actor_state: np.ndarray,
        critic_state: np.ndarray,
        global_feats: np.ndarray,
        legal_mask: np.ndarray,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        compute_value: bool = True,
    ) -> Tuple[int, float, float, Tuple[torch.Tensor, torch.Tensor]]:
        """
        根据当前状态、全局特征和合法动作 mask 选择动作。
        actor_state: [3, 4, 15] - Actor的状态（只能看到自己的手牌）
        critic_state: [6, 4, 15] - Critic的状态（能看到所有手牌）
        hidden: (h_actor, c_actor) or None
        compute_value: 是否计算 value（默认 True，保持向后兼容）
        返回：action, log_prob, value, new_hidden
        """
        actor_state_t = (
            torch.from_numpy(actor_state).float().unsqueeze(0).to(self.device)
        )  # [1,3,4,15]
        global_feats_t = (
            torch.from_numpy(global_feats).float().unsqueeze(0).to(self.device)
        )  # [1,5]
        legal_mask_t = (
            torch.from_numpy(legal_mask.astype(np.bool_))
            .unsqueeze(0)
            .to(self.device)
        )  # [1,A]

        # 处理 hidden state（兼容旧格式：若传入 4-tensor，只取 actor 部分）
        actor_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]
        if hidden is None:
            actor_hidden = None
        else:
            # 兼容：某些旧调用可能仍传入 (h_actor, c_actor, h_critic, c_critic)
            if isinstance(hidden, tuple) and len(hidden) == 4:
                h_actor, c_actor, _, _ = hidden  # type: ignore[misc]
                actor_hidden = (h_actor, c_actor)
            else:
                actor_hidden = hidden

        # 调用 Actor
        logits, (new_h_actor, new_c_actor) = self.actor(actor_state_t, global_feats_t, actor_hidden)

        # 将非法动作 logits 置为 -1e9
        logits_masked = logits.clone()
        logits_masked[~legal_mask_t] = -1e9

        probs = torch.softmax(logits_masked, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # 如果需要计算 value，调用 Critic
        if compute_value:
            critic_state_t = (
                torch.from_numpy(critic_state).float().unsqueeze(0).to(self.device)
            )  # [1,6,4,15]
            value = self.critic(critic_state_t, global_feats_t)
            new_hidden = (new_h_actor, new_c_actor)
            return (
                int(action.item()),
                float(log_prob.item()),
                float(value.item()),
                new_hidden,
            )
        else:
            # 只调用 actor，返回 0.0 作为 value，只返回 actor 的 hidden state
            new_hidden = (new_h_actor, new_c_actor)
            return (
                int(action.item()),
                float(log_prob.item()),
                0.0,
                new_hidden,
            )

    @torch.no_grad()
    def select_action_batch(
        self,
        actor_states: np.ndarray,        # [E, 3, 4, 15]
        critic_states: np.ndarray,       # [E, 6, 4, 15]
        global_feats: np.ndarray,        # [E, 5]
        legal_masks: np.ndarray,         # [E, A] bool / int
        hiddens: Optional[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        compute_value: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Batched 版本的 select_action：
        - 一次性对 E 个环境的状态进行前向，返回每个环境的动作/对数概率/value 和新的 hidden。
        - actor_states: [E, 3, 4, 15]
        - critic_states: [E, 6, 4, 15]
        - global_feats: [E, 5]
        - legal_masks: [E, A]，True 表示该动作合法
        - hiddens: 长度为 E 的列表/元组，每个元素为 (h, c) 或 None

        返回：
        - actions: [E] int
        - log_probs: [E] float
        - values: [E] float（若 compute_value=False，则为0）
        - new_hiddens: 长度为 E 的列表，每个元素为 (h, c) 或 None
        """
        actor_states_t = torch.from_numpy(actor_states).float().to(self.device)          # [E,3,4,15]
        critic_states_t = torch.from_numpy(critic_states).float().to(self.device)        # [E,6,4,15]
        global_feats_t = torch.from_numpy(global_feats).float().to(self.device)          # [E,5]
        legal_masks_t = torch.from_numpy(legal_masks.astype(np.bool_)).to(self.device)   # [E,A]

        E = actor_states_t.size(0)

        # 处理 hidden state：hiddens 为长度 E 的列表/元组
        if hiddens is None:
            h0 = torch.zeros(1, E, self.actor.actor_lstm.hidden_size, device=self.device)
            c0 = torch.zeros(1, E, self.actor.actor_lstm.hidden_size, device=self.device)
        else:
            hs = []
            cs = []
            for item in hiddens:
                if item is None:
                    hs.append(torch.zeros(1, 1, self.actor.actor_lstm.hidden_size, device=self.device))
                    cs.append(torch.zeros(1, 1, self.actor.actor_lstm.hidden_size, device=self.device))
                else:
                    h, c = item
                    hs.append(h.to(self.device))
                    cs.append(c.to(self.device))
            h0 = torch.cat(hs, dim=1)  # [1,E,H]
            c0 = torch.cat(cs, dim=1)  # [1,E,H]

        # Actor 前向：Actor 期望输入 [B,3,4,15]，内部会加一维时间
        logits, (new_h, new_c) = self.actor(actor_states_t, global_feats_t, (h0, c0))  # logits: [E,A]

        # 将非法动作 logits 置为 -1e9
        logits_masked = logits.clone()
        logits_masked[~legal_masks_t] = -1e9

        probs = torch.softmax(logits_masked, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions_t = dist.sample()              # [E]
        log_probs_t = dist.log_prob(actions_t) # [E]

        if compute_value:
            # Critic 前向
            B = critic_states_t.size(0)
            critic_in = critic_states_t.reshape(B, *critic_states_t.shape[1:])  # [B,6,4,15]
            value_t = self.critic(critic_in, global_feats_t)  # [B]
            values_np = value_t.detach().cpu().numpy().astype(np.float32)
        else:
            values_np = np.zeros((E,), dtype=np.float32)

        actions_np = actions_t.detach().cpu().numpy().astype(np.int64)
        log_probs_np = log_probs_t.detach().cpu().numpy().astype(np.float32)

        # 拆分新的 hidden state 为 per-env 格式 (h,c)，形状 [1,1,H]
        new_hiddens = []
        for i in range(E):
            h_i = new_h[:, i : i + 1, :].detach()
            c_i = new_c[:, i : i + 1, :].detach()
            new_hiddens.append((h_i, c_i))

        return actions_np, log_probs_np, values_np, tuple(new_hiddens)

    def update(self, batch: Dict[str, torch.Tensor]):
        """
        使用 PPO-Clip 算法更新策略与价值函数（Actor 为 RNN，按 seq_len 分段 unroll）。
        batch 中的张量在调用前应已搬到 CPU，由本函数负责搬到 device。
        """
        actor_states = batch["actor_states"]      # [T,E,3,4,15]
        critic_states = batch["critic_states"]    # [T,E,6,4,15]
        actions = batch["actions"]                # [T,E]
        old_log_probs = batch["log_probs"]        # [T,E]
        returns = batch["returns"]                # [T,E]
        advantages = batch["advantages"]          # [T,E]
        legal_masks = batch["legal_masks"]        # [T,E,A]
        global_feats = batch["global_feats"]      # [T,E,5]
        dones_t = batch["dones_t"]                # [T,E] bool
        actor_h0 = batch["actor_h0"]              # [T,E,H]
        actor_c0 = batch["actor_c0"]              # [T,E,H]
        valid_mask = batch.get("valid_mask", None)  # [T,E] bool（废局过滤）
        opponent_strategy_types = batch.get("opponent_strategy_types", None)  # List[str] or None

        T, E = actions.shape
        seq_len = int(self.seq_len)
        assert seq_len > 0 and seq_len <= T, f"seq_len must be in (0, T], got {seq_len} with T={T}"

        # 优势归一化（基于整个 rollout）
        if valid_mask is None:
            valid_mask_t = torch.ones((T, E), dtype=torch.bool)
        else:
            valid_mask_t = valid_mask.bool()

        # 进一步剔除 idiot 对手的数据：只用于评估，不参与训练
        if opponent_strategy_types is not None:
            idiot_env_mask = torch.tensor(
                [s == "idiot" for s in opponent_strategy_types], dtype=torch.bool
            )
            if idiot_env_mask.any():
                valid_mask_t[:, idiot_env_mask] = False

        valid_adv = advantages[valid_mask_t].float()
        if valid_adv.numel() > 0:
            adv_mean = valid_adv.mean()
            adv_std = valid_adv.std()
        else:
            adv_mean = advantages.float().mean()
            adv_std = advantages.float().std()
        advantages = (advantages.float() - adv_mean) / (adv_std + 1e-8)

        # 构建所有可用的 (env_idx, t0) 序列起点（默认非重叠 stride=seq_len）
        valid_mask_np = valid_mask_t.numpy()
        seq_starts = [
            (e, t0)
            for e in range(E)
            for t0 in range(0, T - seq_len + 1, seq_len)
            if valid_mask_np[t0 : t0 + seq_len, e].all()
        ]
        if len(seq_starts) == 0:
            return

        seqs_per_batch = max(1, self.batch_size // seq_len)
        num_seqs = len(seq_starts)
        inds = np.arange(num_seqs)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(inds)
            for start in range(0, num_seqs, seqs_per_batch):
                mb_inds = inds[start : start + seqs_per_batch]

                # 组装 minibatch 序列张量：[L, B, ...]
                mb_actor_states = []
                mb_critic_states = []
                mb_actions = []
                mb_old_log_probs = []
                mb_returns = []
                mb_advantages = []
                mb_legal_masks = []
                mb_global_feats = []
                mb_dones = []
                mb_h0 = []
                mb_c0 = []

                for idx in mb_inds:
                    e, t0 = seq_starts[int(idx)]
                    sl = slice(t0, t0 + seq_len)
                    mb_actor_states.append(actor_states[sl, e])
                    mb_critic_states.append(critic_states[sl, e])
                    mb_actions.append(actions[sl, e])
                    mb_old_log_probs.append(old_log_probs[sl, e])
                    mb_returns.append(returns[sl, e])
                    mb_advantages.append(advantages[sl, e])
                    mb_legal_masks.append(legal_masks[sl, e])
                    mb_global_feats.append(global_feats[sl, e])
                    mb_dones.append(dones_t[sl, e])
                    mb_h0.append(actor_h0[t0, e])
                    mb_c0.append(actor_c0[t0, e])

                mb_actor_states_t = torch.stack(mb_actor_states, dim=1).to(self.device)      # [L,B,3,4,15]
                mb_critic_states_t = torch.stack(mb_critic_states, dim=1).to(self.device)    # [L,B,6,4,15]
                mb_actions_t = torch.stack(mb_actions, dim=1).to(self.device)                # [L,B]
                mb_old_log_probs_t = torch.stack(mb_old_log_probs, dim=1).to(self.device)    # [L,B]
                mb_returns_t = torch.stack(mb_returns, dim=1).to(self.device)                # [L,B]
                mb_advantages_t = torch.stack(mb_advantages, dim=1).to(self.device)          # [L,B]
                mb_legal_masks_t = torch.stack(mb_legal_masks, dim=1).to(self.device)        # [L,B,A]
                mb_global_feats_t = torch.stack(mb_global_feats, dim=1).to(self.device)      # [L,B,5]
                mb_dones_t = torch.stack(mb_dones, dim=1).to(self.device)                    # [L,B]

                B = mb_actions_t.size(1)
                h = torch.stack(mb_h0, dim=0).to(self.device).unsqueeze(0)  # [1,B,H]
                c = torch.stack(mb_c0, dim=0).to(self.device).unsqueeze(0)  # [1,B,H]

                new_log_probs_steps = []
                entropies = []

                # RNN unroll
                for t in range(seq_len):
                    logits, (h, c) = self.actor(mb_actor_states_t[t], mb_global_feats_t[t], (h, c))

                    logits_masked = logits.clone()
                    logits_masked[~mb_legal_masks_t[t]] = -1e9
                    probs = torch.softmax(logits_masked, dim=-1)
                    dist = torch.distributions.Categorical(probs)

                    logp = dist.log_prob(mb_actions_t[t])
                    new_log_probs_steps.append(logp)
                    entropies.append(dist.entropy())

                    # done 后清理 hidden，避免跨局泄漏
                    done_mask = (1.0 - mb_dones_t[t].float()).view(1, B, 1)
                    h = h * done_mask
                    c = c * done_mask

                new_log_probs_t = torch.stack(new_log_probs_steps, dim=0)  # [L,B]
                entropy = torch.cat(entropies, dim=0).mean()

                # Critic（前馈，无需序列 hidden）
                values = self.critic(
                    mb_critic_states_t.reshape(seq_len * B, *mb_critic_states_t.shape[2:]),
                    mb_global_feats_t.reshape(seq_len * B, mb_global_feats_t.shape[-1]),
                ).reshape(seq_len, B)

                ratio = torch.exp(new_log_probs_t - mb_old_log_probs_t)
                surr1 = ratio * mb_advantages_t
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages_t
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = (mb_returns_t - values).pow(2).mean()

                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                params = list(self.actor.parameters()) + list(self.critic.parameters())
                nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.optimizer.step()

    def save_checkpoint(self, filepath: str, episode: int, best_reward: float = None):
        """
        保存 checkpoint，包含模型权重、optimizer 状态和训练进度。
        """
        checkpoint = {
            "episode": episode,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_reward": best_reward,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str, device: torch.device):
        """
        从 checkpoint 加载模型和训练进度。
        返回：episode, best_reward
        """
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        # 由于网络结构可能会迭代（例如移除 Critic LSTM），这里使用 strict=False 提升兼容性。
        self.actor.load_state_dict(checkpoint.get("actor_state_dict", {}), strict=False)
        self.critic.load_state_dict(checkpoint.get("critic_state_dict", {}), strict=False)
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception:
            # optimizer 结构不兼容时允许跳过（继续训练会重新累积动量等状态）
            pass
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
        # 标准 GAE：使用当前步 dones[t] 截断 bootstrapping
        # - 若 dones[t]=True，则 next_non_terminal=0，下一状态 value 不参与
        # - 若 dones[t]=False，则可用 next_values（t==T-1 用 last_values，否则用 values[t+1]）
        next_values = last_values if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t].astype(np.float32)

        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return returns, advantages

