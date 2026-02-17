"""
Actor-Critic 网络定义，处理
- 空间状态 `[C_s, 4, 15]`，以及
- 全局标量特征 `global_feats`（长度为 5）。
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Actor 网络，用于策略输出。
    
    Actor输入：3通道 [自己的手牌, 墓地, 目标牌]

    参数：
      - num_actions: 动作空间大小
      - hidden_size: LSTM 以及空间特征投影维度
      - actor_channels: Actor输入通道数（默认为 3）
      - global_dim: 全局特征维度（默认为 5）
      - global_hidden_dim: 全局特征 MLP 输出维度

    输入：
      - actor_state: [B, 3, 4, 15] - Actor的状态（只能看到自己的手牌）
      - global_feats: [B, global_dim]
      - hidden: (h_actor, c_actor) or None，形状 [num_layers, B, hidden_size]
    
    返回：
      - logits: [B, num_actions]
      - new_hidden: (h_actor, c_actor)
    """

    def __init__(
        self,
        num_actions: int,
        hidden_size: int = 256,
        actor_channels: int = 3,
        global_dim: int = 5,
        global_hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_actions = num_actions

        # Actor的CNN backbone
        self.actor_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=actor_channels,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        conv_out_dim = 64 * 4 * 15

        # Actor的特征投影
        self.actor_fc = nn.Sequential(
            nn.Linear(conv_out_dim, hidden_size),
            nn.ReLU(),
        )

        # 全局特征 MLP 分支
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, global_hidden_dim),
            nn.ReLU(),
        )

        # Actor的LSTM记忆模块
        lstm_input_size = hidden_size + global_hidden_dim
        self.actor_lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=1,
        )

        # Actor 头
        self.actor_head = nn.Linear(hidden_size, num_actions)

    def forward(
        self,
        actor_state: torch.Tensor,
        global_feats: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        actor_state: [B, 3, 4, 15] - Actor的状态（只能看到自己的手牌）
        global_feats: [B, global_dim]
        hidden: (h_actor, c_actor) or None
        返回：logits, new_hidden
        """
        B = actor_state.size(0)

        # Actor分支：空间特征提取
        actor_x = self.actor_conv(actor_state)  # [B, 64, 4, 15]
        actor_x = actor_x.view(B, -1)
        actor_x = self.actor_fc(actor_x)  # [B, H]

        # 全局特征
        g = self.global_mlp(global_feats)  # [B, G]

        # Actor: 拼接后送入 LSTM
        actor_x = torch.cat([actor_x, g], dim=-1)  # [B, H+G]
        actor_x = actor_x.unsqueeze(0)  # [1, B, H+G]

        # 处理hidden state
        if hidden is None:
            h_actor = torch.zeros(1, B, self.actor_lstm.hidden_size, device=actor_state.device)
            c_actor = torch.zeros(1, B, self.actor_lstm.hidden_size, device=actor_state.device)
            hidden = (h_actor, c_actor)
        else:
            h_actor, c_actor = hidden

        # Actor LSTM
        actor_lstm_out, (new_h_actor, new_c_actor) = self.actor_lstm(
            actor_x, (h_actor, c_actor)
        )  # [1, B, H]
        actor_lstm_out = actor_lstm_out.squeeze(0)  # [B, H]

        logits = self.actor_head(actor_lstm_out)  # [B, A]

        new_hidden = (new_h_actor, new_c_actor)

        return logits, new_hidden


class Critic(nn.Module):
    """
    Critic 网络，用于价值估计。
    
    Critic输入：6通道 [自己的手牌, 对手1手牌, 对手2手牌, 对手3手牌, 墓地, 目标牌]

    参数：
      - hidden_size: LSTM 以及空间特征投影维度
      - critic_channels: Critic输入通道数（默认为 6）
      - global_dim: 全局特征维度（默认为 5）
      - global_hidden_dim: 全局特征 MLP 输出维度

    输入：
      - critic_state: [B, 6, 4, 15] - Critic的状态（能看到所有手牌）
      - global_feats: [B, global_dim]
      - hidden: (h_critic, c_critic) or None，形状 [num_layers, B, hidden_size]
    
    返回：
      - value: [B]
      - new_hidden: (h_critic, c_critic)
    """

    def __init__(
        self,
        hidden_size: int = 256,
        critic_channels: int = 6,
        global_dim: int = 5,
        global_hidden_dim: int = 64,
    ):
        super().__init__()

        # Critic的CNN backbone
        self.critic_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=critic_channels,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        conv_out_dim = 64 * 4 * 15

        # Critic的特征投影
        self.critic_fc = nn.Sequential(
            nn.Linear(conv_out_dim, hidden_size),
            nn.ReLU(),
        )

        # 全局特征 MLP 分支
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, global_hidden_dim),
            nn.ReLU(),
        )

        # Critic的LSTM记忆模块
        lstm_input_size = hidden_size + global_hidden_dim
        self.critic_lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=1,
        )

        # Critic 头
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        critic_state: torch.Tensor,
        global_feats: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        critic_state: [B, 6, 4, 15] - Critic的状态（能看到所有手牌）
        global_feats: [B, global_dim]
        hidden: (h_critic, c_critic) or None
        返回：value, new_hidden
        """
        B = critic_state.size(0)

        # Critic分支：空间特征提取
        critic_x = self.critic_conv(critic_state)  # [B, 64, 4, 15]
        critic_x = critic_x.view(B, -1)
        critic_x = self.critic_fc(critic_x)  # [B, H]

        # 全局特征
        g = self.global_mlp(global_feats)  # [B, G]

        # Critic: 拼接后送入 LSTM
        critic_x = torch.cat([critic_x, g], dim=-1)  # [B, H+G]
        critic_x = critic_x.unsqueeze(0)  # [1, B, H+G]

        # 处理hidden state
        if hidden is None:
            h_critic = torch.zeros(1, B, self.critic_lstm.hidden_size, device=critic_state.device)
            c_critic = torch.zeros(1, B, self.critic_lstm.hidden_size, device=critic_state.device)
            hidden = (h_critic, c_critic)
        else:
            h_critic, c_critic = hidden

        # Critic LSTM
        critic_lstm_out, (new_h_critic, new_c_critic) = self.critic_lstm(
            critic_x, (h_critic, c_critic)
        )  # [1, B, H]
        critic_lstm_out = critic_lstm_out.squeeze(0)  # [B, H]

        value = self.critic_head(critic_lstm_out).squeeze(-1)  # [B]

        new_hidden = (new_h_critic, new_c_critic)

        return value, new_hidden

