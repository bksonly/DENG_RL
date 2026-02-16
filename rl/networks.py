"""
Actor-Critic 网络定义，处理
- 空间状态 `[C_s, 4, 15]`，以及
- 全局标量特征 `global_feats`（长度为 5）。
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """
    处理 `[C_s, 4, 15]` 空间状态和 5 维全局特征的 Actor-Critic 网络，包含 LSTM 记忆。

    参数：
      - num_actions: 动作空间大小
      - hidden_size: LSTM 以及空间特征投影维度
      - in_channels_spatial: 空间通道数 C_s（默认为 3）
      - global_dim: 全局特征维度（默认为 5）
      - global_hidden_dim: 全局特征 MLP 输出维度

    输入：
      - state: [B, C_s, 4, 15]
      - global_feats: [B, global_dim]
      - hidden_state: (h, c)，形状 [num_layers, B, hidden_size]
    """

    def __init__(
        self,
        num_actions: int,
        hidden_size: int = 256,  # 增加网络容量：从 128 提升到 256
        in_channels_spatial: int = 3,
        global_dim: int = 5,
        global_hidden_dim: int = 64,  # 增加全局特征维度：从 32 提升到 64
    ):
        super().__init__()
        self.num_actions = num_actions

        # 简单的 2D CNN 作为空间特征提取 backbone
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels_spatial,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        conv_out_dim = 64 * 4 * 15

        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, hidden_size),
            nn.ReLU(),
        )

        # 全局特征 MLP 分支
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, global_hidden_dim),
            nn.ReLU(),
        )

        # LSTM 记忆模块，输入为拼接后的空间特征 + 全局特征
        lstm_input_size = hidden_size + global_hidden_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=1,
        )

        # Actor / Critic 头
        self.actor_head = nn.Linear(hidden_size, num_actions)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        state: torch.Tensor,
        global_feats: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        state: [B, C_s, 4, 15]
        global_feats: [B, global_dim]
        hidden: (h, c) or None
        返回：logits, value, new_hidden
        """
        B = state.size(0)

        # 空间特征
        x = self.conv(state)  # [B, 64, 4, 15]
        x = x.view(B, -1)
        x = self.fc(x)  # [B, H]

        # 全局特征
        g = self.global_mlp(global_feats)  # [B, G]

        # 拼接后送入 LSTM
        x = torch.cat([x, g], dim=-1)  # [B, H+G]

        # LSTM 期望输入形状 [T, B, H]，这里单步 T=1
        x = x.unsqueeze(0)  # [1, B, H+G]

        if hidden is None:
            h0 = torch.zeros(1, B, self.lstm.hidden_size, device=state.device)
            c0 = torch.zeros(1, B, self.lstm.hidden_size, device=state.device)
            hidden = (h0, c0)

        lstm_out, new_hidden = self.lstm(x, hidden)  # [1, B, H]
        lstm_out = lstm_out.squeeze(0)  # [B, H]

        logits = self.actor_head(lstm_out)  # [B, A]
        value = self.critic_head(lstm_out).squeeze(-1)  # [B]

        return logits, value, new_hidden

