# 干瞪眼游戏强化学习项目

使用 PPO 算法训练一个能玩"干瞪眼"扑克牌游戏的智能体。

## 项目结构

```
DENG_RL/
├── env/              # 游戏环境
│   ├── core_env.py   # 核心游戏逻辑
│   ├── actions.py    # 动作空间和规则判断
│   └── encoding.py   # 状态编码
├── rl/               # 强化学习算法
│   ├── networks.py   # Actor 和 Critic 网络
│   ├── ppo_agent.py # PPO Agent 实现
│   ├── rollout.py   # 轨迹收集
│   └── vector_env.py # 并行环境封装
└── train_ppo.py     # 主训练脚本
```

## 游戏规则

### 基本规则

- 4人游戏，54张牌（52张标准牌 + 2张王）
- 牌的大小顺序：3 < 4 < 5 < 6 < 7 < 8 < 9 < 10 < J < Q < K < A < 2
- 核心机制：出牌必须比上家刚好大1点（"大一点"规则）
- 特殊牌：
  - 2：可以管住任何普通单张或对子（除王和炸弹外）
  - 王：万能牌，可代替任何牌，但不能单出

### 牌型

1. 单张：例如 3
2. 对子：例如 3-3 或 3-王
3. 顺子：3张或更多连续牌，接牌时点数严格+1
4. 炸弹：3张或更多同点数牌，可管住任何非炸弹牌型

### 胜利与奖励

- 第一个出完手牌的玩家获胜
- 奖励计算（仅终局结算）：
  - 输家基础损失 = 剩余手牌数
  - 通关（整局未出牌）：损失 × 2
  - 剩余手牌含王：损失 × 2
  - 本局炸弹次数 k：所有输家损失 × 2^k
  - 个人损失封顶：min(损失, 20)
  - 赢家奖励 = 所有输家损失之和
  - 输家奖励 = -自己的损失

### 特殊规则

- 赢家连庄：获胜者下一局继续坐庄，直到输牌
- 庄家发6张牌，其他玩家发5张
- 牌堆耗尽时游戏无效（废局），不参与训练

## 状态编码

### Actor 状态（3通道）

- Channel 0：自己的手牌 [4, 15]
- Channel 1：墓地（全局已出牌）[4, 15]
- Channel 2：目标牌（当前必须管的牌）[4, 15]

### Critic 状态（6通道）

- Channel 0：自己的手牌
- Channel 1-3：三个对手的手牌（按相对位置）
- Channel 4：墓地
- Channel 5：目标牌

### 全局特征（5维向量）

1. 自己剩余手牌数（归一化）
2. 左家剩余手牌数（归一化）
3. 对家剩余手牌数（归一化）
4. 右家剩余手牌数（归一化）
5. 是否处于"摸牌后首出"状态（0/1）

## 网络架构

### Actor 网络

- 输入：actor_state [B, 3, 4, 15] + global_feats [B, 5]
- CNN backbone：独立处理3通道状态
- MLP：处理全局特征
- LSTM：融合特征并建模历史
- 输出：动作 logits [B, num_actions]

### Critic 网络

- 输入：critic_state [B, 6, 4, 15] + global_feats [B, 5]
- CNN backbone：独立处理6通道状态（包含对手手牌）
- MLP：处理全局特征
- 前馈结构：Critic 输入已马尔可夫（全局可见），不使用 LSTM，而是在融合特征上直接预测 value
- 输出：价值估计 [B]

### 设计说明

- Actor 和 Critic 完全独立，不共享任何层
- Actor 只能看到自己的手牌和公开信息（符合实际游戏）
- Critic 可以看到所有玩家手牌（用于更准确的价值估计）
- 使用 Action Masking：在 Softmax 前将非法动作 logits 设为 -1e9

## 训练流程

### 混合对手策略

训练时，每个环境随机分配以下 4 种策略之一作为对手（概率大致为 greedy≈40%、ep100≈20%、ep50_before≈20%、random≈20%，具体数量根据 `NUM_ENVS` 取整，并将余数加到 greedy 或 random 上）。  
在实验中发现，idiot 的分布与其他对手差异过大，会导致训练不稳定，因此在训练对手池中剔除了 idiot，将其原本约 5% 的份额并入 greedy。

1. 贪心策略（greedy）：使用改进后的“菜鸟”贪心（尽量出长顺子、避免过早掏2/对2）
2. idiot 策略（仅用于评估曲线对比，不参与训练混合对手）：白痴贪心，总是出枚举到的第一个合法非 PASS 动作（通常是最小单牌）
3. 第100个epoch checkpoint（ep100）：固定使用第100个 episode 存下的策略；若该 checkpoint 尚不存在，则回退为 greedy
4. 当前进度前一个“50 区间”的 checkpoint（ep50_before）：从 150 epoch 开始启用，例如 150–199 epoch 使用 ep100，200–249 epoch 使用 ep150；若对应 checkpoint 尚不存在，则回退为 greedy
5. 随机选取一个历史checkpoint（仅从已存在且 episode≥100 的 checkpoint 中随机采样；若不存在，则回退为 greedy）

### 训练配置

- 训练用并行环境数（`NUM_ENVS`）：100（额外约 10% 的环境只用于 vs idiot 评估，不参与训练更新）
- Rollout steps（`ROLLOUT_STEPS`）：256
- PPO epochs：4
- Batch size：512
- 学习率：1e-5
- 最大训练episode：1000
- Checkpoint间隔：每50个episode

### 评估指标

从训练过程中自动统计多种评估指标：

1. vs 贪心策略（greedy）：统计与改进菜鸟对手对战的reward
2. vs idiot 策略：统计与白痴贪心对手对战的reward（注意：idiot 仅作为评估基线，不参与训练时的混合对手）
3. vs 第100个epoch checkpoint：统计与第100个episode模型对战的reward
4. vs 当前前一个“50 区间”的 checkpoint（从150开始）：统计与“当前进度前一个 50 区间 checkpoint”模型对战的reward


### 恢复训练

在 `train_ppo.py` 中设置：

```python
RESUME_FROM_CHECKPOINT = "checkpoints/latest.pt"
```
