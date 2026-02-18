#!/usr/bin/env python3
"""
平滑训练曲线可视化脚本
读取 latest_history.npz 数据，对评估曲线进行平滑处理并绘制带置信带的趋势图。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from scipy import stats


def smooth_moving_average(data: np.ndarray, window_size: int = 20) -> np.ndarray:
    """
    使用移动平均平滑数据
    
    Args:
        data: 输入数据数组（可能包含 NaN）
        window_size: 移动平均窗口大小
    
    Returns:
        平滑后的数据数组
    """
    # 处理 NaN 值：使用有效值进行移动平均
    smoothed = np.full_like(data, np.nan)
    
    for i in range(len(data)):
        # 计算窗口范围
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        
        # 获取窗口内的有效数据（非 NaN）
        window_data = data[start:end]
        valid_data = window_data[~np.isnan(window_data)]
        
        if len(valid_data) > 0:
            smoothed[i] = np.mean(valid_data)
    
    return smoothed


def calculate_confidence_band(
    data: np.ndarray, 
    smoothed: np.ndarray,
    window_size: int = 20, 
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算置信带
    
    Args:
        data: 原始数据数组（可能包含 NaN）
        smoothed: 平滑后的数据数组
        window_size: 滚动窗口大小
        confidence: 置信水平（默认0.95，即95%）
    
    Returns:
        (上界, 下界) 数组
    """
    # 根据置信水平计算 z-score
    # 例如：0.95 -> 1.96, 0.90 -> 1.645, 0.68 -> 1.0
    alpha = 1 - confidence
    z_score = stats.norm.ppf(1 - alpha / 2)
    
    upper_bound = np.full_like(data, np.nan)
    lower_bound = np.full_like(data, np.nan)
    
    for i in range(len(data)):
        # 计算窗口范围
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        
        # 获取窗口内的有效数据（非 NaN）
        window_data = data[start:end]
        window_smoothed = smoothed[start:end]
        valid_mask = ~np.isnan(window_data)
        valid_data = window_data[valid_mask]
        valid_smoothed = window_smoothed[valid_mask]
        
        if len(valid_data) > 1:  # 至少需要2个点才能计算标准差
            # 计算残差（原始数据与平滑数据的差）的标准差
            residuals = valid_data - valid_smoothed
            std = np.std(residuals)
            # 如果标准差为0或太小，使用原始数据的标准差作为后备
            if std < 1e-6:
                std = np.std(valid_data)
            
            mean_val = smoothed[i]
            if not np.isnan(mean_val):
                upper_bound[i] = mean_val + z_score * std
                lower_bound[i] = mean_val - z_score * std
        elif len(valid_data) == 1:
            # 只有一个点，置信带就是该点本身
            upper_bound[i] = valid_data[0]
            lower_bound[i] = valid_data[0]
    
    return upper_bound, lower_bound


def plot_smoothed_training(
    history_path: str = "checkpoints/latest_history.npz",
    output_path: str = "checkpoints/training_progress_smoothed.png",
    window_size: int = 20,
    confidence: float = 0.95,
):
    """
    绘制平滑后的训练曲线，包含趋势线和置信带
    
    Args:
        history_path: 历史数据文件路径
        output_path: 输出图片路径
        window_size: 平滑窗口大小
        confidence: 置信水平（0-1之间，默认0.95即95%）
    """
    # 检查文件是否存在
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"历史数据文件不存在: {history_path}")
    
    # 加载数据
    history = np.load(history_path)
    episodes = history["episodes"]
    
    # 提取评估指标
    eval_rewards_greedy = history["eval_rewards_greedy"]
    eval_rewards_idiot = history["eval_rewards_idiot"]
    eval_rewards_ep100 = history["eval_rewards_ep100"]
    eval_rewards_ep50_before = history["eval_rewards_ep50_before"]
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 定义曲线配置
    curves = [
        {
            "data": eval_rewards_greedy,
            "label": "vs Greedy",
            "color": "blue",
        },
        {
            "data": eval_rewards_idiot,
            "label": "vs Idiot",
            "color": "orange",
        },
        {
            "data": eval_rewards_ep100,
            "label": "vs Ep100 Checkpoint",
            "color": "red",
        },
        {
            "data": eval_rewards_ep50_before,
            "label": "vs Ep50 Before Checkpoint",
            "color": "green",
        },
    ]
    
    # 绘制每条曲线
    for curve in curves:
        data = curve["data"]
        label = curve["label"]
        color = curve["color"]
        
        # 过滤掉 NaN 值，获取有效数据点
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            continue  # 如果没有有效数据，跳过这条曲线
        
        valid_episodes = episodes[valid_mask]
        valid_data = data[valid_mask]
        
        # 对有效数据进行平滑处理
        # 创建一个临时数组，只包含有效数据
        temp_data = np.full(len(episodes), np.nan)
        temp_data[valid_mask] = valid_data
        
        # 平滑处理
        smoothed = smooth_moving_average(temp_data, window_size=window_size)
        
        # 计算置信带
        upper_bound, lower_bound = calculate_confidence_band(
            temp_data, smoothed, window_size=window_size, confidence=confidence
        )
        
        # 只绘制有效数据点对应的平滑结果
        valid_smoothed = smoothed[valid_mask]
        valid_upper = upper_bound[valid_mask]
        valid_lower = lower_bound[valid_mask]
        
        # 绘制置信带（浅色填充，无轮廓）
        plt.fill_between(
            valid_episodes,
            valid_lower,
            valid_upper,
            alpha=0.25,
            color=color,
            edgecolor='none',  # 去掉轮廓
            label=None,  # 不显示在图例中
        )
        
        # 绘制平滑后的趋势线
        plt.plot(
            valid_episodes,
            valid_smoothed,
            linewidth=2.0,
            color=color,
            label=label,
        )
    
    # 设置图形属性
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Mean Reward", fontsize=12)
    confidence_pct = int(confidence * 100)
    plt.title(f"Training Progress - Evaluation Metrics (Smoothed, {confidence_pct}% CI)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存图片
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"平滑训练曲线已保存到: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="绘制平滑后的训练曲线")
    parser.add_argument(
        "--input",
        type=str,
        default="checkpoints/latest_history.npz",
        help="输入历史数据文件路径（默认: checkpoints/latest_history.npz）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/training_progress_smoothed.png",
        help="输出图片路径（默认: checkpoints/training_progress_smoothed.png）",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="平滑窗口大小（默认: 20）",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.68,
        help="置信水平（0-1之间，默认: 0.95 即95%%）。常用值：0.68 (1σ), 0.90, 0.95 (2σ), 0.99",
    )
    
    args = parser.parse_args()
    
    # 验证置信水平范围
    if not 0 < args.confidence < 1:
        raise ValueError(f"置信水平必须在 0 和 1 之间，当前值: {args.confidence}")
    
    plot_smoothed_training(
        history_path=args.input,
        output_path=args.output,
        window_size=args.window,
        confidence=args.confidence,
    )
