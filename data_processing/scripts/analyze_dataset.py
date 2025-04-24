#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据分析脚本
分析处理后的对话数据并生成统计报告
"""

import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from data_processing.modules.dialogue_processor import Dialogue
from data_processing.utils.evaluation_utils import DataEvaluator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("analyze_dataset.log")
    ]
)
logger = logging.getLogger(__name__)

def load_dialogues(file_path: str) -> List[Dialogue]:
    """
    加载对话数据
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        对话列表
    """
    dialogues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            logger.error(f"数据格式错误，期望列表，得到: {type(data)}")
            return []
            
        for item in data:
            dialogue = Dialogue.from_dict(item)
            dialogues.append(dialogue)
            
        logger.info(f"已加载 {len(dialogues)} 个对话")
        
    except Exception as e:
        logger.error(f"加载对话数据出错: {e}")
        
    return dialogues

def generate_plots(stats: Dict[str, Any], output_dir: str):
    """
    生成统计图表
    
    Args:
        stats: 统计数据
        output_dir: 输出目录
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 轮次分布图
        plt.figure(figsize=(10, 6))
        plt.hist(stats.get("turn_counts", []), bins=20, alpha=0.7)
        plt.title("对话轮次分布")
        plt.xlabel("轮次数")
        plt.ylabel("对话数量")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "turn_distribution.png"))
        plt.close()
        
        # 长度分布图
        plt.figure(figsize=(10, 6))
        plt.hist(stats.get("text_lengths", []), bins=20, alpha=0.7)
        plt.title("文本长度分布")
        plt.xlabel("字符数")
        plt.ylabel("文本数量")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "length_distribution.png"))
        plt.close()
        
        # 说话者分布图
        speaker_counts = stats.get("speaker_counts", {})
        if speaker_counts:
            plt.figure(figsize=(12, 6))
            speakers = list(speaker_counts.keys())
            counts = list(speaker_counts.values())
            
            plt.bar(speakers, counts, alpha=0.7)
            plt.title("说话者分布")
            plt.xlabel("说话者")
            plt.ylabel("轮次数")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "speaker_distribution.png"))
            plt.close()
            
        # 质量得分分布图
        plt.figure(figsize=(10, 6))
        plt.hist(stats.get("quality_scores", []), bins=20, alpha=0.7)
        plt.title("对话质量得分分布")
        plt.xlabel("质量得分")
        plt.ylabel("对话数量")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "quality_distribution.png"))
        plt.close()
        
        logger.info(f"已生成统计图表到目录: {output_dir}")
        
    except Exception as e:
        logger.error(f"生成统计图表出错: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="对话数据分析")
    parser.add_argument("--data_file", type=str, required=True, help="处理后的数据文件路径")
    parser.add_argument("--output_dir", type=str, default="data/reports", help="输出目录")
    parser.add_argument("--generate_plots", action="store_true", help="是否生成统计图表")
    args = parser.parse_args()
    
    # 加载对话数据
    logger.info(f"从文件 {args.data_file} 加载对话数据")
    dialogues = load_dialogues(args.data_file)
    
    if not dialogues:
        logger.error("未加载到对话数据，退出分析")
        return
        
    # 创建评估器
    evaluator = DataEvaluator()
    
    # 进行统计分析
    logger.info("正在分析对话数据")
    dialogue_stats = evaluator.calculate_dialogue_stats(dialogues)
    
    # 计算质量得分
    quality_scores = evaluator.calculate_quality_scores(dialogues)
    
    # 检测异常数据
    outliers = evaluator.detect_outliers(dialogues)
    
    # 生成额外统计数据
    turn_counts = [len(dialogue.turns) for dialogue in dialogues]
    text_lengths = []
    speaker_counts = Counter()
    
    for dialogue in dialogues:
        for turn in dialogue.turns:
            text_lengths.append(len(turn.text))
            speaker_counts[turn.speaker] += 1
    
    # 合并统计结果
    stats = {
        "dialogue_stats": dialogue_stats,
        "quality_scores": quality_scores,
        "outliers": outliers,
        "turn_counts": turn_counts,
        "text_lengths": text_lengths,
        "speaker_counts": dict(speaker_counts),
        "quality_scores": [score["quality_score"] for score in quality_scores]
    }
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成报告文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(args.output_dir, f"dialogue_analysis_{timestamp}.json")
    
    # 保存报告
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"分析报告已保存到: {report_file}")
    except Exception as e:
        logger.error(f"保存分析报告出错: {e}")
    
    # 输出基本分析结果
    logger.info(f"对话数据分析结果:")
    logger.info(f"- 对话数量: {len(dialogues)}")
    logger.info(f"- 总轮次数: {sum(turn_counts)}")
    logger.info(f"- 平均轮次: {np.mean(turn_counts):.2f}")
    logger.info(f"- 平均文本长度: {np.mean(text_lengths):.2f}")
    logger.info(f"- 说话者数量: {len(speaker_counts)}")
    logger.info(f"- 异常对话数量: {len(outliers)}")
    logger.info(f"- 平均质量得分: {np.mean(stats['quality_scores']):.2f}")
    
    # 生成统计图表
    if args.generate_plots:
        plots_dir = os.path.join(args.output_dir, f"plots_{timestamp}")
        logger.info(f"正在生成统计图表到目录: {plots_dir}")
        generate_plots(stats, plots_dir)
    

if __name__ == "__main__":
    main() 