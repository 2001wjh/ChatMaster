#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据增强脚本
用于对处理后的对话数据进行增强
"""

import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any
from pathlib import Path
import time

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from data_processing.modules.dialogue_processor import Dialogue
from data_processing.utils.augmentation_utils import DataAugmentor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("augment_dataset.log")
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

def save_dialogues(dialogues: List[Dialogue], output_path: str) -> bool:
    """
    保存对话数据
    
    Args:
        dialogues: 对话列表
        output_path: 输出文件路径
        
    Returns:
        是否保存成功
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # 转换为字典列表
        data = [dialogue.to_dict() for dialogue in dialogues]
        
        # 保存数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"已将 {len(dialogues)} 个对话保存到: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"保存对话数据出错: {e}")
        return False

def filter_dialogues_by_quality(dialogues: List[Dialogue], min_quality: float = 0.6) -> List[Dialogue]:
    """
    根据质量分数过滤对话
    
    Args:
        dialogues: 对话列表
        min_quality: 最低质量分数
        
    Returns:
        过滤后的对话列表
    """
    filtered = []
    
    for dialogue in dialogues:
        quality_score = dialogue.metadata.get("quality_score", 0.0)
        
        if quality_score >= min_quality:
            filtered.append(dialogue)
            
    logger.info(f"质量过滤: {len(dialogues)} -> {len(filtered)} 个对话")
    return filtered

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="对话数据增强")
    parser.add_argument("--data_file", type=str, required=True, help="处理后的数据文件路径")
    parser.add_argument("--output_dir", type=str, default="data/augmented", help="输出目录")
    parser.add_argument("--config", type=str, default="data_processing/config/dialogue_config.json", help="配置文件路径")
    parser.add_argument("--methods", type=str, default="synonym,swap,delete", help="增强方法，逗号分隔")
    parser.add_argument("--count_per_method", type=int, default=1, help="每种方法生成的样本数量")
    parser.add_argument("--min_quality", type=float, default=0.6, help="用于增强的最低质量分数")
    parser.add_argument("--max_dialogues", type=int, default=0, help="最多处理的对话数量，0表示不限制")
    args = parser.parse_args()
    
    # 加载对话数据
    logger.info(f"从文件 {args.data_file} 加载对话数据")
    dialogues = load_dialogues(args.data_file)
    
    if not dialogues:
        logger.error("未加载到对话数据，退出增强")
        return
        
    # 根据质量过滤对话
    if args.min_quality > 0:
        dialogues = filter_dialogues_by_quality(dialogues, args.min_quality)
        
    # 限制对话数量
    if args.max_dialogues > 0 and len(dialogues) > args.max_dialogues:
        logger.info(f"限制对话数量: {len(dialogues)} -> {args.max_dialogues}")
        dialogues = dialogues[:args.max_dialogues]
    
    # 解析增强方法
    methods = [method.strip() for method in args.methods.split(",")]
    logger.info(f"使用增强方法: {methods}")
    
    # 加载配置
    config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件出错: {e}")
    
    # 创建增强器
    augmentor = DataAugmentor(config)
    
    # 进行数据增强
    logger.info(f"开始对 {len(dialogues)} 个对话进行增强")
    start_time = time.time()
    
    all_augmented = []
    for i, dialogue in enumerate(dialogues):
        if (i+1) % 10 == 0:
            logger.info(f"已处理 {i+1}/{len(dialogues)} 个对话")
            
        # 对单个对话进行增强
        augmented = augmentor.augment_dialogue(
            dialogue.to_dict(),
            methods=methods,
            n_per_method=args.count_per_method
        )
        
        # 转换为对话对象
        for aug_dict in augmented:
            aug_dialogue = Dialogue.from_dict(aug_dict)
            all_augmented.append(aug_dialogue)
    
    elapsed = time.time() - start_time
    logger.info(f"增强完成，耗时 {elapsed:.2f} 秒，得到 {len(all_augmented)} 个增强对话")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成输出文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"augmented_dialogues_{timestamp}.json")
    
    # 保存增强后的对话
    save_success = save_dialogues(all_augmented, output_file)
    
    if save_success:
        logger.info(f"增强后的对话已保存到: {output_file}")
        
        # 输出基本统计信息
        original_turns = sum(len(dialogue.turns) for dialogue in dialogues)
        augmented_turns = sum(len(dialogue.turns) for dialogue in all_augmented)
        
        logger.info(f"数据增强统计:")
        logger.info(f"- 原始对话数量: {len(dialogues)}")
        logger.info(f"- 原始轮次数量: {original_turns}")
        logger.info(f"- 增强对话数量: {len(all_augmented)}")
        logger.info(f"- 增强轮次数量: {augmented_turns}")
        logger.info(f"- 数据量增加比例: {len(all_augmented) / max(1, len(dialogues)):.2f}倍")
    

if __name__ == "__main__":
    main() 