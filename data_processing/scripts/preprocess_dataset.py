#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据预处理脚本
用于处理dataset目录下的原始对话数据
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

from data_processing.modules.dialogue_processor import DialogueProcessor, Dialogue, DialogueTurn
from data_processing.utils.text_utils import TextProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("preprocess_dataset.log")
    ]
)
logger = logging.getLogger(__name__)

def load_data_from_directory(data_dir: str, file_pattern: str = "*.json") -> List[Dict[str, Any]]:
    """
    从目录中加载数据文件
    
    Args:
        data_dir: 数据目录
        file_pattern: 文件模式
        
    Returns:
        加载的数据列表
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return []
        
    all_data = []
    
    for file_path in data_path.glob(file_pattern):
        try:
            logger.info(f"正在加载文件: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                all_data.extend(data)
                logger.info(f"已加载 {len(data)} 条记录")
            elif isinstance(data, dict):
                all_data.append(data)
                logger.info(f"已加载 1 条记录")
            else:
                logger.warning(f"未知数据格式: {type(data)}")
                
        except Exception as e:
            logger.error(f"加载文件 {file_path} 出错: {e}")
            
    return all_data

def convert_to_dialogues(data: List[Dict[str, Any]]) -> List[Dialogue]:
    """
    将原始数据转换为对话对象列表
    
    Args:
        data: 原始数据
        
    Returns:
        对话对象列表
    """
    dialogues = []
    
    for item in data:
        try:
            # 检查数据格式
            if "id" not in item or "conversations" not in item:
                logger.warning(f"数据格式不匹配，缺少必要字段: {item.keys()}")
                continue
                
            # 提取ID和对话
            dialogue_id = item["id"]
            conversations = item["conversations"]
            
            # 转换对话轮次
            turns = []
            for conv in conversations:
                if "from" not in conv or "value" not in conv:
                    continue
                    
                turn = DialogueTurn(
                    speaker=conv["from"],
                    text=conv["value"],
                    timestamp=conv.get("timestamp"),
                    metadata=conv.get("metadata", {})
                )
                turns.append(turn)
                
            # 创建对话对象
            metadata = {k: v for k, v in item.items() if k not in ["id", "conversations"]}
            dialogue = Dialogue(
                id=dialogue_id,
                turns=turns,
                metadata=metadata
            )
            
            dialogues.append(dialogue)
            
        except Exception as e:
            logger.error(f"转换对话出错: {e}")
            
    return dialogues

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="对话数据预处理")
    parser.add_argument("--data_dir", type=str, default="dataset", help="原始数据目录")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="输出目录")
    parser.add_argument("--config", type=str, default="data_processing/config/dialogue_config.json", help="配置文件路径")
    parser.add_argument("--file_pattern", type=str, default="*.json", help="文件匹配模式")
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载原始数据
    logger.info(f"从目录 {args.data_dir} 加载数据")
    raw_data = load_data_from_directory(args.data_dir, args.file_pattern)
    logger.info(f"共加载 {len(raw_data)} 条原始记录")
    
    if not raw_data:
        logger.error("未找到数据，退出处理")
        return
        
    # 转换为对话对象
    logger.info("转换数据格式")
    dialogues = convert_to_dialogues(raw_data)
    logger.info(f"转换得到 {len(dialogues)} 个对话")
    
    # 初始化处理器
    logger.info(f"使用配置文件 {args.config} 初始化处理器")
    processor = DialogueProcessor(args.config)
    
    # 处理对话
    logger.info("开始处理对话")
    start_time = time.time()
    processed_dialogues = processor.process_batch(dialogues)
    elapsed = time.time() - start_time
    logger.info(f"处理完成，耗时 {elapsed:.2f} 秒，得到 {len(processed_dialogues)} 个处理后的对话")
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"processed_dialogues_{timestamp}.json")
    
    logger.info(f"保存处理结果到 {output_path}")
    output_file = processor.save_processed_dialogues(processed_dialogues, output_path)
    
    logger.info(f"处理完成，结果已保存到: {output_file}")
    
    # 输出基本统计信息
    total_turns = sum(len(dialogue.turns) for dialogue in processed_dialogues)
    avg_turns = total_turns / len(processed_dialogues) if processed_dialogues else 0
    
    logger.info(f"处理后的对话统计:")
    logger.info(f"- 对话数量: {len(processed_dialogues)}")
    logger.info(f"- 总轮次数: {total_turns}")
    logger.info(f"- 平均轮次: {avg_turns:.2f}")
    

if __name__ == "__main__":
    main() 