#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据向量化脚本
将处理后的对话数据向量化并保存
"""

import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any
from pathlib import Path
import time
import numpy as np

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from data_processing.modules.dialogue_processor import Dialogue
from data_processing.utils.vector_utils import VectorProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("vectorize_dataset.log")
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

def prepare_text_for_vectorization(dialogues: List[Dialogue], mode: str = "full") -> List[Dict[str, Any]]:
    """
    准备向量化的文本数据
    
    Args:
        dialogues: 对话列表
        mode: 向量化模式 (full: 整个对话, turn: 每轮分别向量化)
        
    Returns:
        文本数据列表
    """
    text_data = []
    
    for dialogue in dialogues:
        dialogue_id = dialogue.id
        
        if mode == "full":
            # 整个对话向量化
            full_text = dialogue.get_text(join_char="\n")
            
            if full_text.strip():
                text_data.append({
                    "id": dialogue_id,
                    "text": full_text,
                    "type": "dialogue",
                    "metadata": dialogue.metadata
                })
        else:
            # 每轮分别向量化
            for i, turn in enumerate(dialogue.turns):
                text = turn.text
                
                if text.strip():
                    text_data.append({
                        "id": f"{dialogue_id}_turn_{i}",
                        "text": text,
                        "type": "turn",
                        "speaker": turn.speaker,
                        "turn_index": i,
                        "dialogue_id": dialogue_id,
                        "metadata": turn.metadata
                    })
    
    return text_data

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="对话数据向量化")
    parser.add_argument("--data_file", type=str, required=True, help="处理后的数据文件路径")
    parser.add_argument("--output_dir", type=str, default="data/vectors", help="输出目录")
    parser.add_argument("--config", type=str, default="data_processing/config/dialogue_config.json", help="配置文件路径")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "turn"], help="向量化模式")
    parser.add_argument("--model_name", type=str, default="paraphrase-multilingual-MiniLM-L12-v2", help="向量模型名称")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小")
    parser.add_argument("--cache_name", type=str, default="", help="缓存名称，默认为时间戳")
    parser.add_argument("--build_index", action="store_true", help="是否构建FAISS索引")
    parser.add_argument("--index_type", type=str, default="Flat", choices=["Flat", "IVF"], help="索引类型")
    args = parser.parse_args()
    
    # 加载对话数据
    logger.info(f"从文件 {args.data_file} 加载对话数据")
    dialogues = load_dialogues(args.data_file)
    
    if not dialogues:
        logger.error("未加载到对话数据，退出向量化")
        return
        
    # 准备配置
    config = {
        "model_name": args.model_name,
        "cache_dir": args.output_dir,
        "enable_cache": True,
        "batch_size": args.batch_size
    }
    
    # 加载额外配置
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                extra_config = json.load(f)
                # 更新向量相关配置
                if "retrieval" in extra_config and "vector" in extra_config["retrieval"]:
                    config.update(extra_config["retrieval"]["vector"])
        except Exception as e:
            logger.error(f"加载配置文件出错: {e}")
    
    # 创建向量处理器
    vector_processor = VectorProcessor(config)
    
    # 确保模型加载成功
    if vector_processor.model is None:
        logger.error("向量模型加载失败，退出向量化")
        return
    
    # 准备向量化的文本数据
    logger.info(f"准备文本数据，模式: {args.mode}")
    text_data = prepare_text_for_vectorization(dialogues, args.mode)
    
    if not text_data:
        logger.error("未获取到文本数据，退出向量化")
        return
        
    logger.info(f"准备向量化 {len(text_data)} 条文本")
    
    # 提取文本和元数据
    texts = [item["text"] for item in text_data]
    metadata = [
        {k: v for k, v in item.items() if k != "text"}
        for item in text_data
    ]
    
    # 执行向量化
    logger.info("开始向量化")
    start_time = time.time()
    vectors = vector_processor.encode_text(texts, batch_size=args.batch_size)
    elapsed = time.time() - start_time
    logger.info(f"向量化完成，耗时 {elapsed:.2f} 秒")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 缓存名称
    cache_name = args.cache_name
    if not cache_name:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cache_name = f"dialogue_vectors_{args.mode}_{timestamp}"
    
    # 缓存向量
    logger.info(f"缓存向量，名称: {cache_name}")
    vector_processor.cache_vectors(vectors, metadata, cache_name)
    
    # 构建FAISS索引
    if args.build_index:
        logger.info(f"构建FAISS索引，类型: {args.index_type}")
        index = vector_processor.build_faiss_index(vectors, args.index_type)
        
        if index is not None:
            # 保存索引
            index_path = os.path.join(args.output_dir, f"{cache_name}.index")
            vector_processor.save_faiss_index(index, index_path)
            logger.info(f"索引已保存到: {index_path}")
    
    # 输出向量信息
    logger.info(f"向量化结果:")
    logger.info(f"- 向量数量: {len(vectors)}")
    logger.info(f"- 向量维度: {vectors.shape[1]}")
    logger.info(f"- 向量类型: {vectors.dtype}")
    logger.info(f"- 平均向量模长: {np.mean([np.linalg.norm(v) for v in vectors]):.4f}")
    logger.info(f"- 向量已保存到: {os.path.join(args.output_dir, f'{cache_name}.npy')}")
    logger.info(f"- 元数据已保存到: {os.path.join(args.output_dir, f'{cache_name}_metadata.json')}")
    

if __name__ == "__main__":
    main() 