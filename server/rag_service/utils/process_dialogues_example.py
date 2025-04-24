#!/usr/bin/env python3
"""
对话数据处理示例脚本
展示如何使用DialogueDataProcessor进行对话数据清洗、过滤和处理
"""

import sys
import os
import json
import logging
from typing import List, Dict, Any
from dialogue_data_processor import DialogueDataProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def main():
    """主程序"""
    
    # 示例配置
    config = {
        "enable_gpt4_rules": False,  # 不使用GPT-4规则检查（需要API密钥）
        "min_text_length": 10,
        "max_text_length": 2000,
        "bm25_threshold": 0.7,
        "vector_similarity_threshold": 0.8,
        "sensitive_words": ["不适当词汇1", "不适当词汇2"],  # 自定义敏感词
        "custom_stopwords": ["停用词1", "停用词2"]  # 自定义停用词
    }
    
    # 初始化处理器
    processor = DialogueDataProcessor(config)
    
    # 示例对话数据
    example_dialogues = [
        {
            "text": "你好，我想学习Python编程，请问从哪里开始比较好？",
            "metadata": {"source": "user_question", "scene": "education"}
        },
        {
            "text": "Python是一种非常流行的编程语言，初学者可以从安装Python环境开始，然后学习基本语法、数据类型和控制流程。",
            "metadata": {"source": "assistant_answer"}
        },
        {
            "text": "我已经安装了Python，能推荐一些学习资源吗？",
            "metadata": {"source": "user_question"}
        },
        {
            "text": "这是一段重复的文本 这是一段重复的文本 这是一段重复的文本 这是一段重复的文本",
            "metadata": {"source": "test_data"}
        },
        {
            "text": "你好，我是学生，想了解一下历史事件",
            "metadata": {"source": "user_question"}
        }
    ]
    
    # 处理对话数据
    logger.info("开始处理对话数据")
    processed_results = processor.process_dialogue_batch(example_dialogues)
    
    # 显示处理结果
    for i, result in enumerate(processed_results):
        logger.info(f"对话 {i+1}:")
        logger.info(f"  原始文本: {result['original_text'][:50]}...")
        logger.info(f"  是否有效: {result['is_valid']}")
        if not result['is_valid']:
            logger.info(f"  过滤原因: {result['filter_reason']}")
        else:
            logger.info(f"  处理后文本: {result['processed_text'][:50]}...")
            logger.info(f"  场景: {result['scene']}")
            logger.info(f"  难度: {result['difficulty']}")
        logger.info("")
    
    # 生成统计报告
    report = processor.generate_report(processed_results)
    logger.info("处理统计报告:")
    logger.info(f"  总对话数: {report['total_count']}")
    logger.info(f"  有效对话数: {report['valid_count']}")
    logger.info(f"  拒绝率: {report['rejection_rate']:.2%}")
    
    if report['rejection_reasons']:
        logger.info("  拒绝原因统计:")
        for reason, count in report['rejection_reasons'].items():
            logger.info(f"    {reason}: {count}")
    
    # 导出为JSON
    valid_dialogues = [d for d in processed_results if d['is_valid']]
    if valid_dialogues:
        output_path = "processed_dialogues.json"
        processor.export_to_json(valid_dialogues, output_path)
        logger.info(f"有效对话已导出到 {output_path}")

if __name__ == "__main__":
    main() 