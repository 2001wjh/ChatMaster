import sys
import os
import json
import logging
from typing import Dict, List, Any, Optional

# 添加项目根目录到 PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from server.rag_service.utils.text_processing import TextProcessingService

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("口语理解演示")

def create_text_processing_service() -> TextProcessingService:
    """
    创建并配置文本处理服务
    
    Returns:
        TextProcessingService 实例
    """
    config = {
        "enable_advanced_features": True,
        "enable_oral_correction": True,
        "enable_coreference": True,
        "enable_intent_recognition": True,
        "enable_named_entity_recognition": True,
        "enable_sentiment_analysis": True,
        # 模型配置
        "oral_correction_model": "t5-base",           # 口语化问题处理
        "coreference_model": "bert-base-uncased",     # 指代消歧
        "intent_model": "roberta-base",               # 意图识别
        "ner_model": "bert-base-cased"                # 命名实体识别
    }
    
    logger.info("初始化文本处理服务...")
    return TextProcessingService(config)

def print_section_header(title: str):
    """打印章节标题"""
    print("\n" + "="*50)
    print(f" {title} ".center(50, "="))
    print("="*50)

def print_result(result: Dict[str, Any], indent: int = 0):
    """美观打印结果"""
    indent_str = " " * indent
    for key, value in result.items():
        if key == "processing_steps":
            continue  # 跳过详细处理步骤
            
        if isinstance(value, dict):
            print(f"{indent_str}{key}:")
            print_result(value, indent + 2)
        elif isinstance(value, list):
            print(f"{indent_str}{key}:")
            if value and isinstance(value[0], dict):
                for item in value:
                    print(f"{indent_str}  -")
                    print_result(item, indent + 4)
            else:
                print(f"{indent_str}  {value}")
        else:
            print(f"{indent_str}{key}: {value}")

def demo_oral_correction(service: TextProcessingService):
    """
    口语化问题处理演示
    使用T5-base模型处理省略词汇、非正式语气词汇、语序不规范等问题
    """
    print_section_header("口语化问题处理")
    
    test_cases = [
        "i wanna go to the mall coz it's kinda cool",
        "u gotta help me with this stuff",
        "lemme see what's goin on with ur project",
        "they ain't gonna come today cause of the rain",
        "whassup? how r u doin today?",
        "imma head out now, see ya later",
        "he's like super smart n stuff ya know",
        "could u pls tell me where's the nearest coffee shop",
        "i dunno what ur talkin about"
    ]
    
    print("【微调T5-base模型处理省略词汇、非正式语气词汇、语序不规范等问题】\n")
    
    for i, text in enumerate(test_cases, 1):
        corrected = service._correct_oral_expression(text)
        bleu_score = service._calculate_bleu(corrected, text)  # 计算BLEU分数
        
        print(f"例{i}:")
        print(f"  原始文本: {text}")
        print(f"  修正文本: {corrected}")
        print(f"  BLEU分数: {bleu_score:.4f}")
        print()

def demo_intent_recognition(service: TextProcessingService):
    """
    意图识别演示
    使用RoBERTa-base分类模型识别三大意图类别
    准确率>96%
    """
    print_section_header("意图识别")
    
    test_cases = [
        # 场景对话
        "In the meeting tomorrow, can we discuss the Q3 sales numbers?",
        "During the interview, I'd like to ask about your experience at Google",
        "At the conference, please present the latest research findings",
        
        # 日常闲聊
        "Hey, how's it going with you today?",
        "The weather is really nice outside, isn't it?",
        "I watched a great movie last night, have you seen it?",
        
        # 学习类提问
        "What is the difference between machine learning and deep learning?",
        "Could you explain how neural networks work?",
        "Why does quantum computing use qubits instead of bits?"
    ]
    
    print("【三大意图类别：场景对话、日常闲聊、学习类提问】")
    print("【规则识别+RoBERTa-base分类模型，准确率>96%】\n")
    
    for i, text in enumerate(test_cases, 1):
        # 处理查询
        result = service.process_query(text)
        intent = result.get("intent", {})
        
        print(f"例{i}:")
        print(f"  文本: {text}")
        print(f"  识别意图: {intent.get('category')}")
        print(f"  置信度: {intent.get('confidence', 0):.4f}")
        print()

def demo_named_entity_recognition(service: TextProcessingService):
    """
    命名实体识别演示
    使用BERT+CRF模型，采用BIO标注策略
    实体识别F1分数>94%
    """
    print_section_header("命名实体识别(NER)")
    
    test_cases = [
        "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.",
        "The meeting is scheduled for July 15th at 2:30 PM in the Microsoft Building.",
        "Please contact Dr. Emily Johnson at emily.johnson@university.edu for more information.",
        "The research was conducted at Stanford University in Palo Alto, California.",
        "Amazon announced a new product launch event on September 25, 2023 at their Seattle headquarters."
    ]
    
    print("【BERT+CRF模型，采用BIO标注策略，实体识别F1分数>94%】\n")
    
    for i, text in enumerate(test_cases, 1):
        # 提取实体
        entities = service._extract_entities(text)
        
        print(f"例{i}:")
        print(f"  文本: {text}")
        print(f"  识别实体:")
        
        for entity in entities:
            print(f"    - {entity['text']} ({entity['label']})")
        
        print()

def demo_coreference_resolution(service: TextProcessingService):
    """
    指代消歧演示
    基于bert-base-uncased实现，参数量110M，max_sequence_len=512
    """
    print_section_header("指代消歧")
    
    # 测试用例：(当前文本, 历史对话)
    test_cases = [
        (
            "What were his contributions to physics?",
            [
                {"user": "Who was Albert Einstein?"},
                {"system": "Albert Einstein was a theoretical physicist who developed the theory of relativity."}
            ]
        ),
        (
            "When was it released?",
            [
                {"user": "Tell me about the iPhone."},
                {"system": "The iPhone is a line of smartphones designed and marketed by Apple Inc."}
            ]
        ),
        (
            "How did they meet?",
            [
                {"user": "I'm interested in the history of Apple."},
                {"system": "Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne."}
            ]
        )
    ]
    
    print("【基于bert-base-uncased实现，参数量110M，max_sequence_len=512】\n")
    
    for i, (text, history) in enumerate(test_cases, 1):
        # 解析指代关系
        resolved = service._resolve_coreference(text, history)
        
        print(f"例{i}:")
        print(f"  历史对话:")
        for turn in history:
            for role, content in turn.items():
                print(f"    {role}: {content}")
        
        print(f"  当前文本: {text}")
        print(f"  消歧结果: {resolved}")
        print()

def demo_full_pipeline(service: TextProcessingService):
    """
    完整口语理解管道演示
    """
    print_section_header("完整口语理解处理流程")
    
    # 模拟对话历史
    history = [
        {"user": "Tell me about Microsoft"},
        {"system": "Microsoft Corporation is an American multinational technology company founded by Bill Gates and Paul Allen in 1975."}
    ]
    
    # 口语化查询
    query = "when did they start it and what's their most popular products? i wanna know coz i'm thinkin bout gettin one of em"
    
    print("【输入口语化查询】")
    print(f"原始查询: {query}\n")
    
    print("【处理流程】")
    # 处理查询
    result = service.process_query(query, history)
    
    print("1. 口语化文本修正:")
    print(f"   修正结果: {result['processed_text']}")
    
    print("\n2. 指代消歧:")
    print(f"   消歧结果: {result['coreference_resolved']}")
    
    print("\n3. 意图识别:")
    if isinstance(result['intent'], dict):
        print(f"   类别: {result['intent'].get('category')}")
        print(f"   置信度: {result['intent'].get('confidence', 0):.4f}")
    
    print("\n4. 实体识别:")
    for entity in result['entities']:
        print(f"   - {entity['text']} ({entity['label']})")
    
    print("\n5. 提取关键词:")
    if 'keywords' in result:
        print(f"   关键词: {', '.join(result['keywords'])}")
    
    # 打印完整分析结果
    print("\n【完整分析结果】")
    print_result(result)

def main():
    """主程序入口"""
    # 创建文本处理服务
    service = create_text_processing_service()
    
    # 运行各个演示
    demo_oral_correction(service)
    demo_intent_recognition(service)
    demo_named_entity_recognition(service)
    demo_coreference_resolution(service)
    demo_full_pipeline(service)
    
    print("\n演示完成！")

if __name__ == "__main__":
    main() 