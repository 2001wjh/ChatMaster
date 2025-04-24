import unittest
import json
import logging
from typing import Dict, List, Any

from server.rag_service.utils.text_processing import TextProcessingService

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TestTextProcessing(unittest.TestCase):
    """测试文本处理服务的口语理解优化功能"""
    
    @classmethod
    def setUpClass(cls):
        """初始化测试环境"""
        # 创建配置
        config = {
            "enable_advanced_features": True,
            "enable_oral_correction": True,
            "enable_coreference": True,
            "enable_intent_recognition": True,
            "enable_named_entity_recognition": True,
            "enable_sentiment_analysis": True,
            "oral_correction_model": "t5-base",
            "coreference_model": "bert-base-uncased",
            "intent_model": "roberta-base",
            "ner_model": "bert-base-cased"
        }
        
        # 初始化服务
        cls.service = TextProcessingService(config)
        
        # 测试数据
        cls.oral_test_cases = [
            {
                "input": "i wanna go to the mall coz it's kinda cool",
                "expected": "I want to go to the mall because it is kind of cool"
            },
            {
                "input": "u gotta help me with this",
                "expected": "You got to help me with this"
            },
            {
                "input": "lemme see what's goin on",
                "expected": "Let me see what is going on"
            }
        ]
        
        cls.intent_test_cases = [
            {
                "input": "What is the capital of France?",
                "expected_category": "learning_question"
            },
            {
                "input": "Hey, how's it going?",
                "expected_category": "casual_chat"
            },
            {
                "input": "Can you find me information about AI?",
                "expected_category": "command"
            },
            {
                "input": "During the meeting, can we discuss the project timeline?",
                "expected_category": "scene_conversation"
            }
        ]
        
        cls.entity_test_cases = [
            {
                "input": "Bill Gates founded Microsoft in 1975.",
                "expected_entities": ["Bill Gates", "Microsoft", "1975"]
            },
            {
                "input": "I have a meeting at 10:30 tomorrow at Google Headquarters.",
                "expected_entities": ["10:30", "tomorrow", "Google Headquarters"]
            },
            {
                "input": "Please contact John at john@example.com or visit our website www.example.com.",
                "expected_entities": ["John", "john@example.com", "www.example.com"]
            }
        ]
        
        cls.coreference_test_cases = [
            {
                "input": "John said he will attend the meeting.",
                "expected": "John said John will attend the meeting.",
                "history": []
            },
            {
                "input": "What is its capital?",
                "expected": "What is France's capital?",
                "history": [
                    {"user": "Tell me about France"},
                    {"system": "France is a country in Western Europe."}
                ]
            }
        ]
    
    def test_oral_correction(self):
        """测试口语化问题处理功能"""
        for case in self.oral_test_cases:
            processed = self.service._correct_oral_expression(case["input"])
            self.assertIsNotNone(processed)
            print(f"\n口语修正测试:")
            print(f"原始: {case['input']}")
            print(f"修正: {processed}")
            print(f"期望: {case['expected']}")
            
            # 计算BLEU分数
            bleu = self.service._calculate_bleu(case["expected"], processed)
            print(f"BLEU分数: {bleu:.4f}")
            
            # 验证BLEU分数符合要求
            self.assertGreaterEqual(bleu, 0.5, "BLEU分数应该至少达到0.5")
    
    def test_intent_recognition(self):
        """测试意图识别功能"""
        for case in self.intent_test_cases:
            result = self.service.process_query(case["input"])
            intent = result.get("intent", {})
            
            print(f"\n意图识别测试:")
            print(f"输入: {case['input']}")
            print(f"识别意图: {intent.get('category')}")
            print(f"置信度: {intent.get('confidence', 0):.4f}")
            print(f"期望意图: {case['expected_category']}")
            
            if isinstance(intent, dict) and 'category' in intent:
                # 如果使用基于规则的意图识别
                self.assertIn(intent['category'], self.service.intent_categories.values())
                if case['expected_category'] in self.service.intent_categories.values():
                    self.assertEqual(intent['category'], case['expected_category'])
    
    def test_entity_recognition(self):
        """测试命名实体识别功能"""
        for case in self.entity_test_cases:
            result = self.service.process_query(case["input"])
            entities = result.get("entities", [])
            
            print(f"\n实体识别测试:")
            print(f"输入: {case['input']}")
            print(f"识别实体: {[e['text'] for e in entities]}")
            print(f"期望实体: {case['expected_entities']}")
            
            # 检查是否识别出了一些实体
            self.assertTrue(len(entities) > 0, "应该识别出至少一个实体")
            
            # 检查是否识别出了期望的实体
            found_entities = [e["text"].lower() for e in entities]
            for expected in case["expected_entities"]:
                self.assertTrue(
                    any(expected.lower() in found.lower() for found in found_entities),
                    f"未能识别出实体 {expected}"
                )
    
    def test_coreference_resolution(self):
        """测试指代消歧功能"""
        for i, case in enumerate(self.coreference_test_cases):
            resolved = self.service._resolve_coreference(case["input"], case["history"])
            
            print(f"\n指代消歧测试 {i+1}:")
            print(f"输入: {case['input']}")
            print(f"历史: {case['history']}")
            print(f"消歧结果: {resolved}")
            print(f"期望结果: {case['expected']}")
            
            # 验证消歧结果不为空
            self.assertIsNotNone(resolved)
            self.assertTrue(len(resolved) > 0)
    
    def test_full_processing(self):
        """测试完整的查询处理流程"""
        # 有指代消歧的复杂查询
        history = [
            {"user": "Tell me about Albert Einstein"},
            {"system": "Albert Einstein was a theoretical physicist born in Germany in 1879."}
        ]
        
        query = "What were his major contributions to science?"
        result = self.service.process_query(query, history)
        
        print("\n完整处理流程测试:")
        print(f"原始查询: {query}")
        print(f"处理后: {result['processed_text']}")
        print(f"指代消歧: {result['coreference_resolved']}")
        print(f"意图: {result['intent']['category'] if 'intent' in result and isinstance(result['intent'], dict) else None}")
        print(f"实体: {[e['text'] for e in result['entities']]}")
        print(f"处理步骤: {json.dumps(result.get('processing_steps', []), indent=2)}")
        
        # 验证结果包含关键字段
        self.assertIn("processed_text", result)
        self.assertIn("coreference_resolved", result)
        self.assertIn("intent", result)
        self.assertIn("entities", result)

if __name__ == "__main__":
    unittest.main() 