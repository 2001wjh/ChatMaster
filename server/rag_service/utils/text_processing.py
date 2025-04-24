"""
文本处理工具模块
实现口语理解优化、指代消歧、意图识别等功能
"""

import re
import logging
from typing import Dict, Any, List, Optional
import json
import numpy as np
import torch
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    BertModel, 
    BertTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer
)

logger = logging.getLogger(__name__)

class TextProcessingService:
    """文本处理服务类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化文本处理服务
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.enable_advanced_features = config.get("enable_advanced_features", True)
        
        # 初始化模型
        if self.enable_advanced_features:
            try:
                # 口语化处理模型
                self.oral_correction_model = None
                self.oral_correction_tokenizer = None
                if config.get("enable_oral_correction", False):
                    model_name = config.get("oral_correction_model", "t5-base")
                    self.oral_correction_model = T5ForConditionalGeneration.from_pretrained(model_name)
                    self.oral_correction_tokenizer = T5Tokenizer.from_pretrained(model_name)
                
                # 指代消歧模型
                self.coreference_model = None
                self.coreference_tokenizer = None
                if config.get("enable_coreference", False):
                    model_name = config.get("coreference_model", "bert-base-uncased")
                    self.coreference_model = BertModel.from_pretrained(model_name)
                    self.coreference_tokenizer = BertTokenizer.from_pretrained(model_name)
                
                # 意图识别模型
                self.intent_model = None
                self.intent_tokenizer = None
                if config.get("enable_intent_recognition", False):
                    model_name = config.get("intent_model", "roberta-base")
                    self.intent_model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)
                    self.intent_tokenizer = RobertaTokenizer.from_pretrained(model_name)
                    
                # 意图类别
                self.intent_categories = {
                    0: "scene_conversation",  # 场景对话
                    1: "casual_chat",         # 日常闲聊
                    2: "learning_question"    # 学习类提问
                }
                
                logger.info("文本处理服务初始化完成")
            except Exception as e:
                logger.error(f"初始化文本处理模型失败: {str(e)}")
                # 降级处理，禁用高级功能
                self.enable_advanced_features = False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        处理用户查询
        
        Args:
            query: 用户查询文本
            
        Returns:
            处理结果，包含修正文本、意图、实体等信息
        """
        try:
            result = {
                "original_text": query,
                "processed_text": query,
                "intent": None,
                "entities": [],
                "is_question": self._is_question(query)
            }
            
            # 如果未启用高级功能，仅进行基本处理
            if not self.enable_advanced_features:
                # 简单规则处理
                result["processed_text"] = self._basic_text_processing(query)
                result["intent"] = self._rule_based_intent(query)
                return result
            
            # 口语化文本修正
            if self.oral_correction_model is not None:
                result["processed_text"] = self._correct_oral_expression(query)
            
            # 指代消歧
            if self.coreference_model is not None:
                # 这里应该使用上下文信息进行指代消歧
                # 简化版本直接返回处理文本
                pass
            
            # 意图识别
            if self.intent_model is not None:
                intent_id, intent_confidence = self._recognize_intent(query)
                result["intent"] = {
                    "category": self.intent_categories.get(intent_id, "unknown"),
                    "confidence": intent_confidence
                }
            else:
                # 规则识别
                result["intent"] = self._rule_based_intent(query)
            
            # 实体识别 (简化实现)
            result["entities"] = self._extract_entities(query)
            
            return result
            
        except Exception as e:
            logger.error(f"处理查询失败: {str(e)}")
            # 发生错误时返回原始文本
            return {
                "original_text": query,
                "processed_text": query,
                "intent": self._rule_based_intent(query),
                "entities": [],
                "is_question": self._is_question(query)
            }
    
    def _basic_text_processing(self, text: str) -> str:
        """
        基本文本处理
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的文本
        """
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 修正常见缩写
        text = re.sub(r'\bi\'m\b', 'I am', text, flags=re.IGNORECASE)
        text = re.sub(r'\bdon\'t\b', 'do not', text, flags=re.IGNORECASE)
        text = re.sub(r'\bcan\'t\b', 'cannot', text, flags=re.IGNORECASE)
        text = re.sub(r'\bi\'ll\b', 'I will', text, flags=re.IGNORECASE)
        text = re.sub(r'\bi\'ve\b', 'I have', text, flags=re.IGNORECASE)
        
        return text
    
    def _correct_oral_expression(self, text: str) -> str:
        """
        修正口语表达
        
        Args:
            text: 输入文本
            
        Returns:
            修正后的文本
        """
        try:
            if self.oral_correction_model is None or self.oral_correction_tokenizer is None:
                return text
            
            input_text = f"correct oral: {text}"
            inputs = self.oral_correction_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.oral_correction_model.generate(
                    inputs.input_ids,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            
            corrected_text = self.oral_correction_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 如果修正后文本为空或过短，则返回原文本
            if not corrected_text or len(corrected_text) < len(text) / 2:
                return text
                
            return corrected_text
            
        except Exception as e:
            logger.error(f"修正口语表达失败: {str(e)}")
            return text
    
    def _recognize_intent(self, text: str) -> (int, float):
        """
        识别用户意图
        
        Args:
            text: 输入文本
            
        Returns:
            意图ID和置信度
        """
        try:
            if self.intent_model is None or self.intent_tokenizer is None:
                # 回退到规则识别
                intent = self._rule_based_intent(text)
                category_id = list(self.intent_categories.keys())[list(self.intent_categories.values()).index(intent)]
                return category_id, 0.7
            
            inputs = self.intent_tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
            
            with torch.no_grad():
                outputs = self.intent_model(**inputs)
                
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # 获取最高概率的意图
            max_prob, predicted_class = torch.max(probabilities, dim=1)
            
            return predicted_class.item(), max_prob.item()
            
        except Exception as e:
            logger.error(f"识别意图失败: {str(e)}")
            # 回退到规则识别
            intent = self._rule_based_intent(text)
            category_id = list(self.intent_categories.keys())[list(self.intent_categories.values()).index(intent)]
            return category_id, 0.5
    
    def _rule_based_intent(self, text: str) -> str:
        """
        基于规则的意图识别
        
        Args:
            text: 输入文本
            
        Returns:
            意图类别
        """
        text_lower = text.lower()
        
        # 检查是否为学习类提问
        learning_keywords = [
            'how to', 'what is', 'explain', 'meaning of', 'definition', 
            'grammar', 'vocabulary', 'pronunciation', 'learn', 'study'
        ]
        if self._is_question(text) and any(kw in text_lower for kw in learning_keywords):
            return "learning_question"
        
        # 检查是否为场景对话
        scene_keywords = [
            'restaurant', 'hotel', 'airport', 'shopping', 'travel', 
            'business', 'meeting', 'doctor', 'hospital', 'school',
            'interview', 'reservation', 'booking', 'order'
        ]
        if any(kw in text_lower for kw in scene_keywords):
            return "scene_conversation"
        
        # 默认为日常闲聊
        return "casual_chat"
    
    def _is_question(self, text: str) -> bool:
        """
        判断文本是否为问题
        
        Args:
            text: 输入文本
            
        Returns:
            是否为问题
        """
        # 检查问号
        if text.endswith('?'):
            return True
            
        # 检查疑问词开头
        question_words = ['what', 'when', 'where', 'who', 'whom', 'which', 'why', 'how']
        first_word = text.strip().split()[0].lower() if text.strip() else ""
        if first_word in question_words:
            return True
            
        # 检查助动词开头
        auxiliary_verbs = ['do', 'does', 'did', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'can', 'could', 'will', 'would', 'shall', 'should']
        if first_word in auxiliary_verbs:
            return True
            
        return False
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        提取命名实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        # 简化实现，使用规则识别一些基本实体
        entities = []
        
        # 提取日期
        date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b'
        dates = re.finditer(date_pattern, text, re.IGNORECASE)
        for match in dates:
            entities.append({
                "type": "DATE",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })
        
        # 提取时间
        time_pattern = r'\b\d{1,2}:\d{2}\s*(?:am|pm)?\b|\b\d{1,2}\s*(?:am|pm)\b'
        times = re.finditer(time_pattern, text, re.IGNORECASE)
        for match in times:
            entities.append({
                "type": "TIME",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })
        
        # 提取金额
        money_pattern = r'\$\s*\d+(?:\.\d{2})?\b|\b\d+\s*(?:dollars|USD)\b'
        money = re.finditer(money_pattern, text, re.IGNORECASE)
        for match in money:
            entities.append({
                "type": "MONEY",
                "value": match.group(),
                "start": match.start(),
                "end": match.end()
            })
        
        # 提取位置
        location_pattern = r'\b(?:in|at|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        locations = re.finditer(location_pattern, text)
        for match in locations:
            if match.group(1).lower() not in ['the', 'a', 'an']:
                entities.append({
                    "type": "LOCATION",
                    "value": match.group(1),
                    "start": match.start(1),
                    "end": match.end(1)
                })
        
        return entities
    
    def extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """
        提取关键词
        
        Args:
            text: 输入文本
            max_keywords: 最大关键词数量
            
        Returns:
            关键词列表
        """
        # 简单实现：分词并去除停用词
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 英文停用词
        en_stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
            'when', 'where', 'how', 'who', 'which', 'this', 'that', 'these', 'those', 
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
            'do', 'does', 'did', 'can', 'could', 'will', 'would', 'shall', 'should', 
            'may', 'might', 'must', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 
            'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after'
        }
        
        # 中文停用词
        cn_stopwords = {
            '的', '了', '和', '是', '在', '我', '有', '不', '这', '了', '就', '也', '都', 
            '而', '要', '他', '她', '你', '我们', '你们', '他们', '她们', '它们', '那', '这个'
        }
        
        # 合并停用词
        stopwords = en_stopwords.union(cn_stopwords)
        
        # 去除停用词并统计频率
        filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序并返回前N个
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:max_keywords]]
        
        return keywords
        
    def detect_language(self, text: str) -> str:
        """
        检测文本语言
        
        Args:
            text: 输入文本
            
        Returns:
            语言代码 ("en", "zh" 等)
        """
        # 简单实现：根据字符判断
        # 中文字符占比
        chinese_ratio = len(re.findall(r'[\u4e00-\u9fff]', text)) / max(len(text), 1)
        
        # 英文字符占比
        english_ratio = len(re.findall(r'[a-zA-Z]', text)) / max(len(text), 1)
        
        if chinese_ratio > 0.3:
            return "zh"
        elif english_ratio > 0.3:
            return "en"
        else:
            return "unknown"
            
    def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """
        分析文本复杂度
        
        Args:
            text: 输入文本
            
        Returns:
            复杂度分析结果
        """
        # 分词
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return {
                "complexity_score": 0,
                "difficulty_level": "unknown",
                "average_word_length": 0,
                "sentence_count": 0,
                "word_count": 0
            }
        
        # 统计句子数
        sentences = re.split(r'[.!?]+', text)
        sentence_count = sum(1 for s in sentences if s.strip())
        
        # 计算平均词长
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # 计算复杂词占比
        complex_words = [w for w in words if len(w) > 6]
        complex_ratio = len(complex_words) / len(words)
        
        # 计算复杂度得分 (简化版Flesch-Kincaid)
        if sentence_count == 0:
            complexity_score = 0
        else:
            complexity_score = 0.39 * (len(words) / sentence_count) + 11.8 * (sum(len(word) for word in words) / len(words)) - 15.59
        
        # 确定难度级别
        if complexity_score < 30:
            difficulty_level = "beginner"
        elif complexity_score < 50:
            difficulty_level = "intermediate"
        elif complexity_score < 70:
            difficulty_level = "advanced"
        else:
            difficulty_level = "professional"
        
        return {
            "complexity_score": round(complexity_score, 2),
            "difficulty_level": difficulty_level,
            "average_word_length": round(avg_word_length, 2),
            "sentence_count": sentence_count,
            "word_count": len(words),
            "complex_word_ratio": round(complex_ratio, 2)
        } 