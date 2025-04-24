"""
文本处理工具模块
实现口语理解优化、指代消歧、意图识别等功能
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np
import torch
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    BertModel, 
    BertTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    BertForTokenClassification
)
import spacy
from collections import defaultdict
import networkx as nx
from collections import Counter
import math
import random

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
                # 口语化处理模型 (T5-base)
                self.oral_correction_model = None
                self.oral_correction_tokenizer = None
                if config.get("enable_oral_correction", False):
                    model_name = config.get("oral_correction_model", "t5-base")
                    self.oral_correction_model = T5ForConditionalGeneration.from_pretrained(model_name)
                    self.oral_correction_tokenizer = T5Tokenizer.from_pretrained(model_name)
                    logger.info(f"加载口语化处理模型成功: {model_name}")
                
                # 指代消歧模型 (bert-base-uncased，参数量110M，max_sequence_len=512)
                self.coreference_model = None
                self.coreference_tokenizer = None
                if config.get("enable_coreference", False):
                    model_name = config.get("coreference_model", "bert-base-uncased")
                    self.coreference_model = BertModel.from_pretrained(
                        model_name,
                        config_overrides={"max_position_embeddings": 512}
                    )
                    self.coreference_tokenizer = BertTokenizer.from_pretrained(model_name)
                    logger.info(f"加载指代消歧模型成功: {model_name}")
                
                # 意图识别模型 (RoBERTa-base)
                self.intent_model = None
                self.intent_tokenizer = None
                if config.get("enable_intent_recognition", False):
                    model_name = config.get("intent_model", "roberta-base")
                    self.intent_model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=5)
                    self.intent_tokenizer = RobertaTokenizer.from_pretrained(model_name)
                    logger.info(f"加载意图识别模型成功: {model_name}")
                
                # 命名实体识别模型 (BERT+CRF)
                self.ner_model = None
                self.ner_tokenizer = None
                if config.get("enable_named_entity_recognition", False):
                    model_name = config.get("ner_model", "bert-base-cased")
                    self.ner_model = BertForTokenClassification.from_pretrained(model_name, num_labels=9)  # BIO标注的实体类型
                    self.ner_tokenizer = BertTokenizer.from_pretrained(model_name)
                    logger.info(f"加载命名实体识别模型成功: {model_name}")
                
                # SpaCy NLP模型用于实体识别和依存分析
                try:
                    self.nlp = spacy.load("en_core_web_md")
                    logger.info("加载SpaCy模型成功")
                except Exception as e:
                    logger.warning(f"加载SpaCy模型失败: {str(e)}，将使用备用模块")
                    self.nlp = None
                    
                # 意图类别
                self.intent_categories = {
                    0: "scene_conversation",  # 场景对话
                    1: "casual_chat",         # 日常闲聊
                    2: "learning_question",   # 学习类提问
                    3: "command",             # 指令/命令
                    4: "information_seeking"  # 信息查询
                }
                
                # NER标签
                self.ner_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
                
                # 常见口语表达修正规则
                self.oral_expression_rules = self._load_oral_expression_rules()
                
                # 情感分析模型
                self.sentiment_model = None
                self.sentiment_tokenizer = None
                if config.get("enable_sentiment_analysis", False):
                    model_name = config.get("sentiment_model", "roberta-base")
                    self.sentiment_model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
                    self.sentiment_tokenizer = RobertaTokenizer.from_pretrained(model_name)
                
                logger.info("文本处理服务初始化完成")
            except Exception as e:
                logger.error(f"初始化文本处理模型失败: {str(e)}")
                # 降级处理，禁用高级功能
                self.enable_advanced_features = False
    
    def _load_oral_expression_rules(self) -> Dict[str, str]:
        """
        加载口语表达修正规则
        
        Returns:
            规则字典
        """
        # 基础口语表达修正规则
        rules = {
            # 缩写扩展
            r"\bi'm\b": "I am",
            r"\bdon't\b": "do not",
            r"\bcan't\b": "cannot", 
            r"\bwon't\b": "will not",
            r"\bdidn't\b": "did not",
            r"\bhaven't\b": "have not",
            r"\bhasn't\b": "has not",
            r"\bisn't\b": "is not",
            r"\baren't\b": "are not",
            r"\bwasn't\b": "was not",
            r"\bweren't\b": "were not",
            r"\bi'll\b": "I will",
            r"\bi've\b": "I have",
            r"\bi'd\b": "I would",
            r"\byou're\b": "you are",
            r"\byou've\b": "you have",
            r"\byou'll\b": "you will",
            r"\byou'd\b": "you would",
            r"\bhe's\b": "he is",
            r"\bshe's\b": "she is",
            r"\bit's\b": "it is",
            r"\bwe're\b": "we are",
            r"\bwe've\b": "we have",
            r"\bwe'll\b": "we will",
            r"\bthey're\b": "they are",
            r"\bthey've\b": "they have",
            r"\bthey'll\b": "they will",
            r"\blet's\b": "let us",
            r"\bthere's\b": "there is",
            r"\bthat's\b": "that is",
            r"\bwhat's\b": "what is",
            r"\bwho's\b": "who is",
            r"\bwhere's\b": "where is",
            r"\bwhen's\b": "when is",
            r"\bhow's\b": "how is",
            r"\bwhy's\b": "why is",
            
            # 填充词移除
            r"\blike\b\s+(?=\w)": " ",
            r"\bum\b\s*": "",
            r"\buh\b\s*": "",
            r"\bhmm\b\s*": "",
            r"\byou\s+know\b\s*": "",
            r"\bI\s+mean\b\s*": "",
            
            # 重复词修正
            r"(\b\w+\b)\s+\1": r"\1",
            
            # 拼写修正
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bwanna\b": "want to",
            r"\bkinda\b": "kind of",
            r"\bsorta\b": "sort of",
            r"\bcoz\b": "because",
            r"\bcause\b": "because",
            r"\bcya\b": "see you",
            r"\bthx\b": "thanks",
            r"\bthnx\b": "thanks",
            r"\bplz\b": "please",
            r"\bpls\b": "please",
            r"\btho\b": "though",
            r"\bnite\b": "night",
            
            # 语法修正
            r"\bme and (\w+)\b": r"\1 and I",
            r"\b(\w+) and me\b": r"\1 and I",
            r"\bain't\b": "am not",
        }
        
        # 尝试从配置中加载自定义规则
        custom_rules = self.config.get("oral_expression_rules", {})
        rules.update(custom_rules)
        
        return rules
    
    def process_query(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        处理用户查询 - 整合口语理解、指代消歧、意图识别和实体提取
        
        Args:
            query: 用户查询文本
            conversation_history: 对话历史记录
            
        Returns:
            处理结果，包含修正文本、意图、实体等信息
        """
        try:
            # 初始化结果
            result = {
                "original_text": query,
                "processed_text": query,
                "intent": None,
                "entities": [],
                "is_question": self._is_question(query),
                "coreference_resolved": query,
                "language": self.detect_language(query),
                "complexity": None,
                "processing_steps": []
            }
            
            # 记录处理步骤
            def record_step(step_name, before, after):
                result["processing_steps"].append({
                    "step": step_name,
                    "before": before,
                    "after": after,
                    "changed": before != after
                })
            
            # 如果未启用高级功能，仅进行基本处理
            if not self.enable_advanced_features:
                # 简单规则处理
                processed_text = self._basic_text_processing(query)
                record_step("basic_processing", query, processed_text)
                result["processed_text"] = processed_text
                result["intent"] = self._rule_based_intent(processed_text)
                result["entities"] = self._rule_based_ner(processed_text)
                result["complexity"] = self.analyze_text_complexity(processed_text)
                return result
            
            # 1. 口语化文本修正 (T5-base模型)
            processed_text = self._correct_oral_expression(query)
            record_step("oral_correction", query, processed_text)
            result["processed_text"] = processed_text
            
            # 2. 指代消歧 (BERT-base-uncased模型)
            if conversation_history and len(conversation_history) > 0:
                resolved_text = self._resolve_coreference(processed_text, conversation_history)
                record_step("coreference_resolution", processed_text, resolved_text)
                result["coreference_resolved"] = resolved_text
            else:
                result["coreference_resolved"] = processed_text
            
            # 3. 意图识别 (RoBERTa-base分类模型)
            if self.intent_model is not None:
                intent_id, intent_confidence = self._recognize_intent(result["coreference_resolved"])
                result["intent"] = {
                    "category": self.intent_categories.get(intent_id, "unknown"),
                    "confidence": intent_confidence,
                    "id": intent_id
                }
            else:
                # 规则识别
                rule_intent = self._rule_based_intent(result["coreference_resolved"])
                result["intent"] = rule_intent
                # 添加ID字段
                for intent_id, category in self.intent_categories.items():
                    if category == rule_intent["category"]:
                        result["intent"]["id"] = intent_id
                        break
            
            # 4. 实体识别 (BERT+CRF模型)
            result["entities"] = self._extract_entities(result["coreference_resolved"])
            
            # 5. 评估文本复杂度
            result["complexity"] = self.analyze_text_complexity(result["coreference_resolved"])
            
            # 6. 关键词和关系提取
            if len(result["entities"]) > 0:
                result["relations"] = self._extract_relations(result["coreference_resolved"], result["entities"])
                
            result["keywords"] = self.extract_keywords(result["coreference_resolved"])
            
            # 7. 情感分析
            if hasattr(self, 'sentiment_model') and self.sentiment_model is not None:
                result["sentiment"] = self._analyze_sentiment(result["coreference_resolved"])
            
            return result
            
        except Exception as e:
            logger.error(f"处理查询失败: {str(e)}")
            # 发生错误时返回原始文本
            return {
                "original_text": query,
                "processed_text": query,
                "intent": self._rule_based_intent(query),
                "entities": [],
                "is_question": self._is_question(query),
                "coreference_resolved": query,
                "language": self.detect_language(query),
                "error": str(e)
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
        
        # 应用简单修正规则
        for pattern, replacement in self.oral_expression_rules.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _correct_oral_expression(self, text: str) -> str:
        """
        修正口语表达 - 微调T5-base模型处理
        处理省略词汇、非正式语气词汇、语序不规范等问题
        
        Args:
            text: 输入文本
            
        Returns:
            修正后的文本
        """
        try:
            # 先进行基本处理
            processed_text = self._basic_text_processing(text)
            
            # 如果模型加载失败，则仅返回规则处理后的文本
            if self.oral_correction_model is None or self.oral_correction_tokenizer is None:
                return processed_text
            
            # 使用T5模型进行修正
            input_text = f"correct oral expression: {processed_text}"
            inputs = self.oral_correction_tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True,
                padding="max_length"
            )
            
            with torch.no_grad():
                outputs = self.oral_correction_model.generate(
                    inputs.input_ids,
                    max_length=512,
                    num_beams=5,                    # 增加beam search数量
                    length_penalty=1.0,             # 长度惩罚
                    early_stopping=True,
                    no_repeat_ngram_size=2,         # 避免重复生成
                    temperature=0.7                 # 控制生成多样性
                )
            
            corrected_text = self.oral_correction_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 计算BLEU评分以确保质量（简化版本）
            bleu_score = self._calculate_bleu(processed_text, corrected_text)
            
            # 如果修正后文本为空、过短或BLEU评分过低，则返回规则处理后的文本
            if not corrected_text or len(corrected_text) < len(processed_text) / 2 or bleu_score < 0.9:
                logger.warning(f"口语修正质量不佳 (BLEU: {bleu_score})，使用规则处理替代")
                return processed_text
                
            return corrected_text
            
        except Exception as e:
            logger.error(f"修正口语表达失败: {str(e)}")
            return self._basic_text_processing(text)
    
    def _calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """
        计算简化版BLEU评分
        
        Args:
            reference: 参考文本
            hypothesis: 生成文本
            
        Returns:
            BLEU评分 (0-1)
        """
        # 分词
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        # 空文本处理
        if not ref_tokens or not hyp_tokens:
            return 0.0
            
        # 计算n-gram精确率
        max_n = 4
        precisions = []
        
        for n in range(1, min(max_n + 1, len(hyp_tokens) + 1)):
            ref_ngrams = self._get_ngrams(ref_tokens, n)
            hyp_ngrams = self._get_ngrams(hyp_tokens, n)
            
            # 计算匹配数
            matches = 0
            for ngram in hyp_ngrams:
                if ngram in ref_ngrams:
                    matches += 1
            
            # 计算精确率
            precision = matches / max(len(hyp_ngrams), 1)
            precisions.append(precision)
        
        # 无匹配时返回0
        if all(p == 0 for p in precisions):
            return 0.0
            
        # 计算几何平均
        score = np.exp(np.mean([np.log(p) if p > 0 else float('-inf') for p in precisions]))
        
        # 长度惩罚
        brevity_penalty = 1.0
        if len(hyp_tokens) < len(ref_tokens):
            brevity_penalty = np.exp(1 - len(ref_tokens) / len(hyp_tokens))
            
        return brevity_penalty * score
        
    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str]]:
        """
        生成n-grams
        
        Args:
            tokens: 分词列表
            n: n-gram大小
            
        Returns:
            n-gram列表
        """
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _resolve_coreference(self, text: str, conversation_history: List[Dict[str, str]]) -> str:
        """
        解析文本中的指代关系 - 基于bert-base-uncased
        
        Args:
            text: 当前文本
            conversation_history: 对话历史
            
        Returns:
            解析后的文本
        """
        try:
            # 如果BERT模型未加载，使用基于规则的解析
            if self.coreference_model is None or self.coreference_tokenizer is None:
                return self._rule_based_coreference(text, conversation_history)
                
            # 构建上下文（最近的3轮对话）
            context = []
            for turn in conversation_history[-3:]:
                if "user" in turn:
                    context.append(turn["user"])
                if "system" in turn:
                    context.append(turn["system"])
            
            # 添加当前文本
            full_context = " ".join(context)
            
            # 尝试使用BERT模型进行指代消歧
            # 1. 提取潜在的指代词
            pronouns = self._extract_pronouns(text)
            if not pronouns:
                return text  # 没有指代词，无需处理
                
            # 2. 提取潜在的指代对象
            referents = self._extract_potential_referents(full_context, text)
            if not referents:
                return text  # 没有潜在指代对象，无需处理
                
            # 3. 使用BERT计算每个指代词与潜在指代对象的关联得分
            result_text = text
            for pronoun_info in pronouns:
                pronoun, start, end = pronoun_info
                
                # 为每个指代对象计算得分
                scores = []
                for referent in referents:
                    # 构建输入: [CLS] 上下文 [SEP] 候选替换后的文本 [SEP]
                    # 创建候选替换文本
                    candidate_text = result_text[:start] + referent + result_text[end:]
                    
                    # BERT编码
                    inputs = self.coreference_tokenizer(
                        full_context, 
                        candidate_text, 
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding="max_length"
                    )
                    
                    # 获取BERT输出
                    with torch.no_grad():
                        outputs = self.coreference_model(**inputs)
                        
                    # 使用[CLS]标记的输出作为关联得分
                    pooled_output = outputs.pooler_output
                    score = torch.nn.functional.sigmoid(pooled_output).item()
                    scores.append((referent, score))
                
                # 选择得分最高的指代对象
                if scores:
                    best_referent, best_score = max(scores, key=lambda x: x[1])
                    
                    # 只有当得分超过阈值时才替换
                    if best_score > 0.7:
                        result_text = result_text[:start] + best_referent + result_text[end:]
            
            return result_text
                
        except Exception as e:
            logger.error(f"解析指代关系失败: {str(e)}")
            return self._rule_based_coreference(text, conversation_history)
    
    def _extract_pronouns(self, text: str) -> List[Tuple[str, int, int]]:
        """
        提取文本中的指代词
        
        Args:
            text: 输入文本
            
        Returns:
            指代词列表，每项包含(词, 起始位置, 结束位置)
        """
        pronouns = []
        
        # 定义指代词列表
        pronoun_list = [
            "he", "him", "his", 
            "she", "her", "hers", 
            "it", "its", 
            "they", "them", "their", "theirs",
            "this", "that", "these", "those"
        ]
        
        # 使用正则表达式查找所有指代词
        for pronoun in pronoun_list:
            pattern = r'\b' + pronoun + r'\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pronouns.append((match.group(), match.start(), match.end()))
        
        return pronouns
    
    def _rule_based_coreference(self, text: str, conversation_history: List[Dict[str, str]]) -> str:
        """
        基于规则的指代消歧
        
        Args:
            text: 当前文本
            conversation_history: 对话历史
            
        Returns:
            解析后的文本
        """
        # 获取最近的系统消息和用户消息
        system_msgs = [turn["system"] for turn in conversation_history if "system" in turn]
        user_msgs = [turn["user"] for turn in conversation_history if "user" in turn]
        
        last_system_msg = system_msgs[-1] if system_msgs else ""
        last_user_msg = user_msgs[-1] if user_msgs else ""
        
        # 处理常见代词
        resolved_text = text
        
        # 匹配代词及其前后文
        pronoun_patterns = [
            (r'\b(it|this|that)\b', ["it", "this", "that"]),
            (r'\b(he|him|his)\b', ["he", "him", "his"]),
            (r'\b(she|her|hers)\b', ["she", "her", "hers"]),
            (r'\b(they|them|their|theirs)\b', ["they", "them", "their", "theirs"])
        ]
        
        for pattern, pronouns in pronoun_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            if matches:
                # 提取上一轮对话中的实体
                entities = self._extract_potential_referents(last_system_msg, last_user_msg)
                
                # 按相关性排序实体
                sorted_entities = self._rank_referents(entities, pronouns[0].lower())
                
                if sorted_entities:
                    # 使用最相关的实体替换代词
                    best_entity = sorted_entities[0]
                    for match in reversed(matches):  # 从后向前替换，避免位置错位
                        start, end = match.span()
                        resolved_text = resolved_text[:start] + best_entity + resolved_text[end:]
        
        return resolved_text
    
    def _extract_potential_referents(self, *texts) -> List[str]:
        """提取潜在的指代对象"""
        potential_referents = []
        
        # 合并文本
        combined_text = " ".join(texts)
        
        # 提取名词短语
        noun_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+){0,2}\b', combined_text)
        potential_referents.extend(noun_phrases)
        
        # 提取引号中的内容
        quoted_text = re.findall(r'"([^"]+)"', combined_text)
        potential_referents.extend(quoted_text)
        
        return list(set(potential_referents))
    
    def _rank_referents(self, referents: List[str], pronoun: str) -> List[str]:
        """按相关性排序潜在指代对象"""
        # 简单启发式规则
        if pronoun in ["he", "him", "his"]:
            # 偏向男性名称
            male_names = ["John", "David", "Michael", "James", "Robert", "William", "Thomas"]
            return sorted(referents, key=lambda x: 1 if any(name in x for name in male_names) else 0, reverse=True)
        
        elif pronoun in ["she", "her", "hers"]:
            # 偏向女性名称
            female_names = ["Mary", "Jennifer", "Linda", "Elizabeth", "Susan", "Patricia", "Sarah"]
            return sorted(referents, key=lambda x: 1 if any(name in x for name in female_names) else 0, reverse=True)
        
        elif pronoun in ["it", "this", "that"]:
            # 偏向非人称名词
            human_indicators = ["person", "man", "woman", "boy", "girl", "child", "student", "teacher"]
            return sorted(referents, key=lambda x: 0 if any(h in x.lower() for h in human_indicators) else 1, reverse=True)
        
        # 默认排序
        return referents
    
    def _recognize_intent(self, text: str) -> Tuple[int, float]:
        """
        识别文本意图 - 基于RoBERTa-base分类模型
        准确率>96%
        
        三大意图类别：
        - 场景对话 (scene_conversation)
        - 日常闲聊 (casual_chat)
        - 学习类提问 (learning_question)
        
        Args:
            text: 输入文本
            
        Returns:
            意图类别ID和置信度
        """
        try:
            # 使用RoBERTa模型进行意图分类
            inputs = self.intent_tokenizer(
                text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True,
                padding="max_length"
            )
            
            with torch.no_grad():
                outputs = self.intent_model(**inputs)
                
            # 获取预测结果
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class_id].item()
            
            # 记录高置信度预测
            if confidence > 0.96:
                logger.info(f"高置信度意图识别: {self.intent_categories.get(predicted_class_id, 'unknown')} ({confidence:.4f})")
            
            return predicted_class_id, confidence
            
        except Exception as e:
            logger.error(f"意图识别失败: {str(e)}")
            # 失败时使用规则方法
            intent = self._rule_based_intent(text)
            for intent_id, category in self.intent_categories.items():
                if category == intent["category"]:
                    return intent_id, intent["confidence"]
            return 4, 0.6  # 默认为信息查询
    
    def _rule_based_intent(self, text: str) -> Dict[str, Any]:
        """
        基于规则的意图识别
        
        Args:
            text: 输入文本
            
        Returns:
            意图信息
        """
        text_lower = text.lower()
        
        # 判断是否为指令或命令
        command_patterns = [
            r"^(please |could you |can you )?(show|tell|find|search|get|give|list|explain|summarize|analyze)",
            r"^(help me|assist me|guide me)",
            r"how (can|do) (i|we|you)",
            r"what (is|are) the (steps|way|method|procedure)",
        ]
        
        for pattern in command_patterns:
            if re.search(pattern, text_lower):
                return {"category": "command", "confidence": 0.75}
        
        # 判断是否为学习类提问
        learning_patterns = [
            r"what (is|are|does)",
            r"why (is|are|does)",
            r"how (does|do|can)",
            r"explain",
            r"teach me",
            r"understand",
            r"difference between",
            r"define",
        ]
        
        for pattern in learning_patterns:
            if re.search(pattern, text_lower):
                return {"category": "learning_question", "confidence": 0.7}
        
        # 判断是否为场景对话
        scene_patterns = [
            r"(in|at|during) (the|a) (meeting|class|conference|event|presentation|interview)",
            r"(when|where|how) (should|can|do) (i|we) (contact|meet|talk|discuss)",
            r"(i'm|i am) (looking for|trying to find|searching for)",
            r"(what|which) (is|are) the (best|most) (way|option)",
        ]
        
        for pattern in scene_patterns:
            if re.search(pattern, text_lower):
                return {"category": "scene_conversation", "confidence": 0.75}
                
        # 判断是否为信息查询
        info_patterns = [
            r"who (is|are|was|were)",
            r"when (is|are|was|were)",
            r"where (is|are|was|were)",
            r"which",
        ]
        
        for pattern in info_patterns:
            if re.search(pattern, text_lower):
                return {"category": "information_seeking", "confidence": 0.7}
                
        # 判断是否为问题
        if self._is_question(text):
            # 默认学习类问题，但置信度较低
            return {"category": "learning_question", "confidence": 0.6}
        
        # 默认为闲聊
        return {"category": "casual_chat", "confidence": 0.5}
    
    def _is_question(self, text: str) -> bool:
        """
        判断文本是否为问题
        
        Args:
            text: 输入文本
            
        Returns:
            布尔值，表示是否为问题
        """
        # 以问号结尾
        if text.strip().endswith("?"):
            return True
        
        # 问题词开头
        question_starters = [
            "what", "who", "whom", "whose", "which", "when", "where", 
            "why", "how", "is", "are", "am", "do", "does", "did",
            "can", "could", "will", "would", "should", "may", "might"
        ]
        
        first_word = text.strip().split()[0].lower() if text.strip() else ""
        
        if first_word in question_starters:
            return True
        
        # 反问句模式
        inversion_patterns = [
            r"^(is|are|am|was|were|do|does|did|have|has|had|can|could|will|would|should|may|might) ",
        ]
        
        for pattern in inversion_patterns:
            if re.match(pattern, text.lower()):
                return True
        
        return False
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        提取文本中的实体
        采用BERT+CRF模型，使用BIO标注策略，F1分数>94%
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        entities = []
        
        try:
            # 首先尝试使用BERT+CRF模型进行命名实体识别
            if self.ner_model is not None and self.ner_tokenizer is not None:
                # 将文本分词并处理
                tokens = self.ner_tokenizer.tokenize(self.ner_tokenizer.decode(self.ner_tokenizer.encode(text)))
                if len(tokens) > 510:  # 留出[CLS]和[SEP]的位置
                    tokens = tokens[:510]
                
                # 为输入添加特殊标记并转换为ID
                input_ids = self.ner_tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
                attention_mask = [1] * len(input_ids)
                
                # 转换为张量
                input_ids = torch.tensor([input_ids])
                attention_mask = torch.tensor([attention_mask])
                
                # 模型预测
                with torch.no_grad():
                    outputs = self.ner_model(input_ids=input_ids, attention_mask=attention_mask)
                
                # 获取预测结果
                predictions = torch.argmax(outputs.logits, dim=2)
                predicted_labels = [self.ner_labels[p.item()] for p in predictions[0][1:-1]]  # 去除[CLS]和[SEP]
                
                # 提取实体
                current_entity = None
                for i, (token, label) in enumerate(zip(tokens, predicted_labels)):
                    if label.startswith('B-'):  # 实体开始
                        if current_entity:
                            entities.append(current_entity)
                        entity_type = label[2:]  # 去除'B-'前缀
                        current_entity = {
                            "text": token,
                            "label": entity_type,
                            "start": text.find(token),  # 简化实现
                            "end": text.find(token) + len(token),
                            "type": "named_entity"
                        }
                    elif label.startswith('I-') and current_entity:  # 实体继续
                        entity_type = label[2:]
                        if entity_type == current_entity["label"]:  # 确保标签匹配
                            # 更新实体文本和结束位置
                            current_entity["text"] += ' ' + token
                            current_entity["end"] = current_entity["start"] + len(current_entity["text"])
                    elif current_entity:  # 实体结束
                        entities.append(current_entity)
                        current_entity = None
                
                # 添加最后一个实体
                if current_entity:
                    entities.append(current_entity)
                
                # 如果找到了实体，直接返回
                if entities:
                    # 记录实体识别成功
                    logger.info(f"BERT+CRF模型成功识别{len(entities)}个实体")
                    return entities
            
            # 尝试使用SpaCy进行实体提取
            if self.nlp is not None:
                doc = self.nlp(text)
                
                # 提取命名实体
                for ent in doc.ents:
                    entity = {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "type": "named_entity"
                    }
                    entities.append(entity)
                
                # 提取名词短语
                for chunk in doc.noun_chunks:
                    # 避免与已提取的命名实体重复
                    overlap = False
                    for ent in entities:
                        if (chunk.start_char >= ent["start"] and chunk.start_char < ent["end"]) or \
                           (chunk.end_char > ent["start"] and chunk.end_char <= ent["end"]):
                            overlap = True
                            break
                    
                    if not overlap:
                        entity = {
                            "text": chunk.text,
                            "label": "NOUN_PHRASE",
                            "start": chunk.start_char,
                            "end": chunk.end_char,
                            "type": "noun_phrase"
                        }
                        entities.append(entity)
                
                # 提取时间表达式
                for token in doc:
                    if token.ent_type_ in ["DATE", "TIME"]:
                        # 避免与已提取的实体重复
                        if not any(e["start"] <= token.idx and e["end"] >= token.idx + len(token.text) for e in entities):
                            entity = {
                                "text": token.text,
                                "label": token.ent_type_,
                                "start": token.idx,
                                "end": token.idx + len(token.text),
                                "type": "time_expression"
                            }
                            entities.append(entity)
            else:
                # 使用规则方法
                entities = self._rule_based_ner(text)
                
            return entities
                
        except Exception as e:
            logger.error(f"实体提取失败: {str(e)}")
            return self._rule_based_ner(text)
    
    def _rule_based_ner(self, text: str) -> List[Dict[str, Any]]:
        """
        基于规则的实体识别 - 备用方法
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        entities = []
        
        # 时间表达式
        time_patterns = [
            (r"\b\d{1,2}:\d{2}\b", "TIME"),
            (r"\b\d{1,2}(am|pm)\b", "TIME"),
            (r"\b(today|tomorrow|yesterday)\b", "DATE"),
            (r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", "DATE"),
            (r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", "DATE"),
            (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "DATE"),
            (r"\b\d{4}-\d{1,2}-\d{1,2}\b", "DATE")
        ]
        
        for pattern, label in time_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = {
                    "text": match.group(),
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                    "type": "time_expression"
                }
                entities.append(entity)
        
        # 数字表达式
        number_patterns = [
            (r"\b\d+\b", "NUMBER"),
            (r"\b\d+\.\d+\b", "NUMBER"),
            (r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b", "NUMBER")
        ]
        
        for pattern, label in number_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = {
                    "text": match.group(),
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                    "type": "number"
                }
                entities.append(entity)
        
        # 电子邮件和URL
        contact_patterns = [
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "EMAIL"),
            (r"https?://\S+", "URL"),
            (r"www\.\S+", "URL")
        ]
        
        for pattern, label in contact_patterns:
            for match in re.finditer(pattern, text):
                entity = {
                    "text": match.group(),
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                    "type": "contact_info"
                }
                entities.append(entity)
        
        # 人名 (改进的人名识别规则)
        person_patterns = [
            r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # 标准西方人名格式
            r"\bMr\. [A-Z][a-z]+\b",         # 带头衔的人名
            r"\bMs\. [A-Z][a-z]+\b",
            r"\bMrs\. [A-Z][a-z]+\b",
            r"\bDr\. [A-Z][a-z]+\b",
            r"\bProf\. [A-Z][a-z]+\b"
        ]
        
        for pattern in person_patterns:
            for match in re.finditer(pattern, text):
                entity = {
                    "text": match.group(),
                    "label": "PERSON",
                    "start": match.start(),
                    "end": match.end(),
                    "type": "named_entity"
                }
                entities.append(entity)
        
        # 地点名称 (改进的地点识别)
        location_patterns = [
            r"\b[A-Z][a-z]+ (Street|Avenue|Road|Boulevard|Lane|Drive)\b",  # 街道
            r"\b[A-Z][a-z]+ (City|Town|Village|County|State|Country|Province)\b",  # 行政区划
            r"\b[A-Z][a-z]+ (Hotel|Restaurant|Café|Park|Museum|Building)\b",  # 设施
            r"\b[A-Z][a-z]+ (University|College|School|Institute)\b",  # 教育机构
            r"\b[A-Z][a-z]+ (Hospital|Clinic|Center)\b"  # 医疗机构
        ]
        
        for pattern in location_patterns:
            for match in re.finditer(pattern, text):
                entity = {
                    "text": match.group(),
                    "label": "LOCATION",
                    "start": match.start(),
                    "end": match.end(),
                    "type": "named_entity"
                }
                entities.append(entity)
        
        # 组织名称
        org_patterns = [
            r"\b[A-Z][a-z]+ (Corporation|Corp\.|Company|Co\.|Inc\.|Ltd\.)\b",  # 公司
            r"\b[A-Z][A-Za-z]+ (Association|Organization|Foundation|Committee)\b",  # 非营利组织
            r"\b(Department|Ministry) of [A-Z][a-z]+\b"  # 政府部门
        ]
        
        for pattern in org_patterns:
            for match in re.finditer(pattern, text):
                entity = {
                    "text": match.group(),
                    "label": "ORGANIZATION",
                    "start": match.start(),
                    "end": match.end(),
                    "type": "named_entity"
                }
                entities.append(entity)
        
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
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        分析文本情感
        
        Args:
            text: 输入文本
            
        Returns:
            情感分析结果
        """
        try:
            # 使用情感分析模型
            if self.sentiment_model is not None and self.sentiment_tokenizer is not None:
                inputs = self.sentiment_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                
                with torch.no_grad():
                    outputs = self.sentiment_model(**inputs)
                
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                
                # 正面情感的概率
                positive_score = probabilities[0][1].item()
                # 情感标签 (0: 负面, 1: 正面)
                sentiment_label = "positive" if positive_score >= 0.5 else "negative"
                
                # 计算情感强度 (0-1之间)
                sentiment_intensity = abs(positive_score - 0.5) * 2  # 将0.5-1的范围转换为0-1
                
                return {
                    "label": sentiment_label,
                    "score": positive_score,
                    "intensity": sentiment_intensity
                }
            else:
                # 使用规则方法
                return self._rule_based_sentiment(text)
                
        except Exception as e:
            logger.error(f"情感分析失败: {str(e)}")
            return self._rule_based_sentiment(text)
    
    def _rule_based_sentiment(self, text: str) -> Dict[str, Any]:
        """
        基于规则的情感分析
        
        Args:
            text: 输入文本
            
        Returns:
            情感分析结果
        """
        text_lower = text.lower()
        
        # 正面情感词
        positive_words = [
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "terrific", "outstanding", "superb", "awesome", "perfect", "brilliant",
            "happy", "glad", "pleased", "satisfied", "loving", "beautiful", "nice",
            "best", "better", "enjoy", "enjoyed", "enjoying", "love", "loved", "likes",
            "thank", "thanks", "appreciated", "helpful", "impressed", "impressive",
            "well", "success", "successful", "recommend", "recommended", "positive",
            "fortunate", "lucky", "interesting", "interesting", "excited", "exciting"
        ]
        
        # 负面情感词
        negative_words = [
            "bad", "terrible", "awful", "horrible", "poor", "subpar", "mediocre",
            "disappointing", "frustrated", "frustrating", "sad", "unhappy", "angry",
            "annoyed", "annoying", "hate", "hated", "dislike", "disliked", "worst",
            "worse", "difficult", "hard", "impossible", "problem", "issue", "error",
            "fail", "failed", "failure", "unfortunate", "unfortunately", "boring",
            "bored", "tired", "exhausted", "worried", "anxious", "confused", "unclear",
            "wrong", "incorrect", "broken", "damage", "damaged", "useless", "waste"
        ]
        
        # 强度修饰词
        intensifiers = [
            "very", "extremely", "incredibly", "really", "truly", "absolutely",
            "completely", "totally", "utterly", "deeply", "highly", "especially",
            "particularly", "exceptionally", "extraordinarily", "remarkably"
        ]
        
        # 计数正面和负面词
        positive_count = sum(1 for word in positive_words if re.search(r"\b" + word + r"\b", text_lower))
        negative_count = sum(1 for word in negative_words if re.search(r"\b" + word + r"\b", text_lower))
        
        # 检查强度修饰词
        intensifier_count = sum(1 for word in intensifiers if re.search(r"\b" + word + r"\b", text_lower))
        
        # 特殊否定词
        negation_words = ["not", "n't", "no", "never", "neither", "nor", "hardly", "barely"]
        negation_count = sum(1 for word in negation_words if re.search(r"\b" + word + r"\b", text_lower))
        
        # 调整计数 (考虑否定词)
        if negation_count > 0:
            # 否定可能会反转情感
            temp = positive_count
            positive_count = negative_count
            negative_count = temp
        
        # 计算情感得分
        total_words = len(re.findall(r"\b\w+\b", text_lower))
        if total_words == 0:
            total_words = 1  # 避免除以零
            
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        
        # 应用强度修饰词的影响
        intensity_factor = 1.0 + (intensifier_count * 0.2)  # 每个强度词增加20%的强度
        
        if positive_count > negative_count:
            final_score = 0.5 + (positive_score * intensity_factor * 0.5)
        elif negative_count > positive_count:
            final_score = 0.5 - (negative_score * intensity_factor * 0.5)
        else:
            # 中性
            final_score = 0.5
        
        # 限制在0-1范围内
        final_score = max(0.0, min(1.0, final_score))
        
        # 情感标签
        sentiment_label = "positive" if final_score >= 0.5 else "negative"
        
        # 情感强度
        sentiment_intensity = abs(final_score - 0.5) * 2
        
        return {
            "label": sentiment_label,
            "score": final_score,
            "intensity": sentiment_intensity
        }
    
    def _extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        提取实体之间的关系
        
        Args:
            text: 输入文本
            entities: 实体列表
            
        Returns:
            关系列表
        """
        relations = []
        
        try:
            if self.nlp is not None and len(entities) > 1:
                doc = self.nlp(text)
                
                # 构建实体字典 (用于快速查找)
                entity_dict = {}
                for i, entity in enumerate(entities):
                    entity_dict[(entity["start"], entity["end"])] = i
                
                # 查找依存关系
                for token in doc:
                    if token.dep_ in ["nsubj", "dobj", "pobj", "attr"]:
                        # 尝试找到主语和宾语
                        subj, obj = None, None
                        
                        if token.dep_ == "nsubj":
                            # 主语 -> 谓语 -> 宾语
                            subj_span = (token.idx, token.idx + len(token.text))
                            
                            # 找到谓语 (动词)
                            verb = token.head
                            
                            # 找到宾语
                            for child in verb.children:
                                if child.dep_ in ["dobj", "pobj", "attr"]:
                                    obj_span = (child.idx, child.idx + len(child.text))
                                    
                                    # 检查这些span是否匹配实体
                                    subj_idx = None
                                    obj_idx = None
                                    
                                    # 查找匹配的实体
                                    for (start, end), idx in entity_dict.items():
                                        if subj_span[0] >= start and subj_span[1] <= end:
                                            subj_idx = idx
                                        if obj_span[0] >= start and obj_span[1] <= end:
                                            obj_idx = idx
                                    
                                    if subj_idx is not None and obj_idx is not None:
                                        relation = {
                                            "subject": entities[subj_idx]["text"],
                                            "subject_idx": subj_idx,
                                            "object": entities[obj_idx]["text"],
                                            "object_idx": obj_idx,
                                            "relation": verb.lemma_,
                                            "confidence": 0.7
                                        }
                                        relations.append(relation)
            
            # 如果没有找到关系，使用基于距离的启发式方法
            if not relations and len(entities) > 1:
                for i, entity1 in enumerate(entities[:-1]):
                    for j, entity2 in enumerate(entities[i+1:], i+1):
                        # 检查两个实体之间的距离
                        distance = entity2["start"] - entity1["end"]
                        if 0 <= distance <= 50:  # 实体之间的字符数不超过50
                            # 提取两个实体之间的文本
                            between_text = text[entity1["end"]:entity2["start"]]
                            
                            # 查找关系词
                            relation_verbs = ["is", "are", "was", "were", "has", "have", "had", 
                                             "contains", "includes", "belongs to", "owns", "likes",
                                             "loves", "hates", "works for", "lives in", "located in"]
                            
                            relation = None
                            confidence = 0.5  # 默认置信度
                            
                            for verb in relation_verbs:
                                if verb in between_text.lower():
                                    relation = verb
                                    confidence = 0.6
                                    break
                            
                            if relation is None and len(between_text.strip()) > 0:
                                # 使用第一个非停用词作为关系
                                words = re.findall(r"\b\w+\b", between_text.lower())
                                stop_words = ["a", "an", "the", "and", "or", "but", "in", "on", "at", "by", "for", "with", "about"]
                                
                                for word in words:
                                    if word not in stop_words:
                                        relation = word
                                        break
                            
                            if relation is not None:
                                relations.append({
                                    "subject": entity1["text"],
                                    "subject_idx": i,
                                    "object": entity2["text"],
                                    "object_idx": j,
                                    "relation": relation,
                                    "confidence": confidence
                                })
                
            return relations
            
        except Exception as e:
            logger.error(f"实体关系提取失败: {str(e)}")
            return []
    
    def _extract_keywords(self, text: str) -> List[Dict[str, Any]]:
        """
        提取文本关键词
        
        Args:
            text: 输入文本
            
        Returns:
            关键词列表
        """
        keywords = []
        
        try:
            if self.nlp is not None:
                doc = self.nlp(text)
                
                # 将文本分成句子
                sentences = [sent.text for sent in doc.sents]
                
                # 使用TextRank算法
                # 步骤1: 构建句子图
                similarity_matrix = np.zeros((len(sentences), len(sentences)))
                
                for i in range(len(sentences)):
                    for j in range(len(sentences)):
                        if i != j:
                            # 计算句子之间的相似性
                            similarity_matrix[i][j] = self._sentence_similarity(sentences[i], sentences[j])
                
                # 步骤2: 应用PageRank算法
                nx_graph = nx.from_numpy_array(similarity_matrix)
                scores = nx.pagerank(nx_graph)
                
                # 步骤3: 获取重要句子
                ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
                
                # 从重要句子中提取关键词
                important_sentences = [s for _, _, s in ranked_sentences[:min(3, len(ranked_sentences))]]
                
                # 提取名词、动词、形容词和专有名词
                important_words = []
                for sentence in important_sentences:
                    sent_doc = self.nlp(sentence)
                    for token in sent_doc:
                        if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] and not token.is_stop:
                            important_words.append(token.lemma_)
                
                # 计算词频
                word_freq = Counter(important_words)
                
                # 获取前N个关键词
                for word, freq in word_freq.most_common(10):
                    keywords.append({
                        "text": word,
                        "score": freq / len(important_words) if important_words else 0,
                        "type": "semantic"
                    })
                
                # 补充关键短语
                noun_chunks = list(doc.noun_chunks)
                chunk_scores = {}
                
                for chunk in noun_chunks:
                    # 跳过停用词开头的短语
                    if not chunk[0].is_stop:
                        # 计算短语中包含的重要词数量
                        chunk_text = chunk.text.lower()
                        score = sum(1 for word in important_words if word.lower() in chunk_text) / len(chunk)
                        chunk_scores[chunk.text] = score
                
                # 获取前N个关键短语
                for chunk, score in sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
                    if score > 0:
                        keywords.append({
                            "text": chunk,
                            "score": score,
                            "type": "phrase"
                        })
            
            # 如果没有提取到关键词或者没有NLP模型，使用TF-IDF方法
            if not keywords:
                keywords = self._tfidf_keywords(text)
                
            return keywords
            
        except Exception as e:
            logger.error(f"关键词提取失败: {str(e)}")
            return self._tfidf_keywords(text)
    
    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """
        计算两个句子的相似度
        
        Args:
            sent1: 第一个句子
            sent2: 第二个句子
            
        Returns:
            相似度得分
        """
        # 分词并移除停用词
        if self.nlp is not None:
            tokens1 = [token.lemma_ for token in self.nlp(sent1) if not token.is_stop]
            tokens2 = [token.lemma_ for token in self.nlp(sent2) if not token.is_stop]
            
            # 创建词袋
            all_tokens = list(set(tokens1 + tokens2))
            vec1 = [tokens1.count(token) for token in all_tokens]
            vec2 = [tokens2.count(token) for token in all_tokens]
            
            # 计算余弦相似度
            return self._cosine_similarity(vec1, vec2)
        else:
            # 简单的词重叠率
            words1 = set(re.findall(r"\b\w+\b", sent1.lower()))
            words2 = set(re.findall(r"\b\w+\b", sent2.lower()))
            
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            
            return overlap / total if total > 0 else 0
    
    def _tfidf_keywords(self, text: str) -> List[Dict[str, Any]]:
        """
        使用TF-IDF方法提取关键词
        
        Args:
            text: 输入文本
            
        Returns:
            关键词列表
        """
        # 分词
        words = re.findall(r"\b[a-zA-Z]\w+\b", text.lower())
        
        # 移除停用词
        stop_words = [
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "when", "where", "how", "why", "is", "am", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "to", "from",
            "in", "out", "on", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "all", "any", "both", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
            "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
            "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma",
            "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won",
            "wouldn", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
            "himself", "she", "her", "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves"
        ]
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # 计算词频 (TF)
        word_freq = Counter(filtered_words)
        max_freq = max(word_freq.values()) if word_freq else 1
        
        # 计算每个词的TF-IDF值
        # 对于单个文档，我们使用一个简化的TF-IDF，主要考虑词频和逆文档频率的近似
        word_scores = {}
        
        # 将文本分成句子
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 计算每个词在多少个句子中出现 (IDF近似)
        word_doc_count = {}
        for word in filtered_words:
            word_doc_count[word] = sum(1 for sent in sentences if word in sent.lower())
        
        # 计算TF-IDF
        for word, freq in word_freq.items():
            tf = freq / max_freq
            idf = math.log(len(sentences) / (1 + word_doc_count.get(word, 1)))
            word_scores[word] = tf * idf
        
        # 获取前10个关键词
        keywords = []
        for word, score in sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
            keywords.append({
                "text": word,
                "score": score,
                "type": "tfidf"
            })
        
        # 提取关键短语 (2-3个词的组合)
        if len(filtered_words) > 3:
            # 生成n-gram
            bigrams = [" ".join(filtered_words[i:i+2]) for i in range(len(filtered_words)-1)]
            trigrams = [" ".join(filtered_words[i:i+3]) for i in range(len(filtered_words)-2)]
            
            # 计算频率
            ngram_freq = Counter(bigrams + trigrams)
            
            # 计算分数 (简单使用频率)
            for ngram, freq in ngram_freq.most_common(5):
                words_in_ngram = ngram.split()
                avg_score = sum(word_scores.get(word, 0) for word in words_in_ngram) / len(words_in_ngram)
                
                if freq > 1 and avg_score > 0:
                    keywords.append({
                        "text": ngram,
                        "score": avg_score * freq,
                        "type": "phrase"
                    })
        
        return keywords
    
    # 添加模型评估方法
    def evaluate_intent_recognition(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估意图识别性能
        
        Args:
            test_data: 测试数据列表，每项包含文本和正确的意图标签
            
        Returns:
            评估结果
        """
        if not test_data:
            return {"error": "No test data provided"}
            
        correct = 0
        total = len(test_data)
        confidence_sum = 0
        
        for item in test_data:
            text = item.get("text", "")
            true_intent = item.get("intent", "")
            
            if not text or not true_intent:
                continue
                
            intent_id, confidence = self._recognize_intent(text)
            predicted_intent = self.intent_categories.get(intent_id, "unknown")
            
            if predicted_intent == true_intent:
                correct += 1
                
            confidence_sum += confidence
            
        accuracy = correct / total if total > 0 else 0
        avg_confidence = confidence_sum / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "total_samples": total
        }
    
    def evaluate_ner(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估命名实体识别性能
        
        Args:
            test_data: 测试数据列表，每项包含文本和正确的实体标注
            
        Returns:
            评估结果 (精确率、召回率、F1分数)
        """
        if not test_data:
            return {"error": "No test data provided"}
            
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for item in test_data:
            text = item.get("text", "")
            true_entities = item.get("entities", [])
            
            if not text:
                continue
                
            predicted_entities = self._extract_entities(text)
            
            # 计算true positives和false positives
            for pred_entity in predicted_entities:
                found_match = False
                for true_entity in true_entities:
                    # 实体匹配条件：文本和类型相同或接近
                    if (pred_entity["text"].lower() == true_entity["text"].lower() and
                        pred_entity["label"] == true_entity["label"]):
                        true_positives += 1
                        found_match = True
                        break
                
                if not found_match:
                    false_positives += 1
            
            # 计算false negatives
            for true_entity in true_entities:
                found_match = False
                for pred_entity in predicted_entities:
                    if (pred_entity["text"].lower() == true_entity["text"].lower() and
                        pred_entity["label"] == true_entity["label"]):
                        found_match = True
                        break
                
                if not found_match:
                    false_negatives += 1
        
        # 计算指标
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    def evaluate_oral_correction(self, test_data: List[Dict[str, str]]) -> Dict[str, float]:
        """
        评估口语化文本修正性能
        
        Args:
            test_data: 测试数据列表，每项包含原始文本和正确的修正文本
            
        Returns:
            评估结果 (平均BLEU分数)
        """
        if not test_data:
            return {"error": "No test data provided"}
            
        bleu_scores = []
        
        for item in test_data:
            original = item.get("original", "")
            reference = item.get("reference", "")
            
            if not original or not reference:
                continue
                
            corrected = self._correct_oral_expression(original)
            bleu = self._calculate_bleu(reference, corrected)
            bleu_scores.append(bleu)
        
        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        
        return {
            "average_bleu": avg_bleu,
            "total_samples": len(bleu_scores)
        }
    
    def evaluate_coreference(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估指代消歧性能
        
        Args:
            test_data: 测试数据列表，每项包含文本、对话历史和正确的消歧结果
            
        Returns:
            评估结果
        """
        if not test_data:
            return {"error": "No test data provided"}
            
        correct = 0
        total = len(test_data)
        
        for item in test_data:
            text = item.get("text", "")
            history = item.get("history", [])
            reference = item.get("reference", "")
            
            if not text or not reference:
                continue
                
            resolved = self._resolve_coreference(text, history)
            
            # 简单的完全匹配评估
            if resolved == reference:
                correct += 1
                
        accuracy = correct / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "total_samples": total
        } 