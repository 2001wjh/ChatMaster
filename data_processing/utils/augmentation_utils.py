#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据增强工具模块
提供多种文本数据增强方法
"""

import logging
import random
import re
import jieba
from typing import List, Dict, Any, Optional, Union, Tuple
import os
import json
from pathlib import Path

# 配置日志
logger = logging.getLogger(__name__)

class DataAugmentor:
    """数据增强工具类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化数据增强器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.synonym_dict = self._load_synonym_dict()
        
    def _load_synonym_dict(self) -> Dict[str, List[str]]:
        """
        加载同义词字典
        
        Returns:
            同义词字典
        """
        synonym_dict = {}
        
        # 尝试加载同义词文件
        dict_path = self.config.get("synonym_dict_path")
        if dict_path and os.path.exists(dict_path):
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    synonym_dict = json.load(f)
                logger.info(f"已加载同义词字典: {dict_path}，包含 {len(synonym_dict)} 个词条")
            except Exception as e:
                logger.error(f"加载同义词字典出错: {e}")
                
        return synonym_dict
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """
        同义词替换增强
        
        Args:
            text: 原始文本
            n: 替换词数量
            
        Returns:
            增强后的文本
        """
        if not text:
            return text
            
        if not self.synonym_dict:
            logger.warning("未加载同义词字典，无法进行同义词替换")
            return text
            
        words = jieba.lcut(text)
        
        # 如果文本太短，减少替换数量
        n = min(n, max(1, len(words) // 10))
        
        # 获取可替换的词索引（排除标点符号和过短的词）
        indices = [i for i, word in enumerate(words) if len(word) > 1 and word in self.synonym_dict]
        
        if not indices:
            return text
            
        # 随机选择n个词进行替换
        replace_indices = random.sample(indices, min(n, len(indices)))
        
        for idx in replace_indices:
            word = words[idx]
            synonyms = self.synonym_dict.get(word, [])
            
            if synonyms:
                words[idx] = random.choice(synonyms)
                
        return ''.join(words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """
        随机插入增强
        
        Args:
            text: 原始文本
            n: 插入次数
            
        Returns:
            增强后的文本
        """
        if not text:
            return text
            
        words = jieba.lcut(text)
        
        # 如果文本太短，减少插入数量
        n = min(n, max(1, len(words) // 5))
        
        # 选择可用于插入的词
        candidates = [word for word in words if len(word) > 1]
        
        if not candidates:
            return text
            
        for _ in range(n):
            # 随机选择一个词
            word = random.choice(candidates)
            
            # 随机选择插入位置
            insert_pos = random.randint(0, len(words))
            
            words.insert(insert_pos, word)
            
        return ''.join(words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """
        随机交换增强
        
        Args:
            text: 原始文本
            n: 交换次数
            
        Returns:
            增强后的文本
        """
        if not text:
            return text
            
        words = jieba.lcut(text)
        
        if len(words) < 2:
            return text
            
        # 如果文本太短，减少交换数量
        n = min(n, max(1, len(words) // 4))
        
        for _ in range(n):
            # 随机选择两个不同位置
            pos1, pos2 = random.sample(range(len(words)), 2)
            
            # 交换词语
            words[pos1], words[pos2] = words[pos2], words[pos1]
            
        return ''.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        随机删除增强
        
        Args:
            text: 原始文本
            p: 每个词被删除的概率
            
        Returns:
            增强后的文本
        """
        if not text:
            return text
            
        words = jieba.lcut(text)
        
        if len(words) <= 3:
            return text
            
        # 随机删除词语，但保留至少一半的词
        min_keep = max(3, int(len(words) * 0.5))
        kept_words = []
        
        for word in words:
            if random.random() >= p or len(kept_words) < min_keep:
                kept_words.append(word)
                
        # 如果删除太多，确保保留一定数量的词
        if len(kept_words) < min_keep:
            indices_to_add = random.sample(range(len(words)), min_keep - len(kept_words))
            for idx in indices_to_add:
                if words[idx] not in kept_words:
                    kept_words.append(words[idx])
                    
        return ''.join(kept_words)
    
    def back_translation(self, text: str, target_lang: str = "en") -> str:
        """
        回译增强（需要外部翻译API支持）
        
        Args:
            text: 原始文本
            target_lang: 目标语言
            
        Returns:
            增强后的文本
        """
        logger.warning("回译增强需要外部API支持，当前为示例实现")
        
        # 这里需要接入翻译API
        # 示例实现，实际应用中需替换为真实API调用
        if target_lang == "en":
            # 模拟中文到英文再到中文的回译过程
            if "你好" in text:
                return text.replace("你好", "您好")
            if "早上好" in text:
                return text.replace("早上好", "早安")
                
        return text
    
    def rule_based_augmentation(self, text: str, rules: List[Tuple[str, str]]) -> str:
        """
        基于规则的增强
        
        Args:
            text: 原始文本
            rules: 规则列表，每个规则是(pattern, replacement)元组
            
        Returns:
            增强后的文本
        """
        if not text or not rules:
            return text
            
        result = text
        
        # 应用所有规则
        for pattern, replacement in rules:
            result = re.sub(pattern, replacement, result)
            
        return result
    
    def augment(self, text: str, methods: List[str] = None, n_per_method: int = 1) -> List[str]:
        """
        综合数据增强
        
        Args:
            text: 原始文本
            methods: 使用的增强方法列表
            n_per_method: 每种方法生成的样本数量
            
        Returns:
            增强后的文本列表
        """
        if not text:
            return []
            
        if methods is None:
            methods = ["synonym", "insert", "swap", "delete"]
            
        augmented_texts = []
        
        for method in methods:
            for _ in range(n_per_method):
                if method == "synonym":
                    aug_text = self.synonym_replacement(text, n=random.randint(1, 3))
                elif method == "insert":
                    aug_text = self.random_insertion(text, n=random.randint(1, 2))
                elif method == "swap":
                    aug_text = self.random_swap(text, n=random.randint(1, 2))
                elif method == "delete":
                    aug_text = self.random_deletion(text, p=random.uniform(0.05, 0.1))
                elif method == "backtrans":
                    aug_text = self.back_translation(text)
                else:
                    continue
                    
                # 确保增强后的文本与原文本不同
                if aug_text != text and aug_text not in augmented_texts:
                    augmented_texts.append(aug_text)
                    
        return augmented_texts
    
    def augment_dialogue(self, dialogue: Dict[str, Any], methods: List[str] = None, 
                        n_per_method: int = 1) -> List[Dict[str, Any]]:
        """
        对话数据增强
        
        Args:
            dialogue: 对话数据
            methods: 使用的增强方法列表
            n_per_method: 每种方法生成的样本数量
            
        Returns:
            增强后的对话数据列表
        """
        if not dialogue or "turns" not in dialogue:
            return []
            
        augmented_dialogues = []
        turns = dialogue.get("turns", [])
        
        if not turns:
            return []
            
        # 对每一轮对话进行增强
        for _ in range(n_per_method * len(methods) if methods else 0):
            new_turns = []
            
            for turn in turns:
                speaker = turn.get("speaker", "")
                text = turn.get("text", "")
                
                # 对用户或客户的话语进行增强，保持系统或客服的话语不变
                if speaker.lower() in ["user", "customer", "用户", "客户"]:
                    # 随机选择一种增强方法
                    method = random.choice(methods) if methods else "synonym"
                    
                    if method == "synonym":
                        aug_text = self.synonym_replacement(text, n=random.randint(1, 2))
                    elif method == "insert":
                        aug_text = self.random_insertion(text, n=1)
                    elif method == "swap":
                        aug_text = self.random_swap(text, n=1)
                    elif method == "delete":
                        aug_text = self.random_deletion(text, p=0.05)
                    else:
                        aug_text = text
                else:
                    aug_text = text
                    
                new_turns.append({
                    "speaker": speaker,
                    "text": aug_text,
                    "timestamp": turn.get("timestamp"),
                    "metadata": turn.get("metadata", {})
                })
                
            # 创建新的对话
            new_dialogue = dialogue.copy()
            new_dialogue["id"] = f"{dialogue.get('id', 'unknown')}_aug_{len(augmented_dialogues)}"
            new_dialogue["turns"] = new_turns
            
            # 添加元数据标记
            if "metadata" not in new_dialogue:
                new_dialogue["metadata"] = {}
            new_dialogue["metadata"]["augmented"] = True
            new_dialogue["metadata"]["original_id"] = dialogue.get("id", "unknown")
            
            augmented_dialogues.append(new_dialogue)
            
        return augmented_dialogues 