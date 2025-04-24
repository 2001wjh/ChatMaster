#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本处理工具模块
提供文本清洗、分词、实体识别等功能
"""

import re
import unicodedata
import logging
import jieba
import json
from typing import List, Dict, Any, Set, Optional
from pathlib import Path
import hashlib

# 配置日志
logger = logging.getLogger(__name__)

class TextProcessor:
    """文本处理工具类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化文本处理器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.stopwords = self._load_stopwords()
        self.sensitive_words = self._load_sensitive_words()
        
        # 加载jieba自定义词典（如果指定）
        if self.config.get("custom_dict_path") and Path(self.config.get("custom_dict_path")).exists():
            jieba.load_userdict(self.config.get("custom_dict_path"))
            logger.info(f"已加载自定义词典: {self.config.get('custom_dict_path')}")
            
    def _load_stopwords(self) -> Set[str]:
        """加载停用词"""
        stopwords = set()
        filepath = self.config.get("stopwords_file")
        
        if filepath and Path(filepath).exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    stopwords = set([line.strip() for line in f if line.strip()])
                logger.info(f"已加载 {len(stopwords)} 个停用词")
            except Exception as e:
                logger.error(f"加载停用词文件出错: {e}")
        
        return stopwords
    
    def _load_sensitive_words(self) -> Set[str]:
        """加载敏感词"""
        sensitive_words = set()
        filepath = self.config.get("sensitive_words_file")
        
        if filepath and Path(filepath).exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    sensitive_words = set([line.strip() for line in f if line.strip()])
                logger.info(f"已加载 {len(sensitive_words)} 个敏感词")
            except Exception as e:
                logger.error(f"加载敏感词文件出错: {e}")
        
        return sensitive_words
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not text or not isinstance(text, str):
            return ""
            
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 标准化空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # Unicode标准化
        text = unicodedata.normalize('NFKC', text)
        
        # 去除特殊字符
        if self.config.get("remove_special_chars", True):
            text = re.sub(r'[^\w\s\u4e00-\u9fff.,?!;:，。？！；：""''()]', '', text)
        
        # 去除首尾空白
        text = text.strip()
        
        return text
    
    def filter_by_length(self, text: str) -> bool:
        """
        根据长度过滤文本
        
        Args:
            text: 待过滤文本
            
        Returns:
            是否保留文本
        """
        min_length = self.config.get("min_length", 10)
        max_length = self.config.get("max_length", 1000)
        
        text_len = len(text)
        return min_length <= text_len <= max_length
    
    def contains_sensitive_words(self, text: str) -> bool:
        """
        检查文本是否包含敏感词
        
        Args:
            text: 待检查文本
            
        Returns:
            是否包含敏感词
        """
        if not self.sensitive_words:
            return False
            
        for word in self.sensitive_words:
            if word in text:
                return True
                
        return False
    
    def segment_text(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 待分词文本
            
        Returns:
            词语列表
        """
        if not text:
            return []
            
        # 使用jieba分词
        words = jieba.lcut(text)
        
        # 过滤停用词
        if self.stopwords and self.config.get("filter_stopwords", True):
            words = [w for w in words if w not in self.stopwords]
            
        return words
    
    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """
        提取关键词
        
        Args:
            text: 待处理文本
            top_k: 提取的关键词数量
            
        Returns:
            关键词列表
        """
        if not text:
            return []
            
        try:
            # 使用jieba提取关键词
            keywords = jieba.analyse.extract_tags(text, topK=top_k)
            return keywords
        except Exception as e:
            logger.error(f"提取关键词出错: {e}")
            return []
    
    def is_question(self, text: str) -> bool:
        """
        判断文本是否为问句
        
        Args:
            text: 待判断文本
            
        Returns:
            是否为问句
        """
        # 结尾有问号
        if re.search(r'[?？]$', text.strip()):
            return True
            
        # 中文问句特征
        if re.search(r'^(什么|谁|哪|何|怎|为什么|如何|是否|能否|可不可以)', text.strip()):
            return True
            
        # 英文问句特征
        if re.search(r'^(what|who|where|when|why|how|is|are|do|does|can|could|would|will|should)', 
                    text.strip().lower()):
            return True
            
        return False
    
    def calculate_text_hash(self, text: str) -> str:
        """
        计算文本哈希值
        
        Args:
            text: 文本内容
            
        Returns:
            哈希值
        """
        # 使用md5计算哈希
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def detect_language(self, text: str) -> str:
        """
        检测文本语言
        
        Args:
            text: 文本内容
            
        Returns:
            语言代码（cn/en）
        """
        # 简易语言检测
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        en_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if cn_chars > en_chars:
            return "cn"
        else:
            return "en"
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        简单的实体抽取
        
        Args:
            text: 文本内容
            
        Returns:
            实体列表
        """
        entities = []
        
        # 时间实体
        time_patterns = [
            (r'\d{4}年\d{1,2}月\d{1,2}日', 'TIME'),
            (r'\d{4}-\d{1,2}-\d{1,2}', 'TIME'),
            (r'\d{1,2}月\d{1,2}日', 'TIME'),
            (r'\d{1,2}:\d{2}', 'TIME')
        ]
        
        for pattern, label in time_patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                entities.append({
                    'text': match.group(),
                    'type': label,
                    'start': start,
                    'end': end
                })
        
        # 数字实体
        number_patterns = [
            (r'\d+\.\d+%', 'PERCENTAGE'),
            (r'\d+%', 'PERCENTAGE'),
            (r'\d+\.?\d*万', 'NUMBER'),
            (r'\d+\.?\d*亿', 'NUMBER'),
            (r'\d+\.?\d*元', 'MONEY')
        ]
        
        for pattern, label in number_patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                entities.append({
                    'text': match.group(),
                    'type': label,
                    'start': start,
                    'end': end
                })
        
        return entities 