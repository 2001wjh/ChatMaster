#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对话数据处理模块
实现文本清洗、特征提取、质量评估和数据增强等功能
"""

import os
import json
import logging
import re
import unicodedata
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import time
import hashlib
from dataclasses import dataclass, field

try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

@dataclass
class DialogueTurn:
    """对话轮次数据结构"""
    speaker: str
    text: str
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "speaker": self.speaker,
            "text": self.text,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DialogueTurn':
        """从字典创建对话轮次对象"""
        return cls(
            speaker=data.get("speaker", ""),
            text=data.get("text", ""),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {})
        )

@dataclass
class Dialogue:
    """完整对话数据结构"""
    id: str
    turns: List[DialogueTurn]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "turns": [turn.to_dict() for turn in self.turns],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Dialogue':
        """从字典创建对话对象"""
        return cls(
            id=data.get("id", ""),
            turns=[DialogueTurn.from_dict(turn) for turn in data.get("turns", [])],
            metadata=data.get("metadata", {})
        )
    
    def get_text(self, join_char="\n") -> str:
        """获取对话的完整文本"""
        return join_char.join([f"{turn.speaker}: {turn.text}" for turn in self.turns])

class DialogueProcessor:
    """对话数据处理器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化对话处理器
        
        参数:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.logger = self._setup_logger()
        self.config = self._load_config(config_path)
        self.embedding_model = self._load_embedding_model()
        self.stopwords = self._load_stopwords()
        self.sensitive_words = self._load_sensitive_words()
        self.logger.info("对话处理器初始化完成")
    
    def _setup_logger(self) -> logging.Logger:
        """配置日志记录器"""
        logger = logging.getLogger("DialogueProcessor")
        logger.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "text_processing": {
                "min_length": 10,
                "max_length": 1000,
                "enable_gpt4_rules": True,
                "quality_threshold": 0.7
            },
            "filtering": {
                "stopwords_file": "",
                "sensitive_words_file": "",
                "enable_sensitive_check": True,
                "enable_stopword_filtering": True
            },
            "retrieval": {
                "bm25": {"enabled": True},
                "vector": {"enabled": True, "model_name": "paraphrase-multilingual-MiniLM-L12-v2"},
                "hybrid": {"enabled": True, "bm25_weight": 0.4, "vector_weight": 0.6}
            },
            "output": {
                "format": "json",
                "include_metadata": True,
                "export_dir": "data/processed/"
            },
            "logging": {
                "level": "INFO",
                "file": "",
                "console": True
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # 合并配置
                    self._update_nested_dict(default_config, loaded_config)
                    self.logger.info(f"配置文件已加载: {config_path}")
            except Exception as e:
                self.logger.error(f"加载配置文件出错: {e}")
        else:
            self.logger.warning("未找到配置文件，使用默认配置")
            
        # 配置日志级别
        log_level = getattr(logging, default_config["logging"]["level"], logging.INFO)
        self.logger.setLevel(log_level)
        
        # 添加文件处理器
        if default_config["logging"]["file"]:
            try:
                log_dir = os.path.dirname(default_config["logging"]["file"])
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                file_handler = logging.FileHandler(default_config["logging"]["file"], encoding='utf-8')
                file_handler.setLevel(log_level)
                file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.error(f"设置日志文件出错: {e}")
                
        return default_config
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """递归更新嵌套字典"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _load_embedding_model(self) -> Optional[Any]:
        """加载嵌入模型"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning("sentence_transformers 库未安装，向量检索功能不可用")
            return None
            
        if self.config["retrieval"]["vector"]["enabled"]:
            try:
                model_name = self.config["retrieval"]["vector"]["model_name"]
                self.logger.info(f"正在加载嵌入模型: {model_name}")
                model = SentenceTransformer(model_name)
                self.logger.info("嵌入模型加载完成")
                return model
            except Exception as e:
                self.logger.error(f"加载嵌入模型出错: {e}")
                return None
        return None
    
    def _load_stopwords(self) -> Set[str]:
        """加载停用词"""
        stopwords = set()
        filepath = self.config["filtering"]["stopwords_file"]
        
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    stopwords = set([line.strip() for line in f if line.strip()])
                self.logger.info(f"已加载 {len(stopwords)} 个停用词")
            except Exception as e:
                self.logger.error(f"加载停用词文件出错: {e}")
        
        return stopwords
    
    def _load_sensitive_words(self) -> Set[str]:
        """加载敏感词"""
        sensitive_words = set()
        filepath = self.config["filtering"]["sensitive_words_file"]
        
        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    sensitive_words = set([line.strip() for line in f if line.strip()])
                self.logger.info(f"已加载 {len(sensitive_words)} 个敏感词")
            except Exception as e:
                self.logger.error(f"加载敏感词文件出错: {e}")
        
        return sensitive_words
    
    def process_dialogue(self, dialogue: Union[Dict[str, Any], Dialogue]) -> Optional[Dialogue]:
        """
        处理单个对话
        
        参数:
            dialogue: 对话数据，可以是字典或Dialogue对象
            
        返回:
            处理后的Dialogue对象，如果处理失败则返回None
        """
        try:
            # 转换为Dialogue对象
            if isinstance(dialogue, dict):
                dialogue = Dialogue.from_dict(dialogue)
            
            # 检查对话是否有效
            if not dialogue.turns:
                self.logger.warning(f"对话 {dialogue.id} 没有轮次数据，跳过处理")
                return None
                
            # 处理每个对话轮次
            processed_turns = []
            for turn in dialogue.turns:
                processed_turn = self._process_turn(turn)
                if processed_turn:
                    processed_turns.append(processed_turn)
            
            # 如果所有轮次都被过滤掉，则返回None
            if not processed_turns:
                self.logger.warning(f"对话 {dialogue.id} 的所有轮次都被过滤掉")
                return None
            
            # 构建处理后的对话
            processed_dialogue = Dialogue(
                id=dialogue.id,
                turns=processed_turns,
                metadata=dialogue.metadata.copy()
            )
            
            # 添加质量分数
            quality_score = self._evaluate_dialogue_quality(processed_dialogue)
            processed_dialogue.metadata["quality_score"] = quality_score
            
            # 添加处理时间戳
            processed_dialogue.metadata["processed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            return processed_dialogue
        except Exception as e:
            self.logger.error(f"处理对话 {getattr(dialogue, 'id', 'unknown')} 时出错: {e}")
            return None
    
    def _process_turn(self, turn: DialogueTurn) -> Optional[DialogueTurn]:
        """
        处理单个对话轮次
        
        参数:
            turn: 对话轮次对象
            
        返回:
            处理后的对话轮次对象，如果被过滤则返回None
        """
        # 复制元数据，避免修改原始对象
        metadata = turn.metadata.copy() if turn.metadata else {}
        
        # 文本清洗
        text = self._clean_text(turn.text)
        if not text:
            return None
            
        # 检查文本长度
        if len(text) < self.config["text_processing"]["min_length"]:
            return None
        if len(text) > self.config["text_processing"]["max_length"]:
            text = text[:self.config["text_processing"]["max_length"]]
        
        # 敏感词检查
        if self.config["filtering"]["enable_sensitive_check"] and self._contains_sensitive_words(text):
            self.logger.warning(f"文本包含敏感词: {text[:30]}...")
            metadata["contains_sensitive"] = True
        
        # 提取特征
        keywords = self._extract_keywords(text)
        if keywords:
            metadata["keywords"] = keywords
        
        # 创建处理后的轮次
        processed_turn = DialogueTurn(
            speaker=turn.speaker,
            text=text,
            timestamp=turn.timestamp,
            metadata=metadata
        )
        
        return processed_turn
    
    def _clean_text(self, text: str) -> str:
        """
        清洗文本
        
        参数:
            text: 原始文本
            
        返回:
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
        
        # 去除首尾空白
        text = text.strip()
        
        return text
    
    def _contains_sensitive_words(self, text: str) -> bool:
        """
        检查文本是否包含敏感词
        
        参数:
            text: 待检查文本
            
        返回:
            是否包含敏感词
        """
        if not self.sensitive_words:
            return False
            
        for word in self.sensitive_words:
            if word in text:
                return True
                
        return False
    
    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """
        提取文本关键词
        
        参数:
            text: 待处理文本
            top_k: 提取的关键词数量
            
        返回:
            关键词列表
        """
        if not JIEBA_AVAILABLE:
            return []
            
        try:
            # 使用jieba提取关键词
            keywords = jieba.analyse.extract_tags(text, topK=top_k)
            return keywords
        except Exception as e:
            self.logger.error(f"提取关键词出错: {e}")
            return []
    
    def _evaluate_dialogue_quality(self, dialogue: Dialogue) -> float:
        """
        评估对话质量
        
        参数:
            dialogue: 对话对象
            
        返回:
            质量得分 (0.0-1.0)
        """
        if not dialogue.turns:
            return 0.0
            
        # 计算平均轮次长度
        avg_length = sum(len(turn.text) for turn in dialogue.turns) / len(dialogue.turns)
        length_score = min(avg_length / 50, 1.0)  # 50字符为理想长度
        
        # 计算轮次数量得分
        turns_count = len(dialogue.turns)
        turns_score = min(turns_count / 5, 1.0)  # 5轮为理想轮次
        
        # 计算轮次平衡性 (说话者分布)
        speakers = {}
        for turn in dialogue.turns:
            speakers[turn.speaker] = speakers.get(turn.speaker, 0) + 1
        
        # 如果只有一个说话者，平衡性较低
        if len(speakers) <= 1:
            balance_score = 0.3
        else:
            # 计算说话者分布的均匀程度
            counts = list(speakers.values())
            std_dev = np.std(counts) if len(counts) > 1 else 0
            balance_score = 1.0 - min(std_dev / (max(np.mean(counts), 1)), 0.7)
        
        # 组合得分 (可根据需要调整权重)
        final_score = 0.4 * length_score + 0.3 * turns_score + 0.3 * balance_score
        
        return round(final_score, 2)
    
    def process_batch(self, dialogues: List[Union[Dict[str, Any], Dialogue]]) -> List[Dialogue]:
        """
        批量处理对话
        
        参数:
            dialogues: 对话列表
            
        返回:
            处理后的对话列表
        """
        processed = []
        start_time = time.time()
        total = len(dialogues)
        
        self.logger.info(f"开始批量处理 {total} 个对话")
        
        for i, dialogue in enumerate(dialogues):
            if (i + 1) % 100 == 0 or (i + 1) == total:
                self.logger.info(f"已处理 {i+1}/{total} 个对话 ({(i+1)/total*100:.1f}%)")
                
            processed_dialogue = self.process_dialogue(dialogue)
            if processed_dialogue:
                processed.append(processed_dialogue)
        
        elapsed = time.time() - start_time
        self.logger.info(f"批量处理完成，共处理 {total} 个对话，保留 {len(processed)} 个，耗时 {elapsed:.2f} 秒")
        
        return processed
    
    def save_processed_dialogues(self, dialogues: List[Dialogue], output_path: str = None) -> str:
        """
        保存处理后的对话
        
        参数:
            dialogues: 对话列表
            output_path: 输出文件路径，如果为None则使用配置中的默认路径
            
        返回:
            保存的文件路径
        """
        if not dialogues:
            self.logger.warning("没有对话需要保存")
            return ""
            
        # 确定输出路径
        if not output_path:
            output_dir = self.config["output"]["export_dir"]
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"processed_dialogues_{timestamp}.json")
        
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        try:
            # 转换为字典列表
            data_to_save = [dialogue.to_dict() for dialogue in dialogues]
            
            # 保存到文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"已将 {len(dialogues)} 个处理后的对话保存到: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"保存处理后的对话出错: {e}")
            return ""
    
    def generate_dialogue_embeddings(self, dialogues: List[Dialogue]) -> Dict[str, np.ndarray]:
        """
        生成对话嵌入向量
        
        参数:
            dialogues: 对话列表
            
        返回:
            对话ID到嵌入向量的映射
        """
        if not self.embedding_model:
            self.logger.warning("嵌入模型未加载，无法生成对话嵌入")
            return {}
            
        embeddings = {}
        batch_size = self.config["retrieval"]["vector"].get("batch_size", 32)
        
        try:
            # 提取对话文本
            dialogue_texts = []
            dialogue_ids = []
            
            for dialogue in dialogues:
                dialogue_text = dialogue.get_text()
                if dialogue_text.strip():
                    dialogue_texts.append(dialogue_text)
                    dialogue_ids.append(dialogue.id)
            
            # 分批处理
            for i in range(0, len(dialogue_texts), batch_size):
                batch_texts = dialogue_texts[i:i+batch_size]
                batch_ids = dialogue_ids[i:i+batch_size]
                
                # 生成嵌入
                batch_embeddings = self.embedding_model.encode(batch_texts)
                
                # 保存结果
                for j, dialogue_id in enumerate(batch_ids):
                    embeddings[dialogue_id] = batch_embeddings[j]
                    
                self.logger.info(f"已生成 {min(i+batch_size, len(dialogue_texts))}/{len(dialogue_texts)} 个对话的嵌入向量")
                
            return embeddings
        except Exception as e:
            self.logger.error(f"生成对话嵌入出错: {e}")
            return {}
    
    def load_dialogues_from_file(self, file_path: str) -> List[Dialogue]:
        """
        从文件加载对话数据
        
        参数:
            file_path: 文件路径
            
        返回:
            对话列表
        """
        dialogues = []
        
        if not os.path.exists(file_path):
            self.logger.error(f"文件不存在: {file_path}")
            return dialogues
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                for item in data:
                    dialogue = Dialogue.from_dict(item)
                    dialogues.append(dialogue)
            elif isinstance(data, dict):
                dialogue = Dialogue.from_dict(data)
                dialogues.append(dialogue)
                
            self.logger.info(f"从文件 {file_path} 加载了 {len(dialogues)} 个对话")
            return dialogues
        except Exception as e:
            self.logger.error(f"从文件加载对话数据出错: {e}")
            return []


if __name__ == "__main__":
    # 示例用法
    processor = DialogueProcessor("data_processing/config/dialogue_config.json")
    
    # 创建示例对话
    dialogue = Dialogue(
        id="dialogue_001",
        turns=[
            DialogueTurn(speaker="用户", text="你好，我想了解一下你们的产品"),
            DialogueTurn(speaker="客服", text="您好！欢迎咨询我们的产品。我们提供多种智能家居解决方案，包括智能灯光、温控和安防系统。请问您对哪方面更感兴趣呢？"),
            DialogueTurn(speaker="用户", text="我对智能安防比较感兴趣，能详细介绍一下吗？"),
            DialogueTurn(speaker="客服", text="当然可以。我们的智能安防系统包括门窗传感器、移动侦测器和智能摄像头，可以实时监控您的家并发送警报到您的手机。系统支持远程查看和控制，还能与其他智能家居设备联动。")
        ],
        metadata={"source": "模拟数据", "category": "产品咨询"}
    )
    
    # 处理对话
    processed_dialogue = processor.process_dialogue(dialogue)
    if processed_dialogue:
        print(f"处理后的对话质量得分: {processed_dialogue.metadata.get('quality_score', 0)}")
        
        # 保存处理后的对话
        output_path = processor.save_processed_dialogues([processed_dialogue])
        print(f"对话已保存到: {output_path}") 