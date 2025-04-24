#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估工具模块
提供数据质量评估和模型评估指标计算
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Set
import re
import jieba
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

# 配置日志
logger = logging.getLogger(__name__)

class DataEvaluator:
    """数据评估工具类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化评估器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
    def calculate_text_stats(self, texts: List[str]) -> Dict[str, Any]:
        """
        计算文本统计信息
        
        Args:
            texts: 文本列表
            
        Returns:
            统计信息
        """
        if not texts:
            return {
                "count": 0,
                "avg_length": 0,
                "min_length": 0,
                "max_length": 0,
                "std_length": 0
            }
            
        # 计算长度统计
        lengths = [len(text) for text in texts]
        
        stats = {
            "count": len(texts),
            "avg_length": np.mean(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "std_length": np.std(lengths),
            "length_distribution": self._get_length_distribution(lengths)
        }
        
        # 词频统计
        if self.config.get("calculate_word_freq", True):
            all_words = []
            for text in texts:
                all_words.extend(jieba.lcut(text))
                
            word_freq = Counter(all_words)
            stats["vocabulary_size"] = len(word_freq)
            stats["top_words"] = word_freq.most_common(20)
            
        return stats
    
    def _get_length_distribution(self, lengths: List[int]) -> Dict[str, int]:
        """计算长度分布"""
        bins = [0, 10, 50, 100, 200, 500, 1000, float('inf')]
        bin_names = ["1-9", "10-49", "50-99", "100-199", "200-499", "500-999", "1000+"]
        
        distribution = {name: 0 for name in bin_names}
        
        for length in lengths:
            for i, upper in enumerate(bins[1:]):
                if length < upper:
                    distribution[bin_names[i]] += 1
                    break
                    
        return distribution
    
    def calculate_dialogue_stats(self, dialogues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算对话统计信息
        
        Args:
            dialogues: 对话列表
            
        Returns:
            统计信息
        """
        if not dialogues:
            return {"count": 0}
            
        # 对话轮次统计
        turn_counts = []
        
        # 说话者统计
        speaker_counts = Counter()
        
        # 轮次文本
        all_turns = []
        
        # 按说话者分类的文本
        speaker_texts = {}
        
        for dialogue in dialogues:
            turns = dialogue.get("turns", [])
            turn_counts.append(len(turns))
            
            for turn in turns:
                speaker = turn.get("speaker", "unknown")
                text = turn.get("text", "")
                
                speaker_counts[speaker] += 1
                all_turns.append(text)
                
                if speaker not in speaker_texts:
                    speaker_texts[speaker] = []
                speaker_texts[speaker].append(text)
                
        # 基本统计
        stats = {
            "count": len(dialogues),
            "avg_turns": np.mean(turn_counts),
            "min_turns": min(turn_counts),
            "max_turns": max(turn_counts),
            "std_turns": np.std(turn_counts),
            "speaker_distribution": {speaker: count for speaker, count in speaker_counts.most_common()}
        }
        
        # 所有轮次的文本统计
        stats["turn_stats"] = self.calculate_text_stats(all_turns)
        
        # 按说话者的文本统计
        stats["speaker_stats"] = {}
        for speaker, texts in speaker_texts.items():
            stats["speaker_stats"][speaker] = self.calculate_text_stats(texts)
            
        return stats
    
    def calculate_quality_scores(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        计算对话质量得分
        
        Args:
            dialogues: 对话列表
            
        Returns:
            对话质量评分列表
        """
        results = []
        
        for dialogue in dialogues:
            dialogue_id = dialogue.get("id", "unknown")
            turns = dialogue.get("turns", [])
            
            if not turns:
                results.append({
                    "id": dialogue_id,
                    "quality_score": 0.0,
                    "detail_scores": {
                        "length_score": 0.0,
                        "turn_score": 0.0,
                        "balance_score": 0.0,
                        "coherence_score": 0.0
                    }
                })
                continue
                
            # 计算平均轮次长度
            avg_length = sum(len(turn.get("text", "")) for turn in turns) / len(turns)
            length_score = min(avg_length / 50, 1.0)  # 50字符为理想长度
            
            # 计算轮次数量得分
            turns_count = len(turns)
            turn_score = min(turns_count / 5, 1.0)  # 5轮为理想轮次
            
            # 计算轮次平衡性 (说话者分布)
            speakers = {}
            for turn in turns:
                speaker = turn.get("speaker", "unknown")
                speakers[speaker] = speakers.get(speaker, 0) + 1
            
            # 如果只有一个说话者，平衡性较低
            if len(speakers) <= 1:
                balance_score = 0.3
            else:
                # 计算说话者分布的均匀程度
                counts = list(speakers.values())
                std_dev = np.std(counts) if len(counts) > 1 else 0
                balance_score = 1.0 - min(std_dev / (max(np.mean(counts), 1)), 0.7)
            
            # 简单的一致性评分
            coherence_score = 0.8  # 默认值
            
            # 组合得分 (可根据需要调整权重)
            final_score = (0.3 * length_score + 
                          0.3 * turn_score + 
                          0.2 * balance_score + 
                          0.2 * coherence_score)
            
            results.append({
                "id": dialogue_id,
                "quality_score": round(final_score, 2),
                "detail_scores": {
                    "length_score": round(length_score, 2),
                    "turn_score": round(turn_score, 2),
                    "balance_score": round(balance_score, 2),
                    "coherence_score": round(coherence_score, 2)
                }
            })
            
        return results
    
    def detect_outliers(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        检测异常对话
        
        Args:
            dialogues: 对话列表
            
        Returns:
            异常对话信息
        """
        outliers = []
        
        if not dialogues:
            return outliers
            
        # 计算轮次长度统计
        turn_lengths = []
        for dialogue in dialogues:
            for turn in dialogue.get("turns", []):
                turn_lengths.append(len(turn.get("text", "")))
                
        if not turn_lengths:
            return outliers
            
        # 计算统计值
        mean_length = np.mean(turn_lengths)
        std_length = np.std(turn_lengths)
        
        # 检测异常
        for dialogue in dialogues:
            dialogue_id = dialogue.get("id", "unknown")
            turns = dialogue.get("turns", [])
            
            if not turns:
                outliers.append({
                    "id": dialogue_id,
                    "reason": "empty_dialogue",
                    "score": 0.0
                })
                continue
                
            # 检查长度异常
            for i, turn in enumerate(turns):
                text = turn.get("text", "")
                turn_length = len(text)
                
                # 检查过短轮次
                if turn_length < self.config.get("min_turn_length", 2):
                    outliers.append({
                        "id": dialogue_id,
                        "turn_index": i,
                        "reason": "too_short_turn",
                        "score": 0.0
                    })
                
                # 检查过长轮次
                if turn_length > self.config.get("max_turn_length", 500):
                    outliers.append({
                        "id": dialogue_id,
                        "turn_index": i,
                        "reason": "too_long_turn",
                        "score": 0.0
                    })
                
                # 检查长度异常的轮次 (Z分数)
                z_score = (turn_length - mean_length) / max(std_length, 1)
                if abs(z_score) > 3:  # 超过3个标准差
                    outliers.append({
                        "id": dialogue_id,
                        "turn_index": i,
                        "reason": "length_outlier",
                        "score": round(z_score, 2)
                    })
                    
            # 检查对话轮次异常
            if len(turns) < self.config.get("min_dialogue_turns", 2):
                outliers.append({
                    "id": dialogue_id,
                    "reason": "too_few_turns",
                    "score": len(turns)
                })
                
        return outliers


class ModelEvaluator:
    """模型评估工具类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化模型评估器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
    def calculate_classification_metrics(self, y_true: List[Any], y_pred: List[Any], 
                                       average: str = "weighted") -> Dict[str, float]:
        """
        计算分类模型评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            average: 平均方法
            
        Returns:
            评估指标
        """
        if len(y_true) != len(y_pred) or len(y_true) == 0:
            logger.error("标签长度不匹配或为空")
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }
            
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average=average, zero_division=0)
            recall = recall_score(y_true, y_pred, average=average, zero_division=0)
            f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
            
            return {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4)
            }
        except Exception as e:
            logger.error(f"计算分类指标出错: {e}")
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }
    
    def calculate_retrieval_metrics(self, relevant_docs: List[Set[str]], retrieved_docs: List[List[str]],
                                  k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Dict[int, float]]:
        """
        计算检索模型评估指标
        
        Args:
            relevant_docs: 相关文档集合列表
            retrieved_docs: 检索结果文档ID列表
            k_values: 评估的K值列表
            
        Returns:
            评估指标
        """
        if len(relevant_docs) != len(retrieved_docs) or len(relevant_docs) == 0:
            logger.error("检索结果长度不匹配或为空")
            return {
                "precision": {k: 0.0 for k in k_values},
                "recall": {k: 0.0 for k in k_values},
                "f1": {k: 0.0 for k in k_values},
                "mrr": 0.0
            }
            
        # 初始化指标
        metrics = {
            "precision": {k: 0.0 for k in k_values},
            "recall": {k: 0.0 for k in k_values},
            "f1": {k: 0.0 for k in k_values},
            "mrr": 0.0
        }
        
        # 计算MRR
        mrr_sum = 0.0
        
        for rel_docs, ret_docs in zip(relevant_docs, retrieved_docs):
            # MRR
            mrr = 0.0
            for i, doc_id in enumerate(ret_docs):
                if doc_id in rel_docs:
                    mrr = 1.0 / (i + 1)
                    break
            mrr_sum += mrr
            
            # 对每个K值计算指标
            for k in k_values:
                ret_at_k = set(ret_docs[:k])
                
                # Precision@K
                if len(ret_at_k) > 0:
                    precision = len(ret_at_k.intersection(rel_docs)) / len(ret_at_k)
                    metrics["precision"][k] += precision
                
                # Recall@K
                if len(rel_docs) > 0:
                    recall = len(ret_at_k.intersection(rel_docs)) / len(rel_docs)
                    metrics["recall"][k] += recall
                
                # F1@K
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    metrics["f1"][k] += f1
                    
        # 计算平均值
        n = len(relevant_docs)
        
        metrics["mrr"] = round(mrr_sum / n, 4)
        
        for k in k_values:
            metrics["precision"][k] = round(metrics["precision"][k] / n, 4)
            metrics["recall"][k] = round(metrics["recall"][k] / n, 4)
            metrics["f1"][k] = round(metrics["f1"][k] / n, 4)
            
        return metrics
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度得分
        """
        if not text1 or not text2:
            return 0.0
            
        # 分词
        words1 = jieba.lcut(text1)
        words2 = jieba.lcut(text2)
        
        # 创建词袋
        words_set = set(words1 + words2)
        
        # 计算词频向量
        vec1 = {word: words1.count(word) for word in words_set}
        vec2 = {word: words2.count(word) for word in words_set}
        
        # 计算点积
        dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in words_set)
        
        # 计算模长
        norm1 = np.sqrt(sum(vec1.get(word, 0) ** 2 for word in words_set))
        norm2 = np.sqrt(sum(vec2.get(word, 0) ** 2 for word in words_set))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # 计算余弦相似度
        similarity = dot_product / (norm1 * norm2)
        
        return round(similarity, 4)
    
    def evaluate_dialogue_coherence(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        评估对话一致性
        
        Args:
            dialogues: 对话列表
            
        Returns:
            对话一致性评分
        """
        results = []
        
        for dialogue in dialogues:
            dialogue_id = dialogue.get("id", "unknown")
            turns = dialogue.get("turns", [])
            
            if len(turns) < 2:
                results.append({
                    "id": dialogue_id,
                    "coherence_score": 0.0,
                    "turn_similarities": []
                })
                continue
                
            # 计算相邻轮次的相似度
            turn_similarities = []
            coherence_sum = 0.0
            
            for i in range(1, len(turns)):
                prev_text = turns[i-1].get("text", "")
                curr_text = turns[i].get("text", "")
                
                similarity = self.calculate_text_similarity(prev_text, curr_text)
                turn_similarities.append({
                    "turn_idx": i,
                    "similarity": similarity
                })
                
                coherence_sum += similarity
                
            # 计算平均一致性得分
            avg_coherence = coherence_sum / (len(turns) - 1) if len(turns) > 1 else 0.0
            
            results.append({
                "id": dialogue_id,
                "coherence_score": round(avg_coherence, 4),
                "turn_similarities": turn_similarities
            })
            
        return results 