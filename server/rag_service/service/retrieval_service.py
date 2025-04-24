"""
检索服务
负责处理用户查询，实现混合检索和上下文优化
增强了混合检索能力和上下文处理
"""

import os
import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple

from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import numpy as np

logger = logging.getLogger(__name__)

class RetrievalService:
    """检索服务类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化检索服务
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.top_k = config.get("top_k", 5)
        
        # 初始化嵌入模型
        api_key = os.environ.get("OPENAI_API_KEY") or config.get("openai_api_key")
        if not api_key:
            raise ValueError("缺少OpenAI API密钥")
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # 混合检索权重配置
        self.vector_weight = config.get("vector_weight", 0.7)  # 向量检索权重
        self.keyword_weight = config.get("keyword_weight", 0.3)  # 关键词检索权重
        
        # 上下文优化配置
        self.enable_context_optimization = config.get("enable_context_optimization", True)
        self.context_window_size = config.get("context_window_size", 3)  # 上下文窗口大小
        
        # 高级功能设置
        self.enable_advanced_features = config.get("enable_advanced_features", True)
    
    def retrieve(self, query: str, index_service, index_name: str, 
                top_k: Optional[int] = None, 
                search_type: str = "hybrid",
                previous_queries: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        检索相关内容
        
        Args:
            query: 用户查询
            index_service: 索引服务实例
            index_name: 索引名称
            top_k: 返回结果数量
            search_type: 搜索类型 (vector, keyword, hybrid)
            previous_queries: 之前的查询列表（用于上下文感知检索）
            
        Returns:
            检索结果列表
        """
        try:
            if top_k is None:
                top_k = self.top_k
                
            # 调用索引服务的搜索函数
            raw_results = index_service.search(
                index_name=index_name,
                query=query,
                top_k=top_k * 2,  # 获取更多候选结果用于后处理
                search_type=search_type
            )
            
            if not raw_results:
                return []
                
            # 如果启用上下文优化并且有历史查询
            if self.enable_context_optimization and previous_queries and self.enable_advanced_features:
                optimized_results = self._apply_context_optimization(raw_results, query, previous_queries)
            else:
                optimized_results = raw_results
                
            # 结果排序和截断
            sorted_results = self._rank_and_filter_results(optimized_results, top_k)
            
            # 结果后处理（添加元数据等）
            processed_results = self._process_results(sorted_results)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            raise RuntimeError(f"检索失败: {str(e)}")
    
    def _apply_context_optimization(self, results: List[Dict[str, Any]], 
                                   current_query: str, 
                                   previous_queries: List[str]) -> List[Dict[str, Any]]:
        """
        应用上下文优化
        
        Args:
            results: 原始检索结果
            current_query: 当前查询
            previous_queries: 历史查询列表
            
        Returns:
            优化后的结果
        """
        try:
            # 获取最近几次查询
            recent_queries = previous_queries[-self.context_window_size:]
            
            # 计算当前查询与历史查询的相似度
            current_embedding = self.embeddings.embed_query(current_query)
            
            query_similarities = []
            for prev_query in recent_queries:
                prev_embedding = self.embeddings.embed_query(prev_query)
                similarity = self._cosine_similarity(current_embedding, prev_embedding)
                query_similarities.append(similarity)
            
            # 根据查询相似度调整结果排序
            for result in results:
                context_boost = 0.0
                
                # 检查结果是否与历史查询相关（基于关键词匹配等简单方法）
                for i, prev_query in enumerate(recent_queries):
                    query_terms = set(re.findall(r'\w+', prev_query.lower()))
                    content_lower = result["content"].lower()
                    
                    # 计算关键词匹配度
                    matched_terms = sum(1 for term in query_terms if term in content_lower)
                    if matched_terms > 0:
                        # 结合查询相似度和关键词匹配度给予提升
                        term_match_score = matched_terms / len(query_terms)
                        query_recency_factor = (i + 1) / len(recent_queries)  # 越近期的查询权重越高
                        
                        context_boost += term_match_score * query_similarities[i] * query_recency_factor
                
                # 更新结果分数
                result["context_score"] = context_boost
                # 调整总分，增加上下文提升因子
                result["score"] = result["score"] * 0.7 + context_boost * 0.3
            
            return results
            
        except Exception as e:
            logger.error(f"上下文优化失败: {str(e)}")
            # 失败时返回原始结果
            return results
    
    def _rank_and_filter_results(self, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        对结果进行排序和过滤
        
        Args:
            results: 检索结果
            top_k: 返回数量
            
        Returns:
            排序和过滤后的结果
        """
        # 按分数排序
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # 取前 top_k 个结果
        return sorted_results[:top_k]
    
    def _process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理结果，添加额外信息
        
        Args:
            results: 检索结果
            
        Returns:
            处理后的结果
        """
        processed_results = []
        
        for i, result in enumerate(results):
            # 添加排名信息
            processed_result = {
                **result,
                "rank": i + 1
            }
            
            # 提取文档标题和来源信息（如果有）
            if "metadata" in result:
                if "filename" in result["metadata"]:
                    processed_result["document_name"] = result["metadata"]["filename"]
                    
                # 添加文档链接信息（如果有）
                if "file_path" in result["metadata"]:
                    processed_result["document_url"] = result["metadata"]["file_path"]
            
            processed_results.append(processed_result)
        
        return processed_results
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            余弦相似度
        """
        np_vec1 = np.array(vec1)
        np_vec2 = np.array(vec2)
        
        dot_product = np.dot(np_vec1, np_vec2)
        norm_a = np.linalg.norm(np_vec1)
        norm_b = np.linalg.norm(np_vec2)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
        
    def generate_query_expansion(self, query: str) -> List[str]:
        """
        生成查询扩展
        
        Args:
            query: 原始查询
            
        Returns:
            扩展查询列表
        """
        if not self.enable_advanced_features:
            return [query]
            
        try:
            # 简单实现：提取关键词并生成同义词
            # 实际应用中可以使用更复杂的方法，如使用大模型生成
            keywords = re.findall(r'\w+', query.lower())
            
            # 去除停用词
            stopwords = {"的", "了", "是", "在", "和", "有", "中", "与", "为", "以", "及", "或", 
                         "a", "an", "the", "in", "on", "for", "of", "and", "to", "with"}
            keywords = [k for k in keywords if k not in stopwords and len(k) > 1]
            
            # 返回原始查询和关键词组合的查询
            expanded_queries = [query]
            
            if len(keywords) > 1:
                # 组合关键词生成新查询
                for i in range(len(keywords)):
                    for j in range(i+1, len(keywords)):
                        expanded_queries.append(f"{keywords[i]} {keywords[j]}")
            
            return expanded_queries[:3]  # 限制扩展查询数量
            
        except Exception as e:
            logger.error(f"生成查询扩展失败: {str(e)}")
            return [query]  # 失败时返回原始查询 