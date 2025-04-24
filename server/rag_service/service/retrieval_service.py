"""
检索服务模块
实现混合检索、上下文优化及检索结果处理
"""

import logging
import re
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    检索服务
    
    提供混合检索、上下文优化及检索结果处理等功能
    支持向量检索、关键词检索、混合检索以及基于上下文的结果优化
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化检索服务
        
        Args:
            config: 配置参数
        """
        self.config = config
        
        # 混合检索权重
        self.vector_weight = config.get("vector_weight", 0.7)
        self.keyword_weight = config.get("keyword_weight", 0.3)
        
        # 上下文优化配置
        self.enable_context_optimization = config.get("enable_context_optimization", True)
        self.context_recency_weight = config.get("context_recency_weight", 0.3)  # 最近查询的权重
        self.context_similarity_weight = config.get("context_similarity_weight", 0.5)  # 查询相似度的权重
        self.context_relevance_weight = config.get("context_relevance_weight", 0.2)  # 结果相关性的权重
        
        # 高级特性
        self.enable_advanced_features = config.get("enable_advanced_features", True)
        self.enable_query_expansion = config.get("enable_query_expansion", True)
        self.enable_semantic_reranking = config.get("enable_semantic_reranking", True)
        self.enable_query_classification = config.get("enable_query_classification", True)
        self.enable_source_diversification = config.get("enable_source_diversification", True)
        
        # 记忆存储
        self.query_memory = {}  # {index_name: List[query]}
        self.result_memory = {}  # {index_name: {query: List[result]}}
        
        # 最大记忆条目数
        self.max_memory_items = config.get("max_memory_items", 100)
        
        logger.info("检索服务初始化完成")
        
    def retrieve(
        self, 
        query: str, 
        index_service: Any,
        index_name: str,
        top_k: int = 5,
        search_type: str = "hybrid", 
        previous_queries: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行检索
        
        Args:
            query: 用户查询
            index_service: 索引服务实例
            index_name: 索引名称
            top_k: 返回结果数量
            search_type: 检索类型 (vector, keyword, hybrid, auto)
            previous_queries: 之前的查询列表
            filters: 过滤条件
            
        Returns:
            检索结果列表
        """
        logger.info(f"执行检索: query={query}, index={index_name}, type={search_type}, top_k={top_k}")
        start_time = time.time()
        
        # 检查索引是否存在
        if not index_service.index_exists(index_name):
            logger.warning(f"索引不存在: {index_name}")
            return []
        
        # 无效查询处理
        if not query or not query.strip():
            logger.warning("查询为空")
            return []
            
        # 将auto类型转为hybrid    
        if search_type == "auto":
            search_type = "hybrid"
        
        # 应用查询扩展（如果启用）
        expanded_query = query
        query_expansions = []
        if self.enable_advanced_features and self.enable_query_expansion:
            expanded_query, query_expansions = self._generate_query_expansion(query)
            logger.info(f"查询扩展: {query} -> {expanded_query}")
            logger.info(f"扩展关键词: {query_expansions}")
        
        # 执行检索
        results = []
        
        if search_type == "vector" or search_type == "hybrid":
            # 向量检索
            vector_results = index_service.vector_search(
                index_name=index_name,
                query=expanded_query,
                top_k=top_k * 2 if search_type == "hybrid" else top_k,
                filters=filters
            )
            
            if vector_results:
                # 添加检索类型标记
                for item in vector_results:
                    item["search_type"] = "vector"
                
                if search_type == "vector":
                    results = vector_results
                    logger.info(f"向量检索完成: 找到{len(results)}条结果")
        
        if search_type == "keyword" or search_type == "hybrid":
            # 关键词检索
            keyword_results = index_service.keyword_search(
                index_name=index_name,
                query=expanded_query,
                top_k=top_k * 2 if search_type == "hybrid" else top_k,
                filters=filters
            )
            
            if keyword_results:
                # 添加检索类型标记
                for item in keyword_results:
                    item["search_type"] = "keyword"
                
                if search_type == "keyword":
                    results = keyword_results
                    logger.info(f"关键词检索完成: 找到{len(results)}条结果")
        
        # 混合检索结果合并
        if search_type == "hybrid":
            # 合并结果并计算混合得分
            combined_results = []
            seen_doc_ids = set()
            
            # 处理向量结果
            for item in vector_results:
                doc_id = item.get("document_id")
                if doc_id and doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    item["hybrid_score"] = item.get("score", 0) * self.vector_weight
                    combined_results.append(item)
            
            # 处理关键词结果
            for item in keyword_results:
                doc_id = item.get("document_id")
                if doc_id and doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    item["hybrid_score"] = item.get("score", 0) * self.keyword_weight
                    combined_results.append(item)
                elif doc_id:
                    # 更新已存在的结果
                    for existing in combined_results:
                        if existing.get("document_id") == doc_id:
                            keyword_score = item.get("score", 0) * self.keyword_weight
                            existing["hybrid_score"] = existing.get("hybrid_score", 0) + keyword_score
                            existing["keyword_matches"] = item.get("keyword_matches", [])
                            existing["search_type"] = "hybrid"
                            break
            
            # 按混合得分排序
            combined_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
            
            # 更新score字段为hybrid_score
            for item in combined_results:
                if "hybrid_score" in item:
                    item["score"] = item["hybrid_score"]
                    # 保留hybrid_score字段用于调试
            
            results = combined_results[:top_k]
            logger.info(f"混合检索完成: 找到{len(results)}条结果")
        
        # 应用上下文优化（如果启用）
        if self.enable_context_optimization and previous_queries:
            results = self._apply_context_optimization(
                query=query,
                previous_queries=previous_queries,
                results=results,
                index_name=index_name
            )
            logger.info("应用上下文优化完成")
        
        # 应用语义重排序（如果启用）
        if self.enable_advanced_features and self.enable_semantic_reranking and len(results) > 1:
            results = self._semantic_reranking(query, results)
            logger.info("应用语义重排序完成")
            
        # 多样性增强（如果启用）
        if self.enable_advanced_features and self.enable_source_diversification and len(results) > 3:
            results = self._diversify_results(results)
            logger.info("应用来源多样性增强完成")
        
        # 更新结果记忆
        if self.enable_context_optimization:
            if index_name not in self.result_memory:
                self.result_memory[index_name] = {}
            self.result_memory[index_name][query] = results[:top_k]
            
            # 限制记忆大小
            if len(self.result_memory[index_name]) > self.max_memory_items:
                # 移除最旧的条目
                oldest_query = next(iter(self.result_memory[index_name]))
                del self.result_memory[index_name][oldest_query]
        
        # 确保结果不超过top_k
        results = results[:top_k]
        
        # 仅保留必要字段
        results = self._process_results(results)
        
        end_time = time.time()
        logger.info(f"检索完成: 耗时={end_time - start_time:.3f}秒, 结果数={len(results)}")
        
        return results
    
    def _apply_context_optimization(
        self, 
        query: str, 
        previous_queries: List[str],
        results: List[Dict[str, Any]],
        index_name: str
    ) -> List[Dict[str, Any]]:
        """
        应用上下文优化
        
        基于之前的查询和结果调整当前结果排序
        
        Args:
            query: 当前查询
            previous_queries: 之前的查询列表
            results: 当前检索结果
            index_name: 索引名称
            
        Returns:
            优化后的结果列表
        """
        if not results or not previous_queries:
            return results
            
        # 检查是否有历史结果记录
        if index_name not in self.result_memory:
            return results
            
        # 优化后的结果
        optimized_results = results.copy()
        
        # 按照时间顺序计算加权分数（较新的查询权重更高）
        recency_weights = []
        for i in range(len(previous_queries)):
            # 从0开始，越近的查询权重越高
            recency_weights.append((i + 1) / len(previous_queries))
            
        # 计算当前查询与历史查询的相似度
        similarity_scores = []
        for prev_query in previous_queries:
            similarity = self._calculate_cosine_similarity(query, prev_query)
            similarity_scores.append(similarity)
        
        # 调整结果分数
        for i, result in enumerate(optimized_results):
            doc_id = result.get("document_id")
            if not doc_id:
                continue
                
            context_boost = 0.0
            
            # 基于查询历史计算上下文提升
            for j, prev_query in enumerate(previous_queries):
                if prev_query in self.result_memory[index_name]:
                    prev_results = self.result_memory[index_name][prev_query]
                    
                    # 检查当前文档是否在历史结果中
                    for prev_result in prev_results:
                        if prev_result.get("document_id") == doc_id:
                            # 上下文提升 = 时间权重 * 相似度 * 历史结果分数
                            recency_boost = recency_weights[j] * self.context_recency_weight
                            similarity_boost = similarity_scores[j] * self.context_similarity_weight
                            relevance_boost = prev_result.get("score", 0) * self.context_relevance_weight
                            
                            current_boost = recency_boost + similarity_boost + relevance_boost
                            context_boost = max(context_boost, current_boost)  # 取最大提升
                            break
            
            # 应用上下文提升到分数
            if context_boost > 0:
                original_score = result.get("score", 0)
                boosted_score = original_score * (1 + context_boost)
                optimized_results[i]["score"] = boosted_score
                
                # 记录提升信息（用于调试）
                optimized_results[i]["context_boost"] = {
                    "original_score": original_score,
                    "boosted_score": boosted_score,
                    "boost_factor": context_boost
                }
        
        # 重新排序
        optimized_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return optimized_results
    
    def _rank_and_filter_results(
        self, 
        results: List[Dict[str, Any]], 
        query: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        对结果进行排序和过滤
        
        Args:
            results: 检索结果列表
            query: 用户查询
            top_k: 返回数量
            
        Returns:
            排序过滤后的结果
        """
        if not results:
            return []
            
        # 复制结果避免修改原始数据
        ranked_results = results.copy()
        
        # 按分数排序
        ranked_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # 过滤低质量结果（分数低于0.2的）
        filtered_results = [r for r in ranked_results if r.get("score", 0) >= 0.2]
        
        # 如果过滤后结果太少，使用原始排序结果
        if len(filtered_results) < min(2, top_k):
            filtered_results = ranked_results
            
        # 限制结果数量
        return filtered_results[:top_k]
    
    def _process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理结果
        
        标准化和清洗结果数据
        
        Args:
            results: 检索结果列表
            
        Returns:
            处理后的结果
        """
        processed_results = []
        
        for result in results:
            # 提取必要字段
            processed_item = {
                "document_id": result.get("document_id", ""),
                "content": result.get("content", ""),
                "metadata": result.get("metadata", {}),
                "score": result.get("score", 0.0)
            }
            
            # 保留关键词匹配信息（如果有）
            if "keyword_matches" in result:
                processed_item["keyword_matches"] = result["keyword_matches"]
                
            # 保留搜索类型信息（如果有）
            if "search_type" in result:
                processed_item["search_type"] = result["search_type"]
                
            processed_results.append(processed_item)
            
        return processed_results
        
    def _calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        计算余弦相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数
        """
        # 简单分词
        def tokenize(text):
            # 转小写并分词
            text = text.lower()
            # 移除标点
            text = re.sub(r'[^\w\s]', '', text)
            # 分词
            return text.split()
            
        # 获取词汇表和词频
        def get_word_freq(tokens):
            word_freq = {}
            for token in tokens:
                if token in word_freq:
                    word_freq[token] += 1
                else:
                    word_freq[token] = 1
            return word_freq
            
        # 计算TF-IDF向量
        def get_tfidf_vector(word_freq1, word_freq2):
            # 合并词汇表
            vocabulary = sorted(list(set(list(word_freq1.keys()) + list(word_freq2.keys()))))
            
            # 初始化向量
            vector1 = np.zeros(len(vocabulary))
            vector2 = np.zeros(len(vocabulary))
            
            # 填充向量
            for i, word in enumerate(vocabulary):
                if word in word_freq1:
                    vector1[i] = word_freq1[word]
                if word in word_freq2:
                    vector2[i] = word_freq2[word]
                    
            return vector1.reshape(1, -1), vector2.reshape(1, -1)
            
        try:
            # 分词
            tokens1 = tokenize(text1)
            tokens2 = tokenize(text2)
            
            # 空文本处理
            if not tokens1 or not tokens2:
                return 0.0
                
            # 计算词频
            word_freq1 = get_word_freq(tokens1)
            word_freq2 = get_word_freq(tokens2)
            
            # 获取TF-IDF向量
            vector1, vector2 = get_tfidf_vector(word_freq1, word_freq2)
            
            # 计算余弦相似度
            similarity = cosine_similarity(vector1, vector2)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"计算相似度失败: {str(e)}")
            return 0.0
    
    def _generate_query_expansion(self, query: str) -> Tuple[str, List[str]]:
        """
        生成查询扩展
        
        基于关键词提取扩展原始查询
        
        Args:
            query: 原始查询
            
        Returns:
            扩展后的查询, 扩展关键词列表
        """
        # 如果查询太短则不扩展
        if len(query) <= 5:
            return query, []
            
        try:
            # 分词（简化版，实际系统可能使用更复杂的分词器）
            words = query.lower().split()
            
            # 去除停用词（简化版）
            stopwords = {'的', '了', '是', '和', '在', '我', '你', '他', '她', '它', '这', '那', '有', '与', '为', '吗', '呢', '什么', '如何', '怎么'}
            keywords = [word for word in words if word not in stopwords and len(word) > 1]
            
            # 如果没有提取到关键词则返回原始查询
            if not keywords:
                return query, []
                
            # 扩展查询（实际系统可能使用同义词、词干等技术）
            expanded_query = query
            expansion_keywords = []
            
            # 简单规则：为查询添加关键词
            for keyword in keywords[:2]:  # 仅使用前两个关键词
                if len(keyword) > 2 and keyword not in expansion_keywords:
                    expansion_keywords.append(keyword)
            
            # 构建扩展查询（保持原始查询不变，仅返回关键词）
            return query, expansion_keywords
            
        except Exception as e:
            logger.warning(f"查询扩展失败: {str(e)}")
            return query, []
            
    def _semantic_reranking(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        语义重排序
        
        根据内容的语义相关性对结果进行重排序
        
        Args:
            query: 查询
            results: 检索结果
            
        Returns:
            重排序后的结果
        """
        if not results or len(results) <= 1:
            return results
            
        # 复制结果以避免修改原始数据
        reranked_results = results.copy()
        
        # 计算查询与每个结果内容的语义相似度
        for i, result in enumerate(reranked_results):
            content = result.get("content", "")
            if not content:
                continue
                
            # 计算语义相似度
            semantic_score = self._calculate_cosine_similarity(query, content)
            
            # 更新分数，融合原始分数和语义分数
            original_score = result.get("score", 0)
            reranked_score = original_score * 0.7 + semantic_score * 0.3
            reranked_results[i]["score"] = reranked_score
            
            # 记录语义分数（调试用）
            reranked_results[i]["semantic_score"] = semantic_score
            
        # 重新排序
        reranked_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return reranked_results
        
    def _diversify_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        多样性增强
        
        确保结果来源的多样性
        
        Args:
            results: 检索结果
            
        Returns:
            增强多样性后的结果
        """
        if not results or len(results) <= 3:
            return results
            
        # 计算每个来源的结果数量
        source_counts = {}
        for result in results:
            metadata = result.get("metadata", {})
            source = metadata.get("source", "unknown")
            if source in source_counts:
                source_counts[source] += 1
            else:
                source_counts[source] = 1
                
        # 如果没有多个来源，返回原始结果
        if len(source_counts) <= 1:
            return results
            
        # 复制结果
        diversified_results = []
        remaining_results = []
        
        # 获取每个来源的最高分结果（前2个）
        used_doc_ids = set()
        source_used_counts = {source: 0 for source in source_counts}
        
        # 首先按分数排序
        sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        
        # 第一步：从每个来源选择至少一个结果
        for result in sorted_results:
            doc_id = result.get("document_id")
            if doc_id in used_doc_ids:
                continue
                
            metadata = result.get("metadata", {})
            source = metadata.get("source", "unknown")
            
            # 如果这个来源还没达到最大数量，加入多样化结果
            if source_used_counts[source] < 2:  # 每个来源最多2个结果
                diversified_results.append(result)
                used_doc_ids.add(doc_id)
                source_used_counts[source] += 1
            else:
                remaining_results.append(result)
        
        # 第二步：添加剩余的高分结果
        for result in remaining_results:
            doc_id = result.get("document_id")
            if doc_id not in used_doc_ids:
                diversified_results.append(result)
                used_doc_ids.add(doc_id)
                
        # 按原始分数重新排序
        diversified_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return diversified_results 