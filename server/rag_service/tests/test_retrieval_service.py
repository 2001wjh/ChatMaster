"""
检索服务测试模块
测试检索服务的各项功能
"""

import os
import pytest
from typing import Dict, Any, List
from server.rag_service.service.retrieval_service import RetrievalService
from server.rag_service.service.index_service import IndexService

# 测试配置
TEST_CONFIG = {
    "model_name": "gpt-3.5-turbo",
    "vector_store_type": "faiss",
    "embedding_model": "text-embedding-ada-002",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "top_k": 5,
    "enable_advanced_features": True,
    "vector_weight": 0.7,
    "keyword_weight": 0.3,
    "enable_context_optimization": True
}

@pytest.fixture
def retrieval_service():
    """创建检索服务实例"""
    return RetrievalService(TEST_CONFIG)

@pytest.fixture
def index_service():
    """创建索引服务实例"""
    return IndexService(TEST_CONFIG)

def test_retrieval_service_initialization(retrieval_service):
    """测试检索服务初始化"""
    assert retrieval_service is not None
    assert retrieval_service.config == TEST_CONFIG
    assert retrieval_service.top_k == TEST_CONFIG["top_k"]
    assert retrieval_service.vector_weight == TEST_CONFIG["vector_weight"]
    assert retrieval_service.keyword_weight == TEST_CONFIG["keyword_weight"]
    assert retrieval_service.enable_context_optimization == TEST_CONFIG["enable_context_optimization"]

def test_retrieve_with_empty_query(retrieval_service, index_service):
    """测试空查询的检索"""
    results = retrieval_service.retrieve(
        query="",
        index_service=index_service,
        index_name="test_index"
    )
    assert isinstance(results, list)
    assert len(results) == 0

def test_retrieve_with_invalid_index(retrieval_service, index_service):
    """测试无效索引的检索"""
    with pytest.raises(Exception):
        retrieval_service.retrieve(
            query="test query",
            index_service=index_service,
            index_name="non_existent_index"
        )

def test_context_optimization(retrieval_service):
    """测试上下文优化"""
    # 准备测试数据
    results = [
        {
            "content": "test content 1",
            "metadata": {"chunk_id": "1"},
            "score": 0.8
        },
        {
            "content": "test content 2",
            "metadata": {"chunk_id": "2"},
            "score": 0.6
        }
    ]
    
    current_query = "test query"
    previous_queries = ["previous query 1", "previous query 2"]
    
    # 测试上下文优化
    optimized_results = retrieval_service._apply_context_optimization(
        results=results,
        current_query=current_query,
        previous_queries=previous_queries
    )
    
    assert isinstance(optimized_results, list)
    assert len(optimized_results) == len(results)
    assert "context_score" in optimized_results[0]
    assert "score" in optimized_results[0]

def test_rank_and_filter_results(retrieval_service):
    """测试结果排序和过滤"""
    # 准备测试数据
    results = [
        {"score": 0.8, "content": "content 1"},
        {"score": 0.6, "content": "content 2"},
        {"score": 0.9, "content": "content 3"},
        {"score": 0.7, "content": "content 4"}
    ]
    
    # 测试排序和过滤
    filtered_results = retrieval_service._rank_and_filter_results(results, top_k=2)
    
    assert len(filtered_results) == 2
    assert filtered_results[0]["score"] == 0.9  # 最高分应该在第一位
    assert filtered_results[1]["score"] == 0.8  # 第二高分应该在第二位

def test_process_results(retrieval_service):
    """测试结果处理"""
    # 准备测试数据
    results = [
        {
            "content": "test content",
            "metadata": {
                "filename": "test.txt",
                "file_path": "/path/to/test.txt"
            },
            "score": 0.8
        }
    ]
    
    # 测试结果处理
    processed_results = retrieval_service._process_results(results)
    
    assert len(processed_results) == 1
    assert "rank" in processed_results[0]
    assert processed_results[0]["rank"] == 1
    assert "document_name" in processed_results[0]
    assert processed_results[0]["document_name"] == "test.txt"
    assert "document_url" in processed_results[0]
    assert processed_results[0]["document_url"] == "/path/to/test.txt"

def test_cosine_similarity(retrieval_service):
    """测试余弦相似度计算"""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    vec3 = [1.0, 1.0, 0.0]
    
    # 测试正交向量
    similarity1 = retrieval_service._cosine_similarity(vec1, vec2)
    assert similarity1 == 0.0
    
    # 测试相同向量
    similarity2 = retrieval_service._cosine_similarity(vec1, vec1)
    assert similarity2 == 1.0
    
    # 测试45度角向量
    similarity3 = retrieval_service._cosine_similarity(vec1, vec3)
    assert abs(similarity3 - 0.7071) < 0.0001  # cos(45°) ≈ 0.7071

def test_generate_query_expansion(retrieval_service):
    """测试查询扩展生成"""
    query = "test query expansion"
    
    # 测试查询扩展
    expanded_queries = retrieval_service.generate_query_expansion(query)
    
    assert isinstance(expanded_queries, list)
    assert len(expanded_queries) > 0
    assert query in expanded_queries  # 原始查询应该在扩展结果中 