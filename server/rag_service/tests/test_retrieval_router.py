"""
检索服务路由测试
测试检索API的各项功能
"""

import pytest
from fastapi.testclient import TestClient

from server.rag_service.main import app
from server.rag_service.service.retrieval_service import RetrievalService
from server.rag_service.schemas.qa import SourceDocument

# 创建测试客户端
client = TestClient(app)


@pytest.fixture
def mock_service(monkeypatch):
    """模拟服务实例"""
    
    # 模拟检索结果
    def mock_retrieve(*args, **kwargs):
        return [
            SourceDocument(
                document_id="doc1",
                content="这是一个关于检索增强生成的示例文档。",
                metadata={"title": "RAG简介", "author": "测试用户"},
                score=0.95,
                keyword_matches=[{"keyword": "检索", "count": 1}]
            ),
            SourceDocument(
                document_id="doc2",
                content="向量检索和关键词检索可以结合使用。",
                metadata={"title": "混合检索", "author": "测试用户"},
                score=0.85,
                keyword_matches=[{"keyword": "检索", "count": 2}]
            )
        ]
    
    # 模拟上下文优化
    def mock_apply_context_optimization(*args, **kwargs):
        results = kwargs.get("results") or args[0]
        # 简单返回输入结果
        return results
    
    # 模拟查询扩展
    def mock_generate_query_expansion(*args, **kwargs):
        query = kwargs.get("query") or args[0]
        return [
            f"{query} 的定义",
            f"{query} 的应用",
            f"{query} 的示例"
        ]
    
    # 创建模拟服务
    mock_retrieval_service = RetrievalService(
        vector_search_weight=0.7,
        keyword_search_weight=0.3,
        enable_context_optimization=True,
        enable_query_expansion=True
    )
    
    # 替换方法
    monkeypatch.setattr(mock_retrieval_service, "retrieve", mock_retrieve)
    monkeypatch.setattr(mock_retrieval_service, "_apply_context_optimization", mock_apply_context_optimization)
    monkeypatch.setattr(mock_retrieval_service, "generate_query_expansion", mock_generate_query_expansion)
    
    # 替换服务获取函数
    def get_service():
        return {
            "retrieval_service": mock_retrieval_service,
            "text_processing_service": None,
            "index_service": None
        }
    
    from server.rag_service.router import retrieval_router
    monkeypatch.setattr(retrieval_router, "get_service", get_service)


def test_retrieve(mock_service):
    """测试检索接口"""
    response = client.post(
        "/api/retrieval/retrieve",
        json={
            "query": "检索增强生成是什么?",
            "index_name": "test_index",
            "top_k": 5,
            "search_type": "hybrid"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["query"] == "检索增强生成是什么?"
    assert len(data["results"]) == 2
    assert data["results"][0]["document_id"] == "doc1"
    assert data["results"][1]["document_id"] == "doc2"
    assert data["search_type"] == "hybrid"


def test_optimize_context(mock_service):
    """测试上下文优化接口"""
    # 首先获取检索结果
    retrieve_response = client.post(
        "/api/retrieval/retrieve",
        json={
            "query": "检索增强生成是什么?",
            "index_name": "test_index",
            "top_k": 5,
            "search_type": "hybrid"
        }
    )
    
    retrieve_data = retrieve_response.json()
    
    # 然后测试上下文优化
    response = client.post(
        "/api/retrieval/optimize_context",
        json={
            "query": "检索增强生成是什么?",
            "results": retrieve_data["results"],
            "previous_queries": ["什么是RAG?"],
            "optimization_type": "rerank"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["query"] == "检索增强生成是什么?"
    assert len(data["optimized_results"]) == 2
    assert data["original_results"] == data["optimized_results"]
    assert data["optimization_type"] == "rerank"


def test_expand_query(mock_service):
    """测试查询扩展接口"""
    response = client.post(
        "/api/retrieval/expand_query",
        json={
            "query": "检索增强生成",
            "num_variations": 3
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["original_query"] == "检索增强生成"
    assert len(data["expanded_queries"]) == 3
    assert "检索增强生成 的定义" in data["expanded_queries"]
    assert "检索增强生成 的应用" in data["expanded_queries"]
    assert "检索增强生成 的示例" in data["expanded_queries"] 