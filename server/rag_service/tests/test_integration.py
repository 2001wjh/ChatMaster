"""
集成测试模块
测试检索服务与其他服务的交互
"""

import os
import pytest
from typing import Dict, Any, List
from server.rag_service.service.retrieval_service import RetrievalService
from server.rag_service.service.index_service import IndexService
from server.rag_service.service.qa_service import QAService

def test_retrieval_with_qa_service(test_config, test_index_name, test_document_path):
    """测试检索服务与问答服务的集成"""
    # 初始化服务
    retrieval_service = RetrievalService(test_config)
    index_service = IndexService(test_config)
    qa_service = QAService(test_config)
    
    # 创建测试索引
    index_service.create_index(test_index_name)
    
    # 添加测试文档
    index_service.add_document(test_index_name, test_document_path)
    
    # 执行检索
    query = "什么是人工智能？"
    results = retrieval_service.retrieve(
        query=query,
        index_service=index_service,
        index_name=test_index_name
    )
    
    # 验证检索结果
    assert len(results) > 0
    assert "content" in results[0]
    assert "score" in results[0]
    
    # 使用检索结果生成回答
    answer_result = qa_service.answer(
        query=query,
        context=results,
        conversation_id="test_conversation",
        retrieval_service=retrieval_service,
        index_service=index_service,
        index_name=test_index_name
    )
    
    # 验证回答结果
    assert "answer" in answer_result
    assert len(answer_result["answer"]) > 0
    assert "retrieval_results" in answer_result
    assert len(answer_result["retrieval_results"]) > 0

def test_context_optimization_with_qa_service(test_config, test_index_name, test_document_path):
    """测试上下文优化与问答服务的集成"""
    # 初始化服务
    retrieval_service = RetrievalService(test_config)
    index_service = IndexService(test_config)
    qa_service = QAService(test_config)
    
    # 创建测试索引
    index_service.create_index(test_index_name)
    
    # 添加测试文档
    index_service.add_document(test_index_name, test_document_path)
    
    # 模拟对话历史
    conversation_id = "test_conversation"
    previous_queries = [
        "什么是机器学习？",
        "深度学习和机器学习有什么关系？"
    ]
    
    # 执行带上下文的检索
    query = "自然语言处理有哪些应用？"
    results = retrieval_service.retrieve(
        query=query,
        index_service=index_service,
        index_name=test_index_name,
        previous_queries=previous_queries
    )
    
    # 验证检索结果
    assert len(results) > 0
    assert "content" in results[0]
    assert "score" in results[0]
    
    # 使用检索结果生成回答
    answer_result = qa_service.answer(
        query=query,
        context=results,
        conversation_id=conversation_id,
        retrieval_service=retrieval_service,
        index_service=index_service,
        index_name=test_index_name
    )
    
    # 验证回答结果
    assert "answer" in answer_result
    assert len(answer_result["answer"]) > 0
    assert "retrieval_results" in answer_result
    assert len(answer_result["retrieval_results"]) > 0

def test_query_expansion_with_qa_service(test_config, test_index_name, test_document_path):
    """测试查询扩展与问答服务的集成"""
    # 初始化服务
    retrieval_service = RetrievalService(test_config)
    index_service = IndexService(test_config)
    qa_service = QAService(test_config)
    
    # 创建测试索引
    index_service.create_index(test_index_name)
    
    # 添加测试文档
    index_service.add_document(test_index_name, test_document_path)
    
    # 生成查询扩展
    query = "计算机视觉的应用"
    expanded_queries = retrieval_service.generate_query_expansion(query)
    
    # 验证查询扩展
    assert len(expanded_queries) > 0
    assert query in expanded_queries
    
    # 使用扩展查询执行检索
    all_results = []
    for expanded_query in expanded_queries:
        results = retrieval_service.retrieve(
            query=expanded_query,
            index_service=index_service,
            index_name=test_index_name
        )
        all_results.extend(results)
    
    # 去重和排序
    unique_results = retrieval_service._rank_and_filter_results(all_results, top_k=5)
    
    # 验证检索结果
    assert len(unique_results) > 0
    assert "content" in unique_results[0]
    assert "score" in unique_results[0]
    
    # 使用检索结果生成回答
    answer_result = qa_service.answer(
        query=query,
        context=unique_results,
        conversation_id="test_conversation",
        retrieval_service=retrieval_service,
        index_service=index_service,
        index_name=test_index_name
    )
    
    # 验证回答结果
    assert "answer" in answer_result
    assert len(answer_result["answer"]) > 0
    assert "retrieval_results" in answer_result
    assert len(answer_result["retrieval_results"]) > 0 