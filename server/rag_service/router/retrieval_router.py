"""
检索服务路由模块
提供与检索服务相关的API接口
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from server.rag_service.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    OptimizationRequest,
    OptimizationResponse,
    QueryExpansionRequest,
    QueryExpansionResponse
)
from server.rag_service.service import init_rag_service

# 设置日志
logger = logging.getLogger(__name__)

# 创建路由
router = APIRouter()

# 服务实例(延迟加载)
_service_instance = None

def get_service():
    """获取服务实例(单例模式)"""
    global _service_instance
    if _service_instance is None:
        _service_instance = init_rag_service()
    return _service_instance


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(
    request: RetrievalRequest,
    services=Depends(get_service)
):
    """
    内容检索接口
    
    根据用户查询和知识库ID检索相关内容
    """
    try:
        retrieval_service = services["retrieval_service"]
        index_service = services["index_service"]
        
        # 获取查询历史(如果有)
        previous_queries = request.previous_queries or []
        
        # 执行检索
        search_results = retrieval_service.retrieve(
            query=request.query,
            index_service=index_service,
            index_name=request.index_name,
            top_k=request.top_k,
            search_type=request.search_type,
            previous_queries=previous_queries,
            filters=request.filters,
            hybrid_alpha=request.hybrid_alpha
        )
        
        return RetrievalResponse(
            query=request.query,
            results=search_results,
            search_type=request.search_type
        )
        
    except Exception as e:
        logger.error(f"检索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@router.post("/optimize_context", response_model=OptimizationResponse)
async def optimize_context(
    request: OptimizationRequest,
    services=Depends(get_service)
):
    """
    上下文优化接口
    
    对检索结果进行上下文优化处理
    """
    try:
        retrieval_service = services["retrieval_service"]
        
        # 执行上下文优化
        optimized_results = retrieval_service._apply_context_optimization(
            results=request.results,
            query=request.query,
            previous_queries=request.previous_queries or [],
            optimization_type=request.optimization_type,
            context_window=request.context_window
        )
        
        return OptimizationResponse(
            query=request.query,
            original_results=request.results,
            optimized_results=optimized_results,
            optimization_type=request.optimization_type
        )
        
    except Exception as e:
        logger.error(f"上下文优化失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上下文优化失败: {str(e)}")


@router.post("/expand_query", response_model=QueryExpansionResponse)
async def expand_query(
    request: QueryExpansionRequest,
    services=Depends(get_service)
):
    """
    查询扩展接口
    
    生成原始查询的扩展查询
    """
    try:
        retrieval_service = services["retrieval_service"]
        text_processing_service = services["text_processing_service"]
        
        # 执行查询扩展
        expanded_queries = retrieval_service.generate_query_expansion(
            query=request.query,
            text_processing_service=text_processing_service,
            num_variations=request.num_variations
        )
        
        return QueryExpansionResponse(
            original_query=request.query,
            expanded_queries=expanded_queries
        )
        
    except Exception as e:
        logger.error(f"查询扩展失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询扩展失败: {str(e)}") 