"""
检索服务数据模型
定义检索API的请求和响应类型
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

from server.rag_service.schemas.qa import SourceDocument


class RetrievalRequest(BaseModel):
    """检索请求"""
    query: str = Field(..., description="用户查询")
    index_name: str = Field(..., description="索引名称/知识库ID")
    top_k: int = Field(5, description="返回结果数量")
    search_type: str = Field("hybrid", description="搜索类型: vector, keyword, hybrid, auto")
    previous_queries: Optional[List[str]] = Field(None, description="历史查询列表")
    filters: Optional[Dict[str, Any]] = Field({}, description="过滤条件")
    hybrid_alpha: Optional[float] = Field(None, description="混合检索权重(0-1之间，越大向量检索权重越高)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "检索增强生成是什么?",
                "index_name": "default",
                "top_k": 5,
                "search_type": "hybrid",
                "previous_queries": [],
                "filters": {},
                "hybrid_alpha": 0.7
            }
        }


class RetrievalResponse(BaseModel):
    """检索响应"""
    query: str = Field(..., description="用户查询")
    results: List[SourceDocument] = Field([], description="检索结果列表")
    search_type: str = Field(..., description="实际使用的搜索类型")


class OptimizationRequest(BaseModel):
    """上下文优化请求"""
    query: str = Field(..., description="用户查询")
    results: List[SourceDocument] = Field(..., description="原始检索结果")
    previous_queries: Optional[List[str]] = Field(None, description="历史查询列表")
    optimization_type: str = Field("rerank", description="优化类型: rerank, cluster, diversity")
    context_window: Optional[int] = Field(None, description="上下文窗口大小")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "检索增强生成是什么?",
                "results": [],
                "previous_queries": [],
                "optimization_type": "rerank",
                "context_window": 3
            }
        }


class OptimizationResponse(BaseModel):
    """上下文优化响应"""
    query: str = Field(..., description="用户查询")
    original_results: List[SourceDocument] = Field(..., description="原始检索结果")
    optimized_results: List[SourceDocument] = Field(..., description="优化后的检索结果")
    optimization_type: str = Field(..., description="使用的优化类型")


class QueryExpansionRequest(BaseModel):
    """查询扩展请求"""
    query: str = Field(..., description="原始查询")
    num_variations: int = Field(3, description="生成扩展查询的数量")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "检索增强生成是什么?",
                "num_variations": 3
            }
        }


class QueryExpansionResponse(BaseModel):
    """查询扩展响应"""
    original_query: str = Field(..., description="原始查询")
    expanded_queries: List[str] = Field(..., description="扩展后的查询列表") 