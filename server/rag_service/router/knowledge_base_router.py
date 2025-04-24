"""
知识库服务路由模块
提供知识库管理相关的API接口
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body

from server.rag_service.schemas.knowledge_base import (
    KnowledgeBaseRequest,
    KnowledgeBaseResponse,
    KnowledgeBaseListResponse
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


@router.post("", response_model=KnowledgeBaseResponse)
async def create_knowledge_base(
    request: KnowledgeBaseRequest,
    services=Depends(get_service)
):
    """
    创建知识库
    
    创建新的知识库
    """
    try:
        kb_service = services["knowledge_base_service"]
        index_service = services["index_service"]
        
        # 判断知识库是否存在
        if index_service.exists(request.id):
            raise HTTPException(status_code=400, detail=f"知识库已存在: {request.id}")
            
        # 创建知识库
        index_service.create_index(
            index_name=request.id,
            description=request.description
        )
        
        # 创建知识库元数据
        kb_service.create_knowledge_base({
            "id": request.id,
            "name": request.name,
            "description": request.description,
            "created_at": "2023-09-01T12:00:00Z"  # 实际应用中使用真实时间
        })
        
        return KnowledgeBaseResponse(
            id=request.id,
            name=request.name,
            description=request.description,
            created_at="2023-09-01T12:00:00Z"  # 实际应用中使用真实时间
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建知识库失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建知识库失败: {str(e)}")


@router.get("", response_model=KnowledgeBaseListResponse)
async def list_knowledge_bases(
    services=Depends(get_service)
):
    """
    获取知识库列表
    
    返回所有可用的知识库列表
    """
    try:
        kb_service = services["knowledge_base_service"]
        
        # 获取知识库列表
        knowledge_bases = kb_service.list_knowledge_bases()
        
        return KnowledgeBaseListResponse(
            total=len(knowledge_bases),
            knowledge_bases=knowledge_bases
        )
        
    except Exception as e:
        logger.error(f"获取知识库列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取知识库列表失败: {str(e)}")


@router.get("/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    knowledge_base_id: str = Path(..., description="知识库ID"),
    services=Depends(get_service)
):
    """
    获取知识库详情
    
    通过ID获取知识库详细信息
    """
    try:
        kb_service = services["knowledge_base_service"]
        
        # 获取知识库信息
        kb_info = kb_service.get_knowledge_base(knowledge_base_id)
        
        if not kb_info:
            raise HTTPException(status_code=404, detail=f"知识库不存在: {knowledge_base_id}")
            
        return KnowledgeBaseResponse(
            id=kb_info["id"],
            name=kb_info["name"],
            description=kb_info.get("description", ""),
            created_at=kb_info.get("created_at", "")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取知识库详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取知识库详情失败: {str(e)}")


@router.delete("/{knowledge_base_id}", response_model=Dict[str, Any])
async def delete_knowledge_base(
    knowledge_base_id: str = Path(..., description="知识库ID"),
    services=Depends(get_service)
):
    """
    删除知识库
    
    删除指定知识库及其所有文档
    """
    try:
        kb_service = services["knowledge_base_service"]
        index_service = services["index_service"]
        
        # 检查知识库是否存在
        if not index_service.exists(knowledge_base_id):
            raise HTTPException(status_code=404, detail=f"知识库不存在: {knowledge_base_id}")
            
        # 删除知识库
        index_service.delete_index(knowledge_base_id)
        
        # 删除知识库元数据
        kb_service.delete_knowledge_base(knowledge_base_id)
        
        return {"status": "success", "message": "知识库已删除"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除知识库失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除知识库失败: {str(e)}")


@router.post("/{knowledge_base_id}/reindex", response_model=Dict[str, Any])
async def reindex_knowledge_base(
    knowledge_base_id: str = Path(..., description="知识库ID"),
    services=Depends(get_service)
):
    """
    重建知识库索引
    
    重新索引知识库中的所有文档
    """
    try:
        index_service = services["index_service"]
        
        # 检查知识库是否存在
        if not index_service.exists(knowledge_base_id):
            raise HTTPException(status_code=404, detail=f"知识库不存在: {knowledge_base_id}")
            
        # 重建索引
        index_service.rebuild_index(knowledge_base_id)
        
        return {"status": "success", "message": "知识库索引已重建"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重建知识库索引失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重建知识库索引失败: {str(e)}") 