"""
文档服务路由模块
提供文档处理相关的API接口
"""

import logging
import os
import uuid
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, Path

from server.rag_service.schemas.document import (
    DocumentResponse, 
    DocumentListResponse,
    DocumentMetadataResponse
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


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base_id: str = Form(..., description="知识库ID"),
    document_type: Optional[str] = Form(None, description="文档类型"),
    description: Optional[str] = Form(None, description="文档描述"),
    services=Depends(get_service)
):
    """
    上传文档
    
    上传文档并索引到指定知识库
    """
    try:
        document_service = services["document_service"]
        index_service = services["index_service"]
        
        # 保存文件
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        save_path = f"uploads/{file_id}{file_ext}"
        
        # 确保目录存在
        os.makedirs("uploads", exist_ok=True)
        
        # 保存上传的文件
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # 处理文档
        document_id = document_service.process_document(
            file_path=save_path,
            metadata={
                "document_type": document_type or "unknown",
                "filename": file.filename,
                "description": description or "",
                "original_file_id": file_id
            }
        )
        
        # 索引文档
        index_service.index_document(
            index_name=knowledge_base_id,
            document_id=document_id
        )
        
        return DocumentResponse(
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
            filename=file.filename,
            status="indexed"
        )
        
    except Exception as e:
        logger.error(f"上传文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传文档失败: {str(e)}")


@router.get("/list/{knowledge_base_id}", response_model=DocumentListResponse)
async def list_documents(
    knowledge_base_id: str = Path(..., description="知识库ID"),
    limit: int = Query(100, description="返回数量限制"),
    offset: int = Query(0, description="偏移量"),
    services=Depends(get_service)
):
    """
    获取文档列表
    
    获取指定知识库的文档列表
    """
    try:
        document_service = services["document_service"]
        
        # 获取文档列表
        documents = document_service.list_documents(
            knowledge_base_id=knowledge_base_id,
            limit=limit,
            offset=offset
        )
        
        return DocumentListResponse(
            total=len(documents),
            documents=documents
        )
        
    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@router.get("/{document_id}", response_model=DocumentMetadataResponse)
async def get_document_metadata(
    document_id: str = Path(..., description="文档ID"),
    services=Depends(get_service)
):
    """
    获取文档元数据
    
    通过文档ID获取文档的元数据信息
    """
    try:
        document_service = services["document_service"]
        
        # 获取文档元数据
        metadata = document_service.get_document_metadata(document_id)
        
        if not metadata:
            raise HTTPException(status_code=404, detail=f"文档不存在: {document_id}")
            
        return DocumentMetadataResponse(
            document_id=document_id,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档元数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文档元数据失败: {str(e)}")


@router.delete("/{document_id}", response_model=Dict[str, Any])
async def delete_document(
    document_id: str = Path(..., description="文档ID"),
    knowledge_base_id: str = Query(..., description="知识库ID"),
    services=Depends(get_service)
):
    """
    删除文档
    
    从知识库中删除指定文档
    """
    try:
        document_service = services["document_service"]
        index_service = services["index_service"]
        
        # 从索引中删除文档
        index_service.delete_document(
            index_name=knowledge_base_id,
            document_id=document_id
        )
        
        # 删除文档记录
        document_service.delete_document(document_id)
        
        return {"status": "success", "message": "文档已删除"}
        
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}") 