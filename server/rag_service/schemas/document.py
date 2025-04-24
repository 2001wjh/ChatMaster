"""
文档服务数据模型
定义文档处理API的请求和响应类型
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentResponse(BaseModel):
    """文档响应"""
    document_id: str = Field(..., description="文档ID")
    knowledge_base_id: str = Field(..., description="知识库ID")
    filename: str = Field(..., description="文件名")
    status: str = Field(..., description="处理状态")


class DocumentInfo(BaseModel):
    """文档信息"""
    document_id: str = Field(..., description="文档ID")
    filename: str = Field(..., description="文件名")
    document_type: str = Field(..., description="文档类型")
    description: Optional[str] = Field(None, description="文档描述")
    status: str = Field(..., description="处理状态")
    created_at: str = Field(..., description="创建时间")
    chunks_count: int = Field(0, description="分块数量")


class DocumentListResponse(BaseModel):
    """文档列表响应"""
    total: int = Field(..., description="文档总数")
    documents: List[DocumentInfo] = Field([], description="文档列表")


class DocumentMetadataResponse(BaseModel):
    """文档元数据响应"""
    document_id: str = Field(..., description="文档ID")
    metadata: Dict[str, Any] = Field(..., description="文档元数据") 