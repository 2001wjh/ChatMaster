"""
知识库服务数据模型
定义知识库管理API的请求和响应类型
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class KnowledgeBaseRequest(BaseModel):
    """知识库请求"""
    id: str = Field(..., description="知识库ID")
    name: str = Field(..., description="知识库名称")
    description: Optional[str] = Field("", description="知识库描述")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "company_kb",
                "name": "公司知识库",
                "description": "包含公司产品、服务和政策的知识库"
            }
        }


class KnowledgeBaseResponse(BaseModel):
    """知识库响应"""
    id: str = Field(..., description="知识库ID")
    name: str = Field(..., description="知识库名称")
    description: str = Field("", description="知识库描述")
    created_at: str = Field(..., description="创建时间")


class KnowledgeBaseListResponse(BaseModel):
    """知识库列表响应"""
    total: int = Field(..., description="知识库总数")
    knowledge_bases: List[KnowledgeBaseResponse] = Field([], description="知识库列表") 