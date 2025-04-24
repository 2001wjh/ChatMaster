"""
场景数据模型
定义场景相关的数据结构和验证规则
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class SceneBase(BaseModel):
    """场景基本信息"""
    name: str = Field(..., description="场景名称")
    description: str = Field(..., description="场景描述")
    difficulty: str = Field(..., description="难度级别：beginner, intermediate, advanced")
    category: str = Field(..., description="场景类别：daily, travel, business, academic, career, healthcare, other")
    language: str = Field("english", description="场景语言：english, chinese")
    example_topics: List[str] = Field([], description="示例主题")
    example_questions: List[str] = Field([], description="示例问题")


class SceneCreate(SceneBase):
    """创建场景"""
    id: Optional[str] = Field(None, description="场景ID，为空则自动生成")
    source_document: Optional[str] = Field(None, description="来源文档")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "旅游场景",
                "description": "在旅游景点和酒店的对话场景",
                "difficulty": "intermediate",
                "category": "travel",
                "language": "english",
                "example_topics": ["hotel", "sightseeing", "transportation"],
                "example_questions": [
                    "Can you recommend a good hotel?",
                    "How do I get to the museum?"
                ]
            }
        }


class SceneUpdate(BaseModel):
    """更新场景"""
    name: Optional[str] = Field(None, description="场景名称")
    description: Optional[str] = Field(None, description="场景描述")
    difficulty: Optional[str] = Field(None, description="难度级别：beginner, intermediate, advanced")
    category: Optional[str] = Field(None, description="场景类别：daily, travel, business, academic, career, healthcare, other")
    language: Optional[str] = Field(None, description="场景语言：english, chinese")
    example_topics: Optional[List[str]] = Field(None, description="示例主题")
    example_questions: Optional[List[str]] = Field(None, description="示例问题")


class Scene(SceneBase):
    """场景完整信息"""
    id: str = Field(..., description="场景ID")
    created_at: Optional[str] = Field(None, description="创建时间")
    updated_at: Optional[str] = Field(None, description="更新时间")
    source_document: Optional[str] = Field(None, description="来源文档")

    class Config:
        orm_mode = True


class SceneFilter(BaseModel):
    """场景筛选条件"""
    difficulty: Optional[List[str]] = Field(None, description="难度级别列表")
    category: Optional[List[str]] = Field(None, description="场景类别列表")
    language: Optional[List[str]] = Field(None, description="语言列表")
    keywords: Optional[str] = Field(None, description="关键词搜索")


class SceneCategory(BaseModel):
    """场景类别"""
    id: str = Field(..., description="类别ID")
    name: str = Field(..., description="类别名称")
    description: str = Field(..., description="类别描述")


class DifficultyLevel(BaseModel):
    """难度级别"""
    id: str = Field(..., description="级别ID")
    name: str = Field(..., description="级别名称")
    description: str = Field(..., description="级别描述") 