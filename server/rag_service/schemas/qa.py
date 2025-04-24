"""
问答服务数据模型
定义问答API的请求和响应类型
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """问题请求"""
    question: str = Field(..., description="用户问题")
    knowledge_base_id: str = Field(..., description="知识库ID")
    conversation_id: Optional[str] = Field(None, description="对话ID，为空则创建新对话")
    top_k: Optional[int] = Field(5, description="返回的结果数量")
    search_type: Optional[str] = Field("hybrid", description="搜索类型: vector, keyword, hybrid")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "什么是检索增强生成?",
                "knowledge_base_id": "default",
                "conversation_id": None,
                "top_k": 5,
                "search_type": "hybrid"
            }
        }


class SourceDocument(BaseModel):
    """来源文档"""
    document_id: str = Field(..., description="文档ID")
    content: str = Field(..., description="文档内容片段")
    metadata: Dict[str, Any] = Field(..., description="文档元数据")
    score: float = Field(..., description="相似度分数")


class QuestionResponse(BaseModel):
    """问题响应"""
    answer: str = Field(..., description="生成的回答")
    conversation_id: str = Field(..., description="对话ID")
    sources: List[SourceDocument] = Field([], description="来源文档列表")
    recommended_questions: List[str] = Field([], description="推荐问题列表")


class FeedbackRequest(BaseModel):
    """反馈请求"""
    conversation_id: str = Field(..., description="对话ID")
    question: str = Field(..., description="用户问题")
    answer: str = Field(..., description="生成的回答")
    feedback: bool = Field(..., description="是否有帮助")
    feedback_text: Optional[str] = Field(None, description="反馈文本")


class RecommendedQuestionResponse(BaseModel):
    """推荐问题响应"""
    questions: List[str] = Field([], description="推荐问题列表") 