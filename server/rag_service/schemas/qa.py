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
    search_type: Optional[str] = Field("hybrid", description="搜索类型: vector, keyword, hybrid, auto")
    language: Optional[str] = Field("chinese", description="语言")
    role: Optional[str] = Field("assistant", description="角色")
    scene: Optional[Dict[str, Any]] = Field(None, description="场景信息")
    filters: Optional[Dict[str, Any]] = Field({}, description="过滤条件")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "什么是检索增强生成?",
                "knowledge_base_id": "default",
                "conversation_id": None,
                "top_k": 5,
                "search_type": "hybrid",
                "language": "chinese",
                "role": "assistant",
                "scene": None,
                "filters": {}
            }
        }


class Entity(BaseModel):
    """实体"""
    text: str = Field(..., description="实体文本")
    type: str = Field(..., description="实体类型")
    start: Optional[int] = Field(None, description="起始位置")
    end: Optional[int] = Field(None, description="结束位置")
    value: Optional[Any] = Field(None, description="实体值")


class SourceDocument(BaseModel):
    """来源文档"""
    document_id: str = Field(..., description="文档ID")
    content: str = Field(..., description="文档内容片段")
    metadata: Dict[str, Any] = Field(..., description="文档元数据")
    score: float = Field(..., description="相似度分数")
    keyword_matches: Optional[List[Dict[str, Any]]] = Field(None, description="关键词匹配信息")


class QuestionResponse(BaseModel):
    """问题响应"""
    answer: str = Field(..., description="生成的回答")
    conversation_id: str = Field(..., description="对话ID")
    sources: List[SourceDocument] = Field([], description="来源文档列表")
    recommended_questions: List[str] = Field([], description="推荐问题列表")
    thinking: Optional[str] = Field(None, description="思考过程")
    intent: Optional[str] = Field(None, description="查询意图")
    intent_confidence: Optional[float] = Field(None, description="意图置信度")
    entities: Optional[List[Entity]] = Field(None, description="实体列表")
    language: Optional[str] = Field(None, description="检测到的语言")


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


class QueryAnalysisRequest(BaseModel):
    """查询分析请求"""
    query: str = Field(..., description="用户查询")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "2023年初，中国的GDP增长了多少?"
            }
        }


class QueryAnalysisResponse(BaseModel):
    """查询分析响应"""
    query: str = Field(..., description="用户查询")
    intent: Optional[str] = Field(None, description="查询意图")
    intent_confidence: Optional[float] = Field(None, description="意图置信度")
    entities: Optional[List[Entity]] = Field([], description="实体列表")
    is_question: Optional[bool] = Field(None, description="是否为问题")
    keywords: Optional[List[str]] = Field([], description="关键词列表")
    language: Optional[str] = Field(None, description="检测到的语言") 