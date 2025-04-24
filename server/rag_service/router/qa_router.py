"""
问答服务路由模块
提供与检索式问答相关的API接口
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body

from server.rag_service.schemas.qa import (
    QuestionRequest, 
    QuestionResponse, 
    FeedbackRequest, 
    RecommendedQuestionResponse,
    QueryAnalysisRequest,
    QueryAnalysisResponse
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


@router.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    services=Depends(get_service)
):
    """
    问答接口
    
    根据用户提问和指定的知识库返回答案
    """
    try:
        qa_service = services["qa_service"]
        retrieval_service = services["retrieval_service"]
        index_service = services["index_service"]
        text_processing_service = services["text_processing_service"]
        
        # 生成对话ID
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # 获取查询历史(如果有)
        previous_queries = qa_service.query_history.get(conversation_id, [])
        
        # 识别意图和实体
        intent = None
        intent_confidence = 0.0
        entities = []
        
        try:
            # 意图识别
            intent, intent_confidence = text_processing_service._recognize_intent(request.question)
            logger.info(f"查询意图: {intent}, 置信度: {intent_confidence}")
            
            # 实体提取
            entities = text_processing_service._extract_entities(request.question)
            logger.info(f"提取实体: {entities}")
            
        except Exception as e:
            logger.warning(f"意图和实体识别失败: {str(e)}")
        
        # 构建过滤条件
        filters = {}
        if entities:
            for entity in entities:
                if entity["type"] == "TIME" and "time" in request.filters:
                    if "time_range" not in filters:
                        filters["time_range"] = []
                    filters["time_range"].append(entity["text"])
                elif entity["type"] == "PERSON" and "author" in request.filters:
                    if "author" not in filters:
                        filters["author"] = []
                    filters["author"].append(entity["text"])
                elif entity["type"] == "LOCATION" and "location" in request.filters:
                    if "location" not in filters:
                        filters["location"] = []
                    filters["location"].append(entity["text"])
        
        # 执行检索
        search_type = request.search_type
        
        # 根据意图调整检索类型（如果用户未明确指定）
        if intent and (search_type == "auto" or not search_type):
            if intent == "command":
                search_type = "keyword"
            elif intent == "learning_question":
                search_type = "hybrid"
            elif intent == "information_seeking":
                search_type = "vector"
            else:
                search_type = "hybrid"  # 默认使用混合检索
        
        search_results = retrieval_service.retrieve(
            query=request.question,
            index_service=index_service,
            index_name=request.knowledge_base_id,
            top_k=request.top_k,
            search_type=search_type,
            previous_queries=previous_queries,
            filters=filters
        )
        
        # 更新查询历史
        if conversation_id not in qa_service.query_history:
            qa_service.query_history[conversation_id] = []
        qa_service.query_history[conversation_id].append(request.question)
        
        # 生成回答
        result = qa_service.answer(
            query=request.question,
            context=search_results,
            conversation_id=conversation_id,
            language=request.language,
            role=request.role,
            scene=request.scene,
            retrieval_service=retrieval_service,
            index_service=index_service,
            index_name=request.knowledge_base_id
        )
        
        # 获取意图和实体（如果QA服务没有返回）
        if "intent" not in result or not result["intent"]:
            result["intent"] = intent
        if "entities" not in result or not result["entities"]:
            result["entities"] = entities
            
        return QuestionResponse(
            answer=result["answer"],
            conversation_id=conversation_id,
            sources=result.get("retrieval_results", []),
            recommended_questions=result.get("recommended_questions", []),
            thinking=result.get("thinking", ""),
            intent=result.get("intent", ""),
            intent_confidence=intent_confidence,
            entities=result.get("entities", []),
            language=result.get("language", request.language)
        )
        
    except Exception as e:
        logger.error(f"问答失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")


@router.post("/feedback", response_model=Dict[str, Any])
async def submit_feedback(
    request: FeedbackRequest,
    services=Depends(get_service)
):
    """
    提交反馈
    
    用户对回答的评价反馈
    """
    try:
        qa_service = services["qa_service"]
        
        # 处理反馈
        qa_service.process_feedback(
            conversation_id=request.conversation_id,
            query=request.question,
            answer=request.answer,
            feedback=request.feedback,
            feedback_text=request.feedback_text
        )
        
        return {"status": "success", "message": "反馈已提交"}
        
    except Exception as e:
        logger.error(f"提交反馈失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")


@router.post("/clear_history/{conversation_id}", response_model=Dict[str, Any])
async def clear_history(
    conversation_id: str = Path(..., description="对话ID"),
    services=Depends(get_service)
):
    """
    清除会话历史
    
    清除指定ID的对话历史记录
    """
    try:
        qa_service = services["qa_service"]
        
        # 清除历史
        qa_service.clear_history(conversation_id)
        
        # 清除查询历史
        if conversation_id in qa_service.query_history:
            del qa_service.query_history[conversation_id]
        
        return {"status": "success", "message": "历史记录已清除"}
        
    except Exception as e:
        logger.error(f"清除历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清除历史失败: {str(e)}")


@router.get("/recommended_questions", response_model=RecommendedQuestionResponse)
async def get_recommended_questions(
    query: str = Query(..., description="用户提问"),
    knowledge_base_id: str = Query(..., description="知识库ID"),
    top_k: int = Query(3, description="推荐问题数量"),
    services=Depends(get_service)
):
    """
    获取推荐问题
    
    根据用户提问和知识库内容生成推荐问题
    """
    try:
        qa_service = services["qa_service"]
        retrieval_service = services["retrieval_service"]
        index_service = services["index_service"]
        
        # 执行检索
        search_results = retrieval_service.retrieve(
            query=query,
            index_service=index_service,
            index_name=knowledge_base_id,
            top_k=top_k
        )
        
        # 生成推荐问题
        recommended_questions = qa_service.generate_recommended_questions(
            user_question=query,
            model_answer="",
            retrieved_content=search_results,
            num_questions=top_k
        )
        
        return RecommendedQuestionResponse(
            questions=recommended_questions
        )
        
    except Exception as e:
        logger.error(f"获取推荐问题失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取推荐问题失败: {str(e)}")


@router.post("/analyze_query", response_model=QueryAnalysisResponse)
async def analyze_query(
    request: QueryAnalysisRequest,
    services=Depends(get_service)
):
    """
    分析查询
    
    分析用户查询的意图和实体
    """
    try:
        text_processing_service = services["text_processing_service"]
        
        # 意图识别
        intent, confidence = text_processing_service._recognize_intent(request.query)
        
        # 实体提取
        entities = text_processing_service._extract_entities(request.query)
        
        # 检查是否为问题
        is_question = text_processing_service._is_question(request.query)
        
        # 提取关键词
        keywords = []
        if hasattr(text_processing_service, "_extract_keywords"):
            keywords = text_processing_service._extract_keywords(
                text=request.query,
                num_keywords=5
            )
        
        # 检测语言
        language = text_processing_service._detect_language(request.query)
        
        return QueryAnalysisResponse(
            query=request.query,
            intent=intent,
            intent_confidence=confidence,
            entities=entities,
            is_question=is_question,
            keywords=keywords,
            language=language
        )
        
    except Exception as e:
        logger.error(f"分析查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分析查询失败: {str(e)}") 