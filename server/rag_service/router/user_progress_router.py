"""
用户进度追踪服务路由模块
提供与用户学习进度相关的API接口
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Body

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


@router.get("/progress/{user_id}", response_model=Dict[str, Any])
async def get_user_progress(
    user_id: str = Path(..., description="用户ID"),
    services=Depends(get_service)
):
    """
    获取用户进度数据
    
    根据用户ID返回其学习进度数据
    """
    try:
        progress_service = services["progress_tracking_service"]
        
        # 获取用户进度
        progress_data = progress_service.get_user_progress(user_id)
        
        return progress_data
        
    except Exception as e:
        logger.error(f"获取用户进度失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取用户进度失败: {str(e)}")


@router.post("/progress/{user_id}/session", response_model=Dict[str, Any])
async def record_session(
    user_id: str = Path(..., description="用户ID"),
    session_data: Dict[str, Any] = Body(..., description="会话数据"),
    services=Depends(get_service)
):
    """
    记录会话数据
    
    记录用户的学习会话数据，更新进度
    """
    try:
        progress_service = services["progress_tracking_service"]
        
        # 记录会话
        session_id = progress_service.record_session(user_id, session_data)
        
        return {
            "status": "success",
            "session_id": session_id,
            "message": "会话记录已保存"
        }
        
    except Exception as e:
        logger.error(f"记录会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"记录会话失败: {str(e)}")


@router.get("/progress/{user_id}/learning_path", response_model=Dict[str, Any])
async def get_learning_path(
    user_id: str = Path(..., description="用户ID"),
    services=Depends(get_service)
):
    """
    获取学习路径
    
    获取为用户推荐的学习路径和建议
    """
    try:
        progress_service = services["progress_tracking_service"]
        
        # 获取学习路径
        learning_path = progress_service.get_learning_path(user_id)
        
        return learning_path
        
    except Exception as e:
        logger.error(f"获取学习路径失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取学习路径失败: {str(e)}")


@router.get("/progress/{user_id}/statistics", response_model=Dict[str, Any])
async def get_user_statistics(
    user_id: str = Path(..., description="用户ID"),
    services=Depends(get_service)
):
    """
    获取用户统计数据
    
    获取用户的学习统计数据
    """
    try:
        progress_service = services["progress_tracking_service"]
        
        # 获取用户进度
        progress_data = progress_service.get_user_progress(user_id)
        
        # 提取统计数据
        statistics = progress_data.get("statistics", {})
        
        return statistics
        
    except Exception as e:
        logger.error(f"获取用户统计数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取用户统计数据失败: {str(e)}")


@router.get("/progress/{user_id}/recommendations", response_model=List[Dict[str, Any]])
async def get_user_recommendations(
    user_id: str = Path(..., description="用户ID"),
    services=Depends(get_service)
):
    """
    获取用户推荐
    
    获取为用户生成的学习建议
    """
    try:
        progress_service = services["progress_tracking_service"]
        
        # 获取用户进度
        progress_data = progress_service.get_user_progress(user_id)
        
        # 提取推荐
        recommendations = progress_data.get("recommendations", [])
        
        return recommendations
        
    except Exception as e:
        logger.error(f"获取用户推荐失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取用户推荐失败: {str(e)}")


@router.get("/progress/{user_id}/skills", response_model=Dict[str, Any])
async def get_user_skills(
    user_id: str = Path(..., description="用户ID"),
    services=Depends(get_service)
):
    """
    获取用户技能数据
    
    获取用户的技能评分和进步情况
    """
    try:
        progress_service = services["progress_tracking_service"]
        
        # 获取用户进度
        progress_data = progress_service.get_user_progress(user_id)
        
        # 提取技能数据
        skills = progress_data.get("skills", {})
        
        return skills
        
    except Exception as e:
        logger.error(f"获取用户技能数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取用户技能数据失败: {str(e)}") 