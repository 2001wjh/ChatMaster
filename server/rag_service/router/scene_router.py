"""
场景路由模块
提供与对话场景相关的API接口
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, UploadFile, File, Form

from server.rag_service.schemas.scene import (
    Scene, 
    SceneCreate, 
    SceneUpdate, 
    SceneFilter,
    SceneCategory,
    DifficultyLevel
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


@router.get("/", response_model=List[Scene])
async def get_scenes(
    difficulty: Optional[str] = Query(None, description="难度等级筛选"),
    category: Optional[str] = Query(None, description="类别筛选"),
    language: Optional[str] = Query(None, description="语言筛选"),
    keywords: Optional[str] = Query(None, description="关键词搜索"),
    services = Depends(get_service)
):
    """
    获取场景列表
    
    根据筛选条件返回场景列表
    """
    try:
        scene_service = services["scene_service"]
        
        # 构建筛选条件
        criteria = {}
        if difficulty:
            criteria["difficulty"] = difficulty
        if category:
            criteria["category"] = category
        if language:
            criteria["language"] = language
        if keywords:
            criteria["keywords"] = keywords
        
        # 获取场景
        scenes = scene_service.filter_scenes(criteria)
        
        return scenes
        
    except Exception as e:
        logger.error(f"获取场景列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取场景列表失败: {str(e)}")


@router.get("/{scene_id}", response_model=Scene)
async def get_scene(
    scene_id: str = Path(..., description="场景ID"),
    services = Depends(get_service)
):
    """
    获取场景详情
    
    根据场景ID返回场景详情
    """
    try:
        scene_service = services["scene_service"]
        
        # 获取场景
        scene = scene_service.get_scene_by_id(scene_id)
        
        if not scene:
            raise HTTPException(status_code=404, detail=f"场景不存在: {scene_id}")
        
        return scene
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取场景失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取场景失败: {str(e)}")


@router.post("/", response_model=Scene)
async def create_scene(
    scene_data: SceneCreate,
    services = Depends(get_service)
):
    """
    创建场景
    
    创建新的对话场景
    """
    try:
        scene_service = services["scene_service"]
        
        # 创建场景
        scene = scene_service.create_scene(scene_data.dict())
        
        return scene
        
    except Exception as e:
        logger.error(f"创建场景失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建场景失败: {str(e)}")


@router.put("/{scene_id}", response_model=Scene)
async def update_scene(
    scene_id: str = Path(..., description="场景ID"),
    scene_data: SceneUpdate = Body(...),
    services = Depends(get_service)
):
    """
    更新场景
    
    更新现有的对话场景
    """
    try:
        scene_service = services["scene_service"]
        
        # 更新场景
        scene = scene_service.update_scene(scene_id, scene_data.dict(exclude_unset=True))
        
        return scene
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"更新场景失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新场景失败: {str(e)}")


@router.delete("/{scene_id}")
async def delete_scene(
    scene_id: str = Path(..., description="场景ID"),
    services = Depends(get_service)
):
    """
    删除场景
    
    删除指定的对话场景
    """
    try:
        scene_service = services["scene_service"]
        
        # 删除场景
        success = scene_service.delete_scene(scene_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"场景不存在: {scene_id}")
        
        return {"status": "success", "message": "场景已删除"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除场景失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除场景失败: {str(e)}")


@router.post("/extract-from-document", response_model=Scene)
async def extract_scene_from_document(
    file: UploadFile = File(...),
    title: str = Form(None),
    services = Depends(get_service)
):
    """
    从文档提取场景
    
    上传文档并自动提取对话场景
    """
    try:
        scene_service = services["scene_service"]
        document_service = services["document_service"]
        
        # 保存上传文件
        document_text, document_metadata = await document_service.process_uploaded_file(file)
        
        # 如果提供了标题，更新元数据
        if title:
            document_metadata["title"] = title
        
        # 提取场景
        scene_data = scene_service.extract_scene_from_document(document_text, document_metadata)
        
        # 创建场景
        scene = scene_service.create_scene(scene_data)
        
        return scene
        
    except Exception as e:
        logger.error(f"从文档提取场景失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"从文档提取场景失败: {str(e)}")


@router.get("/categories/list", response_model=List[SceneCategory])
async def get_scene_categories(
    services = Depends(get_service)
):
    """
    获取场景类别
    
    返回所有可用的场景类别
    """
    try:
        scene_service = services["scene_service"]
        
        # 获取类别
        categories = scene_service.get_scene_categories()
        
        return categories
        
    except Exception as e:
        logger.error(f"获取场景类别失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取场景类别失败: {str(e)}")


@router.get("/difficulty/levels", response_model=List[DifficultyLevel])
async def get_difficulty_levels(
    services = Depends(get_service)
):
    """
    获取难度级别
    
    返回所有可用的难度级别
    """
    try:
        scene_service = services["scene_service"]
        
        # 获取难度级别
        levels = scene_service.get_difficulty_levels()
        
        return levels
        
    except Exception as e:
        logger.error(f"获取难度级别失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取难度级别失败: {str(e)}")


@router.post("/filter", response_model=List[Scene])
async def filter_scenes(
    filter_criteria: SceneFilter,
    services = Depends(get_service)
):
    """
    筛选场景
    
    根据复杂条件筛选场景
    """
    try:
        scene_service = services["scene_service"]
        
        # 筛选场景
        scenes = scene_service.filter_scenes(filter_criteria.dict(exclude_unset=True))
        
        return scenes
        
    except Exception as e:
        logger.error(f"筛选场景失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"筛选场景失败: {str(e)}") 