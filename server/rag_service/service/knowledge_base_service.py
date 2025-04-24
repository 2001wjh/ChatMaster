"""
知识库服务
负责构建和管理向量数据库，支持文本检索功能
"""

import os
import logging
import pickle
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class KnowledgeBaseService:
    """知识库服务类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化知识库服务
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.vector_store_type = config.get("vector_store_type", "faiss")
        self.embedding_model = config.get("embedding_model", "text-embedding-ada-002")
        
        # 初始化OpenAI客户端
        api_key = os.environ.get("OPENAI_API_KEY") or config.get("openai_api_key")
        if not api_key:
            raise ValueError("缺少OpenAI API密钥")
        
        self.client = OpenAI(api_key=api_key)
        
        # 创建embedding模型
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=api_key
        )
        
        # 知识库存储路径
        self.kb_dir = config.get("kb_dir", "dataset/knowledge_base")
        os.makedirs(self.kb_dir, exist_ok=True)
        
        # 场景信息
        self.scenes = []
        
    def create_knowledge_base(self, text_chunks: List[Dict[str, Any]], kb_name: str) -> str:
        """
        创建知识库
        
        Args:
            text_chunks: 文本块列表
            kb_name: 知识库名称
            
        Returns:
            知识库ID
        """
        try:
            # 提取文本内容和元数据
            texts = [chunk["content"] for chunk in text_chunks]
            metadatas = [chunk["metadata"] for chunk in text_chunks]
            
            # 创建向量存储
            vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # 保存向量存储
            kb_path = os.path.join(self.kb_dir, kb_name)
            os.makedirs(kb_path, exist_ok=True)
            
            vector_store.save_local(kb_path)
            
            # 保存原始文本块
            with open(os.path.join(kb_path, "chunks.pkl"), "wb") as f:
                pickle.dump(text_chunks, f)
                
            logger.info(f"知识库 {kb_name} 创建成功")
            return kb_name
            
        except Exception as e:
            logger.error(f"创建知识库失败: {str(e)}")
            raise RuntimeError(f"创建知识库失败: {str(e)}")
    
    def load_knowledge_base(self, kb_name: str) -> FAISS:
        """
        加载知识库
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            向量存储实例
        """
        try:
            kb_path = os.path.join(self.kb_dir, kb_name)
            
            if not os.path.exists(kb_path):
                raise ValueError(f"知识库 {kb_name} 不存在")
                
            # 加载向量存储
            vector_store = FAISS.load_local(
                kb_path,
                self.embeddings
            )
            
            logger.info(f"知识库 {kb_name} 加载成功")
            return vector_store
            
        except Exception as e:
            logger.error(f"加载知识库失败: {str(e)}")
            raise RuntimeError(f"加载知识库失败: {str(e)}")
    
    def search(self, query: str, kb_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        在知识库中搜索相关内容
        
        Args:
            query: 查询文本
            kb_name: 知识库名称
            top_k: 返回结果数量
            
        Returns:
            相关内容列表
        """
        try:
            vector_store = self.load_knowledge_base(kb_name)
            
            # 执行相似度搜索
            docs_with_scores = vector_store.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            # 整理结果
            results = []
            for doc, score in docs_with_scores:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
                
            return results
            
        except Exception as e:
            logger.error(f"搜索知识库失败: {str(e)}")
            raise RuntimeError(f"搜索知识库失败: {str(e)}")
    
    def save_scenes(self, scenes: List[Dict[str, Any]], kb_name: str) -> None:
        """
        保存场景信息
        
        Args:
            scenes: 场景信息列表
            kb_name: 知识库名称
        """
        try:
            kb_path = os.path.join(self.kb_dir, kb_name)
            
            if not os.path.exists(kb_path):
                os.makedirs(kb_path, exist_ok=True)
                
            # 保存场景信息
            with open(os.path.join(kb_path, "scenes.pkl"), "wb") as f:
                pickle.dump(scenes, f)
                
            # 更新内存中的场景列表
            self.scenes = scenes
            
            logger.info(f"场景信息保存成功，共 {len(scenes)} 个场景")
            
        except Exception as e:
            logger.error(f"保存场景信息失败: {str(e)}")
            raise RuntimeError(f"保存场景信息失败: {str(e)}")
    
    def load_scenes(self, kb_name: str) -> List[Dict[str, Any]]:
        """
        加载场景信息
        
        Args:
            kb_name: 知识库名称
            
        Returns:
            场景信息列表
        """
        try:
            kb_path = os.path.join(self.kb_dir, kb_name)
            scene_file = os.path.join(kb_path, "scenes.pkl")
            
            if not os.path.exists(scene_file):
                logger.warning(f"场景信息文件不存在: {scene_file}")
                return []
                
            # 加载场景信息
            with open(scene_file, "rb") as f:
                scenes = pickle.load(f)
                
            # 更新内存中的场景列表
            self.scenes = scenes
            
            logger.info(f"场景信息加载成功，共 {len(scenes)} 个场景")
            return scenes
            
        except Exception as e:
            logger.error(f"加载场景信息失败: {str(e)}")
            raise RuntimeError(f"加载场景信息失败: {str(e)}")
    
    def get_scene_by_id(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取场景信息
        
        Args:
            scene_id: 场景ID
            
        Returns:
            场景信息，如果不存在则返回None
        """
        for scene in self.scenes:
            if scene["id"] == scene_id:
                return scene
        return None 