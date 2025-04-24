#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
向量处理工具模块
提供文本向量化和相似度计算功能
"""

import logging
import numpy as np
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import os
import json
import pickle
import time

# 尝试导入可能不存在的依赖
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# 配置日志
logger = logging.getLogger(__name__)

class VectorProcessor:
    """向量处理工具类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化向量处理器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.model = self._load_model()
        self.cache_dir = self.config.get("cache_dir", "data/cache")
        
        # 创建缓存目录
        if self.config.get("enable_cache", True) and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            
    def _load_model(self) -> Optional[Any]:
        """加载向量模型"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence_transformers 库未安装，向量化功能不可用")
            return None
            
        model_name = self.config.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2")
        try:
            logger.info(f"正在加载向量模型: {model_name}")
            model = SentenceTransformer(model_name)
            logger.info("向量模型加载完成")
            return model
        except Exception as e:
            logger.error(f"加载向量模型出错: {e}")
            return None
    
    def encode_text(self, text: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        将文本编码为向量
        
        Args:
            text: 待编码文本，可以是单个字符串或字符串列表
            batch_size: 批处理大小
            
        Returns:
            编码后的向量
        """
        if not self.model:
            logger.error("向量模型未加载，无法进行编码")
            return np.array([])
            
        try:
            if isinstance(text, str):
                # 单个文本
                return self.model.encode(text)
            else:
                # 文本列表，分批处理
                all_embeddings = []
                for i in range(0, len(text), batch_size):
                    batch = text[i:i+batch_size]
                    batch_embeddings = self.model.encode(batch)
                    all_embeddings.append(batch_embeddings)
                    
                return np.vstack(all_embeddings)
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            return np.array([])
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度得分
        """
        if vec1.size == 0 or vec2.size == 0:
            return 0.0
            
        try:
            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm_a = np.linalg.norm(vec1)
            norm_b = np.linalg.norm(vec2)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return dot_product / (norm_a * norm_b)
        except Exception as e:
            logger.error(f"计算相似度失败: {e}")
            return 0.0
    
    def batch_cosine_similarity(self, query_vec: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        批量计算余弦相似度
        
        Args:
            query_vec: 查询向量
            vectors: 多个向量组成的矩阵
            
        Returns:
            相似度得分列表
        """
        if query_vec.size == 0 or vectors.size == 0:
            return np.array([])
            
        try:
            # 归一化查询向量
            query_norm = np.linalg.norm(query_vec)
            if query_norm > 0:
                query_vec = query_vec / query_norm
                
            # 归一化所有向量
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            non_zero_idx = np.where(norms > 0)[0]
            
            similarities = np.zeros(vectors.shape[0])
            
            if len(non_zero_idx) > 0:
                # 计算非零向量的相似度
                normalized_vectors = vectors[non_zero_idx] / norms[non_zero_idx]
                similarities[non_zero_idx] = np.dot(normalized_vectors, query_vec)
                
            return similarities
        except Exception as e:
            logger.error(f"批量计算相似度失败: {e}")
            return np.array([])
    
    def cache_vectors(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None, 
                     cache_name: str = "vectors") -> bool:
        """
        缓存向量
        
        Args:
            vectors: 向量矩阵
            metadata: 向量对应的元数据
            cache_name: 缓存名称
            
        Returns:
            是否缓存成功
        """
        if not self.config.get("enable_cache", True):
            return False
            
        try:
            # 确保缓存目录存在
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # 缓存向量
            vector_path = os.path.join(self.cache_dir, f"{cache_name}.npy")
            np.save(vector_path, vectors)
            
            # 缓存元数据
            if metadata:
                metadata_path = os.path.join(self.cache_dir, f"{cache_name}_metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False)
                    
            logger.info(f"向量缓存成功: {vector_path}")
            return True
        except Exception as e:
            logger.error(f"缓存向量失败: {e}")
            return False
    
    def load_cached_vectors(self, cache_name: str = "vectors") -> Dict[str, Any]:
        """
        加载缓存向量
        
        Args:
            cache_name: 缓存名称
            
        Returns:
            包含向量和元数据的字典
        """
        result = {"vectors": None, "metadata": None}
        
        if not self.config.get("enable_cache", True):
            return result
            
        try:
            # 检查向量文件
            vector_path = os.path.join(self.cache_dir, f"{cache_name}.npy")
            if not os.path.exists(vector_path):
                logger.warning(f"缓存向量文件不存在: {vector_path}")
                return result
                
            # 加载向量
            vectors = np.load(vector_path)
            result["vectors"] = vectors
            
            # 尝试加载元数据
            metadata_path = os.path.join(self.cache_dir, f"{cache_name}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                result["metadata"] = metadata
                
            logger.info(f"已加载缓存向量: {vector_path}")
            return result
        except Exception as e:
            logger.error(f"加载缓存向量失败: {e}")
            return result
    
    def build_faiss_index(self, vectors: np.ndarray, index_type: str = "Flat") -> Optional[Any]:
        """
        构建FAISS索引
        
        Args:
            vectors: 向量矩阵
            index_type: 索引类型
            
        Returns:
            FAISS索引对象
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS 库未安装，索引功能不可用")
            return None
            
        if vectors.size == 0:
            logger.error("向量为空，无法构建索引")
            return None
            
        try:
            # 获取向量维度
            d = vectors.shape[1]
            
            # 根据索引类型构建索引
            if index_type == "Flat":
                index = faiss.IndexFlatL2(d)
            elif index_type == "IVF":
                nlist = min(4096, max(int(vectors.shape[0] / 10), 1))  # 根据数据量调整聚类数
                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quantizer, d, nlist)
                
                # IVF索引需要训练
                if vectors.shape[0] > nlist:
                    index.train(vectors)
            else:
                # 默认使用平面索引
                index = faiss.IndexFlatL2(d)
                
            # 添加向量
            index.add(vectors.astype(np.float32))
            logger.info(f"FAISS索引构建完成，包含 {vectors.shape[0]} 个向量")
            
            return index
        except Exception as e:
            logger.error(f"构建FAISS索引失败: {e}")
            return None
    
    def save_faiss_index(self, index: Any, save_path: str) -> bool:
        """
        保存FAISS索引
        
        Args:
            index: FAISS索引对象
            save_path: 保存路径
            
        Returns:
            是否保存成功
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS 库未安装，无法保存索引")
            return False
            
        try:
            # 确保目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                
            # 保存索引
            faiss.write_index(index, save_path)
            logger.info(f"FAISS索引已保存到: {save_path}")
            
            return True
        except Exception as e:
            logger.error(f"保存FAISS索引失败: {e}")
            return False
    
    def load_faiss_index(self, load_path: str) -> Optional[Any]:
        """
        加载FAISS索引
        
        Args:
            load_path: 加载路径
            
        Returns:
            FAISS索引对象
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS 库未安装，无法加载索引")
            return None
            
        if not os.path.exists(load_path):
            logger.error(f"索引文件不存在: {load_path}")
            return None
            
        try:
            # 加载索引
            index = faiss.read_index(load_path)
            logger.info(f"已加载FAISS索引: {load_path}，包含 {index.ntotal} 个向量")
            
            return index
        except Exception as e:
            logger.error(f"加载FAISS索引失败: {e}")
            return None
    
    def search(self, index: Any, query_vector: np.ndarray, top_k: int = 5) -> Dict[str, np.ndarray]:
        """
        使用FAISS索引搜索相似向量
        
        Args:
            index: FAISS索引对象
            query_vector: 查询向量
            top_k: 返回结果数量
            
        Returns:
            包含距离和索引的字典
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS 库未安装，搜索功能不可用")
            return {"distances": np.array([]), "indices": np.array([])}
            
        if query_vector.size == 0:
            logger.error("查询向量为空，无法执行搜索")
            return {"distances": np.array([]), "indices": np.array([])}
            
        try:
            # 确保查询向量是二维的
            if len(query_vector.shape) == 1:
                query_vector = query_vector.reshape(1, -1)
                
            # 执行搜索
            distances, indices = index.search(query_vector.astype(np.float32), top_k)
            
            return {"distances": distances[0], "indices": indices[0]}
        except Exception as e:
            logger.error(f"FAISS搜索失败: {e}")
            return {"distances": np.array([]), "indices": np.array([])} 