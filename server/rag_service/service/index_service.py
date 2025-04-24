"""
索引服务
负责管理知识库的创建、更新和检索
增强了文档处理和混合检索功能
"""

import os
import json
import logging
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import hashlib
import datetime
import uuid
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    UnstructuredWordDocumentLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader
)

logger = logging.getLogger(__name__)

class IndexService:
    """索引服务类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化索引服务
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.knowledge_base_dir = Path(config.get("knowledge_base_dir", "knowledge_base"))
        self.knowledge_base_dir.mkdir(exist_ok=True, parents=True)
        
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        
        # 初始化嵌入模型
        api_key = os.environ.get("OPENAI_API_KEY") or config.get("openai_api_key")
        if not api_key:
            raise ValueError("缺少OpenAI API密钥")
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        # 高级功能设置
        self.enable_advanced_features = config.get("enable_advanced_features", True)
    
    def _get_index_dir(self, index_name: str) -> Path:
        """
        获取索引目录
        
        Args:
            index_name: 索引名称
            
        Returns:
            索引目录路径
        """
        index_dir = self.knowledge_base_dir / index_name
        return index_dir
    
    def _get_document_path(self, index_name: str, document_id: str) -> Path:
        """
        获取文档路径
        
        Args:
            index_name: 索引名称
            document_id: 文档ID
            
        Returns:
            文档路径
        """
        document_dir = self._get_index_dir(index_name) / "documents"
        return document_dir / f"{document_id}.json"
    
    def _load_vectorstore(self, index_name: str) -> Optional[FAISS]:
        """
        加载向量存储
        
        Args:
            index_name: 索引名称
            
        Returns:
            FAISS向量存储对象
        """
        index_dir = self._get_index_dir(index_name)
        if not (index_dir / "faiss_index").exists():
            return None
        
        try:
            return FAISS.load_local(str(index_dir / "faiss_index"), self.embeddings)
        except Exception as e:
            logger.error(f"加载向量存储失败: {str(e)}")
            return None
            
    def _get_document_loader(self, file_path: str) -> Any:
        """
        根据文件类型获取适合的文档加载器
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档加载器
        """
        ext = file_path.lower().split('.')[-1]
        
        if ext == 'txt':
            return TextLoader(file_path)
        elif ext == 'pdf':
            return PyPDFLoader(file_path)
        elif ext in ['doc', 'docx']:
            return UnstructuredWordDocumentLoader(file_path)
        elif ext in ['csv']:
            return CSVLoader(file_path)
        elif ext in ['ppt', 'pptx']:
            return UnstructuredPowerPointLoader(file_path)
        elif ext in ['md', 'markdown']:
            return UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {ext}")
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        提取文件元数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            元数据字典
        """
        file_path = Path(file_path)
        
        # 获取文件信息
        stat = file_path.stat()
        file_size = stat.st_size
        mod_time = datetime.datetime.fromtimestamp(stat.st_mtime)
        
        # 创建元数据
        metadata = {
            "filename": file_path.name,
            "file_extension": file_path.suffix.lower()[1:],
            "file_size": file_size,
            "modified_time": mod_time.isoformat(),
            "created_time": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat()
        }
        
        return metadata
    
    def _process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        处理文档，提取文本并分块
        
        Args:
            file_path: 文件路径
            
        Returns:
            分块后的文档列表
        """
        try:
            # 加载文档
            loader = self._get_document_loader(file_path)
            documents = loader.load()
            
            # 提取元数据
            metadata = self._extract_metadata(file_path)
            
            # 创建文本分割器
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # 分割文档
            chunks = text_splitter.split_documents(documents)
            
            # 创建文档块列表
            document_chunks = []
            
            for i, chunk in enumerate(chunks):
                # 创建块ID
                chunk_id = hashlib.md5(f"{Path(file_path).name}-{i}".encode()).hexdigest()
                
                # 合并文档元数据和块特定元数据
                chunk_metadata = {
                    **metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                
                # 添加到文档块列表
                document_chunks.append({
                    "id": chunk_id,
                    "content": chunk.page_content,
                    "metadata": chunk_metadata
                })
            
            return document_chunks
            
        except Exception as e:
            logger.error(f"处理文档失败: {str(e)}")
            raise RuntimeError(f"处理文档失败: {str(e)}")
    
    def create_index(self, index_name: str) -> bool:
        """
        创建索引
        
        Args:
            index_name: 索引名称
            
        Returns:
            是否成功创建
        """
        try:
            # 验证索引名
            if not re.match(r'^[a-zA-Z0-9_-]+$', index_name):
                raise ValueError("索引名称只能包含字母、数字、下划线和连字符")
                
            # 创建索引目录
            index_dir = self._get_index_dir(index_name)
            if index_dir.exists():
                raise ValueError(f"索引 {index_name} 已存在")
                
            # 创建索引子目录
            index_dir.mkdir(parents=True)
            (index_dir / "documents").mkdir()
            (index_dir / "faiss_index").mkdir()
            
            # 创建空的FAISS索引
            empty_vectorstore = FAISS.from_texts(["placeholder"], self.embeddings)
            empty_vectorstore.save_local(str(index_dir / "faiss_index"))
            
            # 创建索引元数据
            metadata = {
                "name": index_name,
                "created_at": datetime.datetime.now().isoformat(),
                "document_count": 0,
                "chunk_count": 0
            }
            
            with open(index_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"索引 {index_name} 创建成功")
            return True
            
        except Exception as e:
            logger.error(f"创建索引失败: {str(e)}")
            # 清理可能部分创建的目录
            try:
                index_dir = self._get_index_dir(index_name)
                if index_dir.exists():
                    shutil.rmtree(index_dir)
            except:
                pass
            raise RuntimeError(f"创建索引失败: {str(e)}")
    
    def delete_index(self, index_name: str) -> bool:
        """
        删除索引
        
        Args:
            index_name: 索引名称
            
        Returns:
            是否成功删除
        """
        try:
            index_dir = self._get_index_dir(index_name)
            if not index_dir.exists():
                raise ValueError(f"索引 {index_name} 不存在")
                
            # 删除索引目录
            shutil.rmtree(index_dir)
            
            logger.info(f"索引 {index_name} 删除成功")
            return True
            
        except Exception as e:
            logger.error(f"删除索引失败: {str(e)}")
            raise RuntimeError(f"删除索引失败: {str(e)}")
    
    def add_document(self, index_name: str, file_path: str) -> Dict[str, Any]:
        """
        向索引添加文档
        
        Args:
            index_name: 索引名称
            file_path: 文件路径
            
        Returns:
            文档信息
        """
        try:
            index_dir = self._get_index_dir(index_name)
            if not index_dir.exists():
                raise ValueError(f"索引 {index_name} 不存在")
                
            # 处理文档
            document_chunks = self._process_document(file_path)
            
            # 创建文档ID
            document_id = str(uuid.uuid4())
            
            # 创建文档记录
            document_info = {
                "id": document_id,
                "filename": Path(file_path).name,
                "chunks": document_chunks,
                "added_at": datetime.datetime.now().isoformat()
            }
            
            # 保存文档信息
            document_dir = index_dir / "documents"
            document_dir.mkdir(exist_ok=True)
            
            with open(document_dir / f"{document_id}.json", "w") as f:
                json.dump(document_info, f, indent=2)
                
            # 更新索引
            texts = [chunk["content"] for chunk in document_chunks]
            metadatas = [chunk["metadata"] for chunk in document_chunks]
            
            # 加载现有向量存储
            vectorstore = self._load_vectorstore(index_name)
            if vectorstore:
                vectorstore.add_texts(texts, metadatas)
                vectorstore.save_local(str(index_dir / "faiss_index"))
            else:
                # 创建新的向量存储
                vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
                vectorstore.save_local(str(index_dir / "faiss_index"))
                
            # 更新索引元数据
            metadata_path = index_dir / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                
            metadata["document_count"] += 1
            metadata["chunk_count"] += len(document_chunks)
            metadata["updated_at"] = datetime.datetime.now().isoformat()
            
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"文档 {Path(file_path).name} 已添加到索引 {index_name}")
            return {
                "document_id": document_id,
                "filename": Path(file_path).name,
                "chunk_count": len(document_chunks)
            }
            
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            raise RuntimeError(f"添加文档失败: {str(e)}")
    
    def delete_document(self, index_name: str, document_id: str) -> bool:
        """
        从索引中删除文档
        
        Args:
            index_name: 索引名称
            document_id: 文档ID
            
        Returns:
            是否成功删除
        """
        try:
            index_dir = self._get_index_dir(index_name)
            if not index_dir.exists():
                raise ValueError(f"索引 {index_name} 不存在")
                
            # 检查文档是否存在
            document_path = self._get_document_path(index_name, document_id)
            if not document_path.exists():
                raise ValueError(f"文档 {document_id} 不存在")
                
            # 读取文档信息
            with open(document_path, "r") as f:
                document_info = json.load(f)
                
            # 删除文档文件
            document_path.unlink()
            
            # 从向量存储中删除文档
            # 注意：当前FAISS不支持直接删除，需要重建索引
            # 这是一个简化的实现，实际应用中可能需要更复杂的处理
            if self.enable_advanced_features:
                # 获取所有文档
                document_dir = index_dir / "documents"
                all_documents = []
                
                for doc_path in document_dir.glob("*.json"):
                    with open(doc_path, "r") as f:
                        doc_data = json.load(f)
                        all_documents.extend(doc_data["chunks"])
                
                # 重建向量存储
                if all_documents:
                    texts = [chunk["content"] for chunk in all_documents]
                    metadatas = [chunk["metadata"] for chunk in all_documents]
                    
                    vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
                    vectorstore.save_local(str(index_dir / "faiss_index"))
                else:
                    # 如果没有文档，创建空索引
                    empty_vectorstore = FAISS.from_texts(["placeholder"], self.embeddings)
                    empty_vectorstore.save_local(str(index_dir / "faiss_index"))
            
            # 更新索引元数据
            metadata_path = index_dir / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                
            metadata["document_count"] -= 1
            metadata["chunk_count"] -= len(document_info["chunks"])
            metadata["updated_at"] = datetime.datetime.now().isoformat()
            
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"文档 {document_id} 已从索引 {index_name} 中删除")
            return True
            
        except Exception as e:
            logger.error(f"删除文档失败: {str(e)}")
            raise RuntimeError(f"删除文档失败: {str(e)}")
    
    def list_indices(self) -> List[Dict[str, Any]]:
        """
        列出所有索引
        
        Returns:
            索引列表
        """
        try:
            indices = []
            
            for index_dir in self.knowledge_base_dir.iterdir():
                if index_dir.is_dir() and (index_dir / "metadata.json").exists():
                    try:
                        with open(index_dir / "metadata.json", "r") as f:
                            metadata = json.load(f)
                            indices.append(metadata)
                    except:
                        logger.warning(f"无法读取索引 {index_dir.name} 的元数据")
            
            return indices
            
        except Exception as e:
            logger.error(f"列出索引失败: {str(e)}")
            raise RuntimeError(f"列出索引失败: {str(e)}")
    
    def list_documents(self, index_name: str) -> List[Dict[str, Any]]:
        """
        列出索引中的所有文档
        
        Args:
            index_name: 索引名称
            
        Returns:
            文档列表
        """
        try:
            index_dir = self._get_index_dir(index_name)
            if not index_dir.exists():
                raise ValueError(f"索引 {index_name} 不存在")
                
            documents = []
            document_dir = index_dir / "documents"
            
            for doc_path in document_dir.glob("*.json"):
                try:
                    with open(doc_path, "r") as f:
                        doc_data = json.load(f)
                        documents.append({
                            "id": doc_data["id"],
                            "filename": doc_data["filename"],
                            "added_at": doc_data["added_at"],
                            "chunk_count": len(doc_data["chunks"])
                        })
                except:
                    logger.warning(f"无法读取文档 {doc_path.name}")
            
            return documents
            
        except Exception as e:
            logger.error(f"列出文档失败: {str(e)}")
            raise RuntimeError(f"列出文档失败: {str(e)}")
    
    def get_document_info(self, index_name: str, document_id: str) -> Dict[str, Any]:
        """
        获取文档信息
        
        Args:
            index_name: 索引名称
            document_id: 文档ID
            
        Returns:
            文档信息
        """
        try:
            document_path = self._get_document_path(index_name, document_id)
            if not document_path.exists():
                raise ValueError(f"文档 {document_id} 不存在")
                
            with open(document_path, "r") as f:
                document_info = json.load(f)
                
            return document_info
            
        except Exception as e:
            logger.error(f"获取文档信息失败: {str(e)}")
            raise RuntimeError(f"获取文档信息失败: {str(e)}")
    
    def search(self, 
             index_name: str, 
             query: str, 
             top_k: int = 5, 
             search_type: str = "hybrid") -> List[Dict[str, Any]]:
        """
        搜索索引
        
        Args:
            index_name: 索引名称
            query: 查询字符串
            top_k: 返回结果数量
            search_type: 搜索类型 (vector, keyword, hybrid)
            
        Returns:
            搜索结果列表
        """
        try:
            index_dir = self._get_index_dir(index_name)
            if not index_dir.exists():
                raise ValueError(f"索引 {index_name} 不存在")
                
            # 加载向量存储
            vectorstore = self._load_vectorstore(index_name)
            if not vectorstore:
                return []
                
            results = []
            
            # 向量搜索
            if search_type in ["vector", "hybrid"]:
                vector_results = vectorstore.similarity_search_with_score(query, k=top_k * 2)
                
                for doc, score in vector_results:
                    # 转换分数到0-1范围（FAISS距离是越小越好）
                    similarity = 1.0 / (1.0 + score)
                    
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": similarity,
                        "source": "vector"
                    })
            
            # 关键词搜索
            if search_type in ["keyword", "hybrid"] and self.enable_advanced_features:
                # 简单关键词匹配实现，实际项目可能使用更复杂的算法
                document_dir = index_dir / "documents"
                keyword_results = []
                
                # 提取查询中的关键词
                keywords = set(re.findall(r'\w+', query.lower()))
                
                for doc_path in document_dir.glob("*.json"):
                    with open(doc_path, "r") as f:
                        doc_data = json.load(f)
                        
                        for chunk in doc_data["chunks"]:
                            content = chunk["content"].lower()
                            
                            # 计算关键词匹配度
                            matched_keywords = sum(1 for kw in keywords if kw in content)
                            if matched_keywords > 0:
                                score = matched_keywords / len(keywords)
                                
                                keyword_results.append({
                                    "content": chunk["content"],
                                    "metadata": chunk["metadata"],
                                    "score": score,
                                    "source": "keyword"
                                })
                
                # 添加关键词搜索结果
                results.extend(sorted(keyword_results, key=lambda x: x["score"], reverse=True)[:top_k])
            
            # 结果排序和去重
            seen_chunks = set()
            filtered_results = []
            
            for result in sorted(results, key=lambda x: x["score"], reverse=True):
                chunk_id = result["metadata"]["chunk_id"]
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    filtered_results.append(result)
                    
                    if len(filtered_results) >= top_k:
                        break
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            raise RuntimeError(f"搜索失败: {str(e)}")
    
    def get_index_statistics(self, index_name: str) -> Dict[str, Any]:
        """
        获取索引统计信息
        
        Args:
            index_name: 索引名称
            
        Returns:
            索引统计信息
        """
        try:
            index_dir = self._get_index_dir(index_name)
            if not index_dir.exists():
                raise ValueError(f"索引 {index_name} 不存在")
                
            # 读取基本元数据
            with open(index_dir / "metadata.json", "r") as f:
                metadata = json.load(f)
                
            # 计算存储大小
            total_size = 0
            
            # 1. 计算向量索引大小
            faiss_dir = index_dir / "faiss_index"
            for path in faiss_dir.glob("**/*"):
                if path.is_file():
                    total_size += path.stat().st_size
            
            # 2. 计算文档大小
            docs_dir = index_dir / "documents"
            for path in docs_dir.glob("**/*"):
                if path.is_file():
                    total_size += path.stat().st_size
            
            # 创建统计信息
            stats = {
                **metadata,
                "storage_size_bytes": total_size,
                "storage_size_mb": round(total_size / (1024 * 1024), 2)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取索引统计信息失败: {str(e)}")
            raise RuntimeError(f"获取索引统计信息失败: {str(e)}") 