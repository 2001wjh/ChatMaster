"""
文档处理服务
负责处理上传的PDF、文本等文档，提取文本内容并进行结构化处理
"""

import os
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentService:
    """文档处理服务类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化文档处理服务
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.chunk_size = config.get("chunk_size", 500)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        
        # 创建文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
    def process_file(self, file_path: str, file_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        处理文件，提取文本并分割成块
        
        Args:
            file_path: 文件路径
            file_type: 文件类型，如果为None则从文件扩展名推断
            
        Returns:
            包含文本块的列表，每个块是一个字典，包含文本内容和元数据
        """
        if file_type is None:
            file_type = Path(file_path).suffix.lower().lstrip('.')
            
        if file_type in ['pdf']:
            return self._process_pdf(file_path)
        elif file_type in ['txt', 'md', 'markdown']:
            return self._process_text_file(file_path)
        elif file_type in ['csv', 'xlsx', 'xls']:
            return self._process_tabular_file(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
    
    def _process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """处理PDF文件"""
        try:
            text_content = ""
            doc = fitz.open(file_path)
            
            # 提取文本内容
            for page_num, page in enumerate(doc):
                text_content += page.get_text()
                text_content += "\n\n"  # 页面之间添加分隔符
                
            # 分割文本
            return self._split_text(text_content, {"source": file_path})
            
        except Exception as e:
            logger.error(f"处理PDF文件失败: {str(e)}")
            raise RuntimeError(f"处理PDF文件失败: {str(e)}")
    
    def _process_text_file(self, file_path: str) -> List[Dict[str, Any]]:
        """处理文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
                
            return self._split_text(text_content, {"source": file_path})
            
        except Exception as e:
            logger.error(f"处理文本文件失败: {str(e)}")
            raise RuntimeError(f"处理文本文件失败: {str(e)}")
    
    def _process_tabular_file(self, file_path: str) -> List[Dict[str, Any]]:
        """处理表格文件（CSV、Excel）"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            else:  # Excel
                df = pd.read_excel(file_path)
                
            # 将表格转换为文本描述
            text_chunks = []
            
            # 处理每一行
            for idx, row in df.iterrows():
                row_text = f"行 {idx+1}:\n"
                for col_name, value in row.items():
                    row_text += f"{col_name}: {value}\n"
                
                chunk = {
                    "content": row_text,
                    "metadata": {
                        "source": file_path,
                        "row": idx,
                    }
                }
                text_chunks.append(chunk)
                
            return text_chunks
            
        except Exception as e:
            logger.error(f"处理表格文件失败: {str(e)}")
            raise RuntimeError(f"处理表格文件失败: {str(e)}")
    
    def _split_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将文本分割成块"""
        try:
            chunks = self.text_splitter.split_text(text)
            
            # 创建带有元数据的块
            text_chunks = []
            for i, chunk_text in enumerate(chunks):
                chunk = {
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_id": i
                    }
                }
                text_chunks.append(chunk)
                
            return text_chunks
            
        except Exception as e:
            logger.error(f"文本分割失败: {str(e)}")
            raise RuntimeError(f"文本分割失败: {str(e)}")
            
    def extract_scene_info(self, text_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        从文本块中提取场景信息
        
        Args:
            text_chunks: 文本块列表
            
        Returns:
            提取出的场景信息列表
        """
        try:
            # 此处实现场景信息提取逻辑
            # 可以使用规则或模型来识别文本中的场景信息
            scenes = []
            
            # 示例：简单地按照块分割场景
            for i, chunk in enumerate(text_chunks):
                scene = {
                    "id": f"scene_{i}",
                    "name": f"场景 {i+1}",
                    "description": chunk["content"][:100] + "...",  # 使用前100个字符作为描述
                    "content": chunk["content"],
                    "difficulty": "中级",  # 默认难度级别
                    "metadata": chunk["metadata"]
                }
                scenes.append(scene)
                
            return scenes
            
        except Exception as e:
            logger.error(f"提取场景信息失败: {str(e)}")
            raise RuntimeError(f"提取场景信息失败: {str(e)}") 