"""
文档解析器服务
负责解析多种格式的文档并提取结构化内容
"""

import os
import logging
import json
import re
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentParserService:
    """文档解析器服务类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化文档解析器服务
        
        Args:
            config: 配置参数
        """
        self.config = config
        
        # 临时文件目录
        self.temp_dir = config.get("temp_dir", "temp")
        Path(self.temp_dir).mkdir(exist_ok=True, parents=True)
        
        # 支持的文件类型
        self.supported_extensions = {
            "txt": self._parse_txt,
            "pdf": self._parse_pdf,
            "docx": self._parse_docx,
            "doc": self._parse_doc,
            "xlsx": self._parse_xlsx,
            "xls": self._parse_xls,
            "csv": self._parse_csv,
            "md": self._parse_markdown,
            "json": self._parse_json,
            "html": self._parse_html,
        }
        
        # 配置其他参数
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        self.max_file_size = config.get("max_file_size", 10 * 1024 * 1024)  # 10MB
    
    def parse_document(self, file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        解析文档并返回内容块和元数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            内容块列表和元数据
        """
        try:
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                raise ValueError(f"文件过大: {file_size / (1024 * 1024):.2f}MB，超过最大限制 {self.max_file_size / (1024 * 1024)}MB")
                
            # 获取文件扩展名
            _, extension = os.path.splitext(file_path)
            extension = extension.lower()[1:]  # 移除前导点号
            
            # 检查文件类型是否支持
            if extension not in self.supported_extensions:
                raise ValueError(f"不支持的文件类型: {extension}")
                
            # 调用相应的解析器
            parser = self.supported_extensions[extension]
            chunks, metadata = parser(file_path)
            
            # 处理结果
            processed_chunks = self._process_chunks(chunks, metadata)
            
            return processed_chunks, metadata
            
        except Exception as e:
            logger.error(f"解析文档失败: {str(e)}")
            raise
    
    def _process_chunks(self, chunks: List[str], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理文本块，添加元数据和ID
        
        Args:
            chunks: 文本块列表
            metadata: 文档元数据
            
        Returns:
            处理后的文本块列表
        """
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": f"{metadata.get('file_id', 'unknown')}-{i}",
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
            
            processed_chunks.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
        
        return processed_chunks
    
    def _parse_txt(self, file_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        解析TXT文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文本块列表和元数据
        """
        try:
            # 读取文件内容
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # 提取元数据
            file_name = os.path.basename(file_path)
            file_id = self._generate_file_id(file_path)
            
            metadata = {
                "file_name": file_name,
                "file_path": file_path,
                "file_id": file_id,
                "file_type": "txt",
                "file_size": os.path.getsize(file_path),
                "mime_type": "text/plain"
            }
            
            # 分块
            chunks = self._split_text(content)
            
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"解析TXT文件失败: {str(e)}")
            raise
    
    def _parse_pdf(self, file_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        解析PDF文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文本块列表和元数据
        """
        try:
            # 导入PDF解析库
            import PyPDF2
            
            # 读取PDF
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                
                # 提取元数据
                info = reader.metadata
                file_name = os.path.basename(file_path)
                file_id = self._generate_file_id(file_path)
                
                metadata = {
                    "file_name": file_name,
                    "file_path": file_path,
                    "file_id": file_id,
                    "file_type": "pdf",
                    "file_size": os.path.getsize(file_path),
                    "mime_type": "application/pdf",
                    "page_count": len(reader.pages),
                    "title": info.title if info else None,
                    "author": info.author if info else None,
                    "subject": info.subject if info else None,
                    "creator": info.creator if info else None,
                    "producer": info.producer if info else None
                }
                
                # 提取文本
                content = ""
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        content += text + "\n\n"
                        
                # 分块
                chunks = self._split_text(content)
                
                return chunks, metadata
                
        except ImportError:
            logger.error("解析PDF文件失败: 未安装PyPDF2库")
            raise ImportError("需要安装PyPDF2库以解析PDF文件")
        except Exception as e:
            logger.error(f"解析PDF文件失败: {str(e)}")
            raise
    
    def _parse_docx(self, file_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        解析DOCX文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文本块列表和元数据
        """
        try:
            # 导入docx解析库
            import docx
            
            # 读取docx
            doc = docx.Document(file_path)
            
            # 提取元数据
            file_name = os.path.basename(file_path)
            file_id = self._generate_file_id(file_path)
            
            metadata = {
                "file_name": file_name,
                "file_path": file_path,
                "file_id": file_id,
                "file_type": "docx",
                "file_size": os.path.getsize(file_path),
                "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "title": doc.core_properties.title,
                "author": doc.core_properties.author,
                "created": doc.core_properties.created.isoformat() if doc.core_properties.created else None,
                "modified": doc.core_properties.modified.isoformat() if doc.core_properties.modified else None
            }
            
            # 提取文本
            content = ""
            for para in doc.paragraphs:
                if para.text:
                    content += para.text + "\n"
                    
            # 分块
            chunks = self._split_text(content)
            
            return chunks, metadata
            
        except ImportError:
            logger.error("解析DOCX文件失败: 未安装python-docx库")
            raise ImportError("需要安装python-docx库以解析DOCX文件")
        except Exception as e:
            logger.error(f"解析DOCX文件失败: {str(e)}")
            raise
    
    def _parse_doc(self, file_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        解析DOC文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文本块列表和元数据
        """
        try:
            # 尝试使用LibreOffice转换
            temp_file = self._convert_doc_to_docx(file_path)
            
            # 解析转换后的docx文件
            if temp_file:
                chunks, metadata = self._parse_docx(temp_file)
                
                # 更新元数据
                metadata.update({
                    "file_name": os.path.basename(file_path),
                    "file_path": file_path,
                    "file_id": self._generate_file_id(file_path),
                    "file_type": "doc",
                    "mime_type": "application/msword"
                })
                
                # 清理临时文件
                try:
                    os.remove(temp_file)
                except:
                    pass
                    
                return chunks, metadata
            else:
                raise ValueError("无法转换DOC文件")
                
        except Exception as e:
            logger.error(f"解析DOC文件失败: {str(e)}")
            raise
    
    def _parse_xlsx(self, file_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        解析XLSX文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文本块列表和元数据
        """
        try:
            # 导入xlsx解析库
            import pandas as pd
            
            # 提取元数据
            file_name = os.path.basename(file_path)
            file_id = self._generate_file_id(file_path)
            
            metadata = {
                "file_name": file_name,
                "file_path": file_path,
                "file_id": file_id,
                "file_type": "xlsx",
                "file_size": os.path.getsize(file_path),
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            }
            
            # 读取Excel
            xl = pd.ExcelFile(file_path)
            sheet_names = xl.sheet_names
            
            # 存储所有表格的内容
            contents = []
            
            # 处理每个工作表
            for sheet_name in sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # 转换为文本表示
                sheet_content = f"--- Sheet: {sheet_name} ---\n"
                sheet_content += df.to_string(index=False) + "\n\n"
                
                contents.append(sheet_content)
                
            # 合并所有工作表的内容
            content = "\n".join(contents)
            
            # 分块
            chunks = self._split_text(content)
            
            return chunks, metadata
            
        except ImportError:
            logger.error("解析XLSX文件失败: 未安装pandas库")
            raise ImportError("需要安装pandas库以解析XLSX文件")
        except Exception as e:
            logger.error(f"解析XLSX文件失败: {str(e)}")
            raise
    
    def _parse_xls(self, file_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        解析XLS文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文本块列表和元数据
        """
        # XLS解析与XLSX类似，可以使用相同的pandas方法
        return self._parse_xlsx(file_path)
    
    def _parse_csv(self, file_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        解析CSV文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文本块列表和元数据
        """
        try:
            # 导入csv解析库
            import pandas as pd
            
            # 提取元数据
            file_name = os.path.basename(file_path)
            file_id = self._generate_file_id(file_path)
            
            metadata = {
                "file_name": file_name,
                "file_path": file_path,
                "file_id": file_id,
                "file_type": "csv",
                "file_size": os.path.getsize(file_path),
                "mime_type": "text/csv"
            }
            
            # 读取CSV
            df = pd.read_csv(file_path)
            
            # 转换为文本表示
            content = df.to_string(index=False)
            
            # 分块
            chunks = self._split_text(content)
            
            return chunks, metadata
            
        except ImportError:
            logger.error("解析CSV文件失败: 未安装pandas库")
            raise ImportError("需要安装pandas库以解析CSV文件")
        except Exception as e:
            logger.error(f"解析CSV文件失败: {str(e)}")
            raise
    
    def _parse_markdown(self, file_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        解析Markdown文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文本块列表和元数据
        """
        try:
            # 读取文件内容
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # 提取元数据
            file_name = os.path.basename(file_path)
            file_id = self._generate_file_id(file_path)
            
            # 尝试提取YAML前置元数据
            metadata = {
                "file_name": file_name,
                "file_path": file_path,
                "file_id": file_id,
                "file_type": "md",
                "file_size": os.path.getsize(file_path),
                "mime_type": "text/markdown"
            }
            
            # 提取文档标题（第一个#标记的行）
            title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
            if title_match:
                metadata["title"] = title_match.group(1).strip()
                
            # 分块
            chunks = self._split_text(content)
            
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"解析Markdown文件失败: {str(e)}")
            raise
    
    def _parse_json(self, file_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        解析JSON文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文本块列表和元数据
        """
        try:
            # 读取JSON
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # 提取元数据
            file_name = os.path.basename(file_path)
            file_id = self._generate_file_id(file_path)
            
            metadata = {
                "file_name": file_name,
                "file_path": file_path,
                "file_id": file_id,
                "file_type": "json",
                "file_size": os.path.getsize(file_path),
                "mime_type": "application/json"
            }
            
            # 转换为文本表示
            content = json.dumps(data, ensure_ascii=False, indent=2)
            
            # 分块
            chunks = self._split_text(content)
            
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"解析JSON文件失败: {str(e)}")
            raise
    
    def _parse_html(self, file_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        解析HTML文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文本块列表和元数据
        """
        try:
            # 导入HTML解析库
            from bs4 import BeautifulSoup
            
            # 读取HTML
            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
                
            # 解析HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # 提取元数据
            file_name = os.path.basename(file_path)
            file_id = self._generate_file_id(file_path)
            
            metadata = {
                "file_name": file_name,
                "file_path": file_path,
                "file_id": file_id,
                "file_type": "html",
                "file_size": os.path.getsize(file_path),
                "mime_type": "text/html",
                "title": soup.title.string if soup.title else None
            }
            
            # 提取文本
            # 移除脚本和样式内容
            for script in soup(["script", "style"]):
                script.decompose()
                
            # 获取文本
            content = soup.get_text()
            
            # 分块
            chunks = self._split_text(content)
            
            return chunks, metadata
            
        except ImportError:
            logger.error("解析HTML文件失败: 未安装BeautifulSoup库")
            raise ImportError("需要安装BeautifulSoup库以解析HTML文件")
        except Exception as e:
            logger.error(f"解析HTML文件失败: {str(e)}")
            raise
    
    def _convert_doc_to_docx(self, file_path: str) -> Optional[str]:
        """
        将DOC转换为DOCX
        
        Args:
            file_path: DOC文件路径
            
        Returns:
            转换后的DOCX文件路径，失败返回None
        """
        try:
            # 使用LibreOffice转换
            import subprocess
            
            # 创建临时文件路径
            output_dir = tempfile.gettempdir()
            output_path = os.path.join(output_dir, os.path.basename(file_path) + "x")
            
            # 执行转换命令
            # 注意：需要安装LibreOffice
            subprocess.run([
                "libreoffice", "--headless", "--convert-to", "docx",
                "--outdir", output_dir, file_path
            ], check=True)
            
            # 检查输出文件是否存在
            if os.path.exists(output_path):
                return output_path
                
            return None
            
        except Exception as e:
            logger.error(f"DOC转DOCX失败: {str(e)}")
            return None
    
    def _split_text(self, text: str) -> List[str]:
        """
        分割文本为若干块
        
        Args:
            text: 要分割的文本
            
        Returns:
            文本块列表
        """
        if not text:
            return []
            
        # 简单按段落分割
        paragraphs = text.split("\n\n")
        
        # 合并段落为块
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # 跳过空段落
            if not para.strip():
                continue
                
            # 如果当前块加上新段落不超过最大长度，则添加到当前块
            if len(current_chunk) + len(para) <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # 保存当前块并开始新块
                if current_chunk:
                    chunks.append(current_chunk)
                    
                # 如果段落长度超过块大小，则直接分块
                if len(para) > self.chunk_size:
                    # 按句子分割长段落
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= self.chunk_size:
                            if current_chunk:
                                current_chunk += " " + sentence
                            else:
                                current_chunk = sentence
                        else:
                            chunks.append(current_chunk)
                            current_chunk = sentence
                else:
                    current_chunk = para
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def _generate_file_id(self, file_path: str) -> str:
        """
        生成文件ID
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件ID
        """
        import hashlib
        
        # 使用文件路径和修改时间生成ID
        file_stat = os.stat(file_path)
        content = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
        
        # 生成MD5哈希
        return hashlib.md5(content.encode()).hexdigest() 