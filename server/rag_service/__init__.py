"""
RAG服务模块初始化文件
集成了基于检索增强生成的文档问答功能
"""

from server.rag_service.service import init_rag_service

__all__ = ["init_rag_service"] 