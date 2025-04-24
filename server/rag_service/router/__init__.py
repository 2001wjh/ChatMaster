"""
路由模块初始化文件
"""

from server.rag_service.router import qa_router, document_router, knowledge_base_router

__all__ = ["qa_router", "document_router", "knowledge_base_router"] 