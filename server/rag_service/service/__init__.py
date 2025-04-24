"""
RAG服务实现模块
提供文档处理、知识库构建和检索问答功能
"""

import logging
import os
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

def init_rag_service(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    初始化RAG服务
    
    Args:
        config: 配置参数
        
    Returns:
        服务实例
    """
    try:
        from server.rag_service.service.document_service import DocumentService
        from server.rag_service.service.knowledge_base_service import KnowledgeBaseService
        from server.rag_service.service.qa_service import QAService
        from server.rag_service.service.index_service import IndexService
        from server.rag_service.service.retrieval_service import RetrievalService
        from server.rag_service.service.scene_service import SceneService
        from server.rag_service.utils.text_processing import TextProcessingService
        from server.rag_service.utils.conversation_manager import ConversationManager
        from server.rag_service.utils.speech_processing import SpeechProcessingService
        
        # 初始化默认配置
        if config is None:
            config = {
                "model_name": "gpt-3.5-turbo",
                "vector_store_type": "faiss",
                "embedding_model": "text-embedding-ada-002",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "top_k": 5,
                "enable_advanced_features": True,
                "vector_weight": 0.7,
                "keyword_weight": 0.3,
                "enable_context_optimization": True,
                "scenes_dir": "data/scenes",
                "asr_service_url": "http://localhost:10095",
                "tts_service_url": "http://localhost:6006",
                "dialogue_history_max_turns": 10,
                "max_memory_items": 100
            }
            
        # 创建服务实例
        document_service = DocumentService(config)
        kb_service = KnowledgeBaseService(config)
        index_service = IndexService(config)
        retrieval_service = RetrievalService(config)
        qa_service = QAService(config)
        scene_service = SceneService(config)
        text_processing_service = TextProcessingService(config)
        conversation_manager = ConversationManager(config)
        speech_processing_service = SpeechProcessingService(config)
        
        return {
            "document_service": document_service,
            "knowledge_base_service": kb_service,
            "index_service": index_service,
            "retrieval_service": retrieval_service,
            "qa_service": qa_service,
            "scene_service": scene_service,
            "text_processing_service": text_processing_service,
            "conversation_manager": conversation_manager,
            "speech_processing_service": speech_processing_service,
            "config": config
        }
        
    except Exception as e:
        logger.error(f"初始化RAG服务失败: {str(e)}")
        raise RuntimeError(f"初始化RAG服务失败: {str(e)}")
        
__all__ = ["init_rag_service"] 