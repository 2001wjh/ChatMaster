"""
助手服务
集成语音识别(ASR)、语音合成(TTS)和RAG功能，提供完整的英语外教体验
"""

import os
import uuid
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple, BinaryIO

logger = logging.getLogger(__name__)

class AssistantService:
    """助手服务类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化助手服务
        
        Args:
            config: 配置参数
        """
        self.config = config
        
        # 初始化服务
        try:
            # 初始化文档服务
            from server.rag_service.service.document_service import DocumentService
            self.document_service = DocumentService(config)
            
            # 初始化知识库服务
            from server.rag_service.service.knowledge_base_service import KnowledgeBaseService
            self.kb_service = KnowledgeBaseService(config)
            
            # 初始化问答服务
            from server.rag_service.service.qa_service import QAService
            self.qa_service = QAService(config)
            
            # 默认知识库和场景
            self.default_kb = config.get("default_kb", "default")
            self._ensure_default_kb()
            
            # 语音识别服务
            from server.FunASR.funasr.runtime.python.asr_api import ASREngine
            self.asr_engine = ASREngine(
                model_dir=config.get("asr_model_dir", "model_zoo/asr"),
                quantize=True,
                output_dir="./runtime_output"
            )
            
            # 语音合成服务
            from server.ChatTTS.ChatTTS.tts_api import TTSEngine
            self.tts_engine = TTSEngine(
                model_dir=config.get("tts_model_dir", "model_zoo/tts"),
                device="cpu"
            )
            
            logger.info("助手服务初始化成功")
            
        except Exception as e:
            logger.error(f"初始化助手服务失败: {str(e)}")
            raise RuntimeError(f"初始化助手服务失败: {str(e)}")
    
    def _ensure_default_kb(self) -> None:
        """确保默认知识库存在"""
        kb_dir = os.path.join(self.config.get("kb_dir", "dataset/knowledge_base"), self.default_kb)
        
        if not os.path.exists(kb_dir):
            # 创建默认场景
            default_scenes = self._create_default_scenes()
            
            # 保存默认场景
            os.makedirs(kb_dir, exist_ok=True)
            self.kb_service.save_scenes(default_scenes, self.default_kb)
            
            logger.info(f"创建默认知识库和场景: {self.default_kb}")
    
    def _create_default_scenes(self) -> List[Dict[str, Any]]:
        """创建默认场景"""
        return [
            {
                "id": "scene_restaurant",
                "name": "餐厅点餐",
                "description": "在餐厅与服务员交流点餐的场景",
                "content": """
                You are in a restaurant and want to order food. The waiter/waitress approaches your table.
                
                Useful phrases:
                - "I'd like to order..."
                - "What's your special today?"
                - "Could you recommend something?"
                - "I'm allergic to..."
                - "How spicy is this dish?"
                """,
                "difficulty": "初级",
                "metadata": {"category": "daily_life"}
            },
            {
                "id": "scene_hotel",
                "name": "酒店入住",
                "description": "在酒店前台办理入住手续的场景",
                "content": """
                You are at a hotel reception desk checking in for your reservation.
                
                Useful phrases:
                - "I have a reservation under the name..."
                - "What time is checkout?"
                - "Is breakfast included?"
                - "Do you have any rooms with a better view?"
                - "Is there free Wi-Fi?"
                """,
                "difficulty": "初级",
                "metadata": {"category": "travel"}
            },
            {
                "id": "scene_airport",
                "name": "机场安检",
                "description": "在机场通过安检的场景",
                "content": """
                You are at the airport security checkpoint before your flight.
                
                Useful phrases:
                - "Where should I put my laptop?"
                - "Do I need to take off my shoes?"
                - "I have a connecting flight to..."
                - "How much time do I have before boarding?"
                - "Is this liquid allowed in carry-on?"
                """,
                "difficulty": "中级",
                "metadata": {"category": "travel"}
            },
            {
                "id": "scene_interview",
                "name": "工作面试",
                "description": "参加工作面试的场景",
                "content": """
                You are at a job interview for a position you're interested in.
                
                Useful phrases:
                - "Let me tell you about my experience with..."
                - "One of my strengths is..."
                - "I'm particularly interested in this role because..."
                - "Could you tell me more about the team I'd be working with?"
                - "What would be the main challenges in this position?"
                """,
                "difficulty": "高级",
                "metadata": {"category": "professional"}
            },
            {
                "id": "scene_shopping",
                "name": "购物场景",
                "description": "在商店购物并与店员交流的场景",
                "content": """
                You are shopping in a store and need help from a sales assistant.
                
                Useful phrases:
                - "Do you have this in a different size/color?"
                - "Where can I find the...?"
                - "Is this on sale?"
                - "Can I return this if it doesn't fit?"
                - "I'm just browsing, thanks."
                """,
                "difficulty": "初级",
                "metadata": {"category": "daily_life"}
            },
            {
                "id": "scene_doctor",
                "name": "看医生",
                "description": "在医院向医生描述症状的场景",
                "content": """
                You are at a doctor's office explaining your symptoms.
                
                Useful phrases:
                - "I've been feeling... for the past few days."
                - "The pain is located in my..."
                - "Is this condition serious?"
                - "What are the side effects of this medication?"
                - "How long will it take to recover?"
                """,
                "difficulty": "中级",
                "metadata": {"category": "health"}
            },
            {
                "id": "scene_business",
                "name": "商务会议",
                "description": "参加商务会议并进行讨论的场景",
                "content": """
                You are in a business meeting with colleagues discussing a project.
                
                Useful phrases:
                - "I'd like to address the issue of..."
                - "From my perspective, we should consider..."
                - "I agree with what was said about..."
                - "Could we circle back to the point about...?"
                - "Let's set a timeline for this deliverable."
                """,
                "difficulty": "高级",
                "metadata": {"category": "professional"}
            },
            {
                "id": "scene_transportation",
                "name": "公共交通",
                "description": "乘坐公共交通工具时的交流场景",
                "content": """
                You are using public transportation in a foreign city.
                
                Useful phrases:
                - "Does this bus/train go to...?"
                - "How much is a ticket to...?"
                - "Could you tell me when we reach...?"
                - "Is this seat taken?"
                - "How frequent are the trains to...?"
                """,
                "difficulty": "初级",
                "metadata": {"category": "travel"}
            },
            {
                "id": "scene_party",
                "name": "社交聚会",
                "description": "在社交聚会上与新朋友交流的场景",
                "content": """
                You are at a social gathering where you don't know many people.
                
                Useful phrases:
                - "How do you know the host?"
                - "What do you do for a living?"
                - "Have you lived in this area long?"
                - "I couldn't help but notice your... It's very nice!"
                - "Would you like to exchange contact information?"
                """,
                "difficulty": "中级",
                "metadata": {"category": "social"}
            },
            {
                "id": "scene_emergency",
                "name": "紧急情况",
                "description": "处理紧急情况时的交流场景",
                "content": """
                You need to communicate during an emergency situation.
                
                Useful phrases:
                - "I need help immediately."
                - "There's been an accident at..."
                - "My friend is injured and needs medical attention."
                - "Could you please call an ambulance?"
                - "Is there a hospital nearby?"
                """,
                "difficulty": "高级",
                "metadata": {"category": "emergency"}
            }
        ]
    
    def upload_document(self, file_path: str, file_type: Optional[str] = None) -> Dict[str, Any]:
        """
        上传并处理文档
        
        Args:
            file_path: 文件路径
            file_type: 文件类型
            
        Returns:
            处理结果
        """
        try:
            # 处理文档
            text_chunks = self.document_service.process_file(file_path, file_type)
            
            # 提取场景信息
            scenes = self.document_service.extract_scene_info(text_chunks)
            
            # 创建知识库
            kb_name = f"kb_{uuid.uuid4().hex[:8]}"
            self.kb_service.create_knowledge_base(text_chunks, kb_name)
            
            # 保存场景信息
            self.kb_service.save_scenes(scenes, kb_name)
            
            return {
                "kb_name": kb_name,
                "scenes": scenes,
                "chunk_count": len(text_chunks)
            }
            
        except Exception as e:
            logger.error(f"处理文档失败: {str(e)}")
            raise RuntimeError(f"处理文档失败: {str(e)}")
    
    def get_scenes(self, kb_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取场景列表
        
        Args:
            kb_name: 知识库名称，如果为None则使用默认知识库
            
        Returns:
            场景列表
        """
        if kb_name is None:
            kb_name = self.default_kb
            
        return self.kb_service.load_scenes(kb_name)
    
    def get_scene_by_id(self, scene_id: str, kb_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        根据ID获取场景信息
        
        Args:
            scene_id: 场景ID
            kb_name: 知识库名称，如果为None则使用默认知识库
            
        Returns:
            场景信息，如果不存在则返回None
        """
        if kb_name is None:
            kb_name = self.default_kb
            
        # 加载场景
        self.kb_service.load_scenes(kb_name)
        
        return self.kb_service.get_scene_by_id(scene_id)
    
    def text_chat(self, 
                 query: str, 
                 conversation_id: Optional[str] = None,
                 kb_name: Optional[str] = None,
                 scene_id: Optional[str] = None,
                 language: str = "english") -> Dict[str, Any]:
        """
        文本对话
        
        Args:
            query: 用户问题
            conversation_id: 对话ID，如果为None则创建新对话
            kb_name: 知识库名称，如果为None则使用默认知识库
            scene_id: 场景ID，如果为None则不使用场景
            language: 语言
            
        Returns:
            对话结果
        """
        try:
            # 设置默认值
            if kb_name is None:
                kb_name = self.default_kb
                
            if conversation_id is None:
                conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
            
            # 获取场景信息
            scene = None
            if scene_id:
                scene = self.get_scene_by_id(scene_id, kb_name)
            
            # 搜索相关内容
            context = []
            if scene:
                # 使用场景内容作为上下文
                context.append({
                    "content": scene["content"],
                    "metadata": scene["metadata"]
                })
            else:
                # 搜索知识库
                search_results = self.kb_service.search(query, kb_name)
                context.extend(search_results)
            
            # 生成回答
            result = self.qa_service.answer(
                query=query,
                context=context,
                conversation_id=conversation_id,
                language=language,
                role="english_teacher",
                scene=scene
            )
            
            return {
                "answer": result["answer"],
                "conversation_id": conversation_id,
                "scene_id": scene_id,
                "kb_name": kb_name
            }
            
        except Exception as e:
            logger.error(f"文本对话失败: {str(e)}")
            raise RuntimeError(f"文本对话失败: {str(e)}")
    
    def speech_to_text(self, audio_file: BinaryIO) -> str:
        """
        语音转文本
        
        Args:
            audio_file: 音频文件
            
        Returns:
            识别结果文本
        """
        try:
            # 保存临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_file.read())
                temp_path = temp_file.name
            
            # 调用语音识别
            result = self.asr_engine.inference(temp_path)
            
            # 清理临时文件
            os.unlink(temp_path)
            
            return result["text"]
            
        except Exception as e:
            logger.error(f"语音识别失败: {str(e)}")
            raise RuntimeError(f"语音识别失败: {str(e)}")
    
    def text_to_speech(self, text: str, voice: str = "en_female") -> bytes:
        """
        文本转语音
        
        Args:
            text: 文本内容
            voice: 语音类型
            
        Returns:
            音频数据
        """
        try:
            # 调用语音合成
            audio_data = self.tts_engine.inference(text, voice=voice)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"语音合成失败: {str(e)}")
            raise RuntimeError(f"语音合成失败: {str(e)}")
    
    def voice_chat(self, 
                  audio_file: BinaryIO,
                  conversation_id: Optional[str] = None,
                  kb_name: Optional[str] = None,
                  scene_id: Optional[str] = None,
                  language: str = "english") -> Dict[str, Any]:
        """
        语音对话
        
        Args:
            audio_file: 音频文件
            conversation_id: 对话ID，如果为None则创建新对话
            kb_name: 知识库名称，如果为None则使用默认知识库
            scene_id: 场景ID，如果为None则不使用场景
            language: 语言
            
        Returns:
            对话结果，包含文本和音频数据
        """
        try:
            # 语音转文本
            query_text = self.speech_to_text(audio_file)
            
            # 文本对话
            result = self.text_chat(
                query=query_text,
                conversation_id=conversation_id,
                kb_name=kb_name,
                scene_id=scene_id,
                language=language
            )
            
            # 文本转语音
            audio_data = self.text_to_speech(result["answer"])
            
            # 返回结果
            return {
                **result,
                "query_text": query_text,
                "audio_data": audio_data
            }
            
        except Exception as e:
            logger.error(f"语音对话失败: {str(e)}")
            raise RuntimeError(f"语音对话失败: {str(e)}")
            
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        获取对话历史
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            对话历史列表
        """
        return self.qa_service.get_conversation_history(conversation_id)
    
    def clear_conversation(self, conversation_id: str) -> None:
        """
        清除对话历史
        
        Args:
            conversation_id: 对话ID
        """
        self.qa_service.clear_conversation(conversation_id) 