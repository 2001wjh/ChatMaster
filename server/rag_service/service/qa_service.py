"""
问答服务
负责处理用户问题，使用检索增强生成回答
集成了工业实战RAG的高级功能
"""

import os
import json
import logging
import uuid
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

class QAService:
    """问答服务类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化问答服务
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.model_name = config.get("model_name", "gpt-3.5-turbo")
        self.top_k = config.get("top_k", 5)
        
        # 初始化OpenAI客户端
        api_key = os.environ.get("OPENAI_API_KEY") or config.get("openai_api_key")
        if not api_key:
            raise ValueError("缺少OpenAI API密钥")
        
        self.client = OpenAI(api_key=api_key)
        
        # 对话管理
        self.conversations = {}
        self.query_history = {}  # 存储每个对话的查询历史
        
        # 是否启用高级功能
        self.enable_advanced_features = config.get("enable_advanced_features", True)
        
        # 混合检索配置
        self.use_hybrid_search = config.get("use_hybrid_search", True)
        
        # 上下文优化配置
        self.enable_context_optimization = config.get("enable_context_optimization", True)
        
        # 自定义分隔符，用于思考链过程
        self.thinking_separator = "----思考过程----"
        
    def _generate_system_prompt(self, role: str, language: str) -> str:
        """
        生成系统提示
        
        Args:
            role: 角色（如"外教"、"助手"等）
            language: 语言（如"英语"、"中文"等）
            
        Returns:
            系统提示
        """
        if role == "english_teacher" and language == "english":
            return """
            You are a professional English teacher and conversation partner.
            You help users practice daily English speaking skills through natural conversation.
            
            Guidelines:
            1. Always respond in English, even if the user writes in another language
            2. Use natural, conversational language appropriate for the context
            3. Correct significant grammar or vocabulary errors gently
            4. Ask follow-up questions to keep the conversation flowing
            5. Adapt your language complexity to match the user's proficiency level
            6. Stay focused on the current conversation topic or scene
            7. Be encouraging and supportive
            8. If the conversation is part of a specific scene or scenario, stay within that context
            
            Remember to keep your responses concise and engaging.
            """
        elif role == "english_teacher" and language == "chinese":
            return """
            你是一位专业的英语教师和对话伙伴。
            你通过自然对话帮助用户练习日常英语口语技能。
            
            指南：
            1. 始终用英语回应，即使用户使用其他语言
            2. 使用适合上下文的自然、对话化语言
            3. 温和地纠正重要的语法或词汇错误
            4. 提出后续问题，保持对话流畅
            5. 调整语言复杂度，以匹配用户的熟练程度
            6. 专注于当前的对话主题或场景
            7. 给予鼓励和支持
            8. 如果对话是特定场景的一部分，请保持在该场景的上下文中
            
            请保持回应简洁且有吸引力。
            """
        else:
            return """
            You are a helpful AI assistant.
            Answer the user's questions based on the provided context.
            If you don't know the answer, say so honestly.
            """
    
    def answer(self, 
              query: str, 
              context: List[Dict[str, Any]], 
              conversation_id: str,
              language: str = "english",
              role: str = "english_teacher",
              scene: Optional[Dict[str, Any]] = None,
              retrieval_service = None,
              index_service = None,
              index_name: Optional[str] = None) -> Dict[str, Any]:
        """
        生成回答
        
        Args:
            query: 用户问题
            context: 相关上下文信息
            conversation_id: 对话ID
            language: 语言
            role: 角色
            scene: 场景信息
            retrieval_service: 检索服务实例
            index_service: 索引服务实例
            index_name: 索引名称
            
        Returns:
            回答结果
        """
        try:
            # 获取或创建对话历史
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = []
                self.query_history[conversation_id] = []
                
            conversation_history = self.conversations[conversation_id]
            
            # 更新查询历史
            self.query_history[conversation_id].append(query)
            previous_queries = self.query_history[conversation_id]
            
            # 识别意图和实体
            intent = None
            entities = []
            try:
                # 导入文本处理服务
                from server.rag_service.utils.text_processing import TextProcessingService
                text_processing_service = TextProcessingService({})
                
                # 意图识别
                intent, confidence = text_processing_service._recognize_intent(query)
                logger.info(f"查询意图: {intent}, 置信度: {confidence}")
                
                # 实体提取
                entities = text_processing_service._extract_entities(query)
                logger.info(f"提取实体: {entities}")
                
                # 检测语言
                detected_language = text_processing_service._detect_language(query)
                if detected_language and detected_language != language:
                    language = detected_language
                    logger.info(f"检测到语言: {language}，调整响应语言")
            except Exception as e:
                logger.warning(f"意图和实体识别失败: {str(e)}")
            
            # 执行混合检索（如果提供了检索服务和索引服务）
            enhanced_context = context
            retrieval_results = []
            if retrieval_service and index_service and index_name:
                try:
                    # 根据意图选择检索类型
                    search_type = "hybrid"  # 默认值
                    
                    if intent == "command":
                        search_type = "keyword"  # 命令类查询使用关键词搜索
                    elif intent == "learning_question":
                        search_type = "hybrid"   # 学习类问题使用混合搜索
                    elif intent == "information_seeking":
                        search_type = "vector"   # 信息查询使用向量搜索
                    
                    # 根据实体构建过滤条件
                    filters = {}
                    if entities:
                        for entity in entities:
                            if entity["type"] == "TIME":
                                if "time_range" not in filters:
                                    filters["time_range"] = []
                                filters["time_range"].append(entity["text"])
                            elif entity["type"] == "PERSON":
                                if "author" not in filters:
                                    filters["author"] = []
                                filters["author"].append(entity["text"])
                            elif entity["type"] == "LOCATION":
                                if "location" not in filters:
                                    filters["location"] = []
                                filters["location"].append(entity["text"])
                    
                    # 获取检索结果
                    retrieval_results = retrieval_service.retrieve(
                        query=query,
                        index_service=index_service,
                        index_name=index_name,
                        top_k=self.top_k,
                        search_type=search_type,
                        previous_queries=previous_queries if self.enable_context_optimization else None,
                        filters=filters
                    )
                    
                    # 如果原始上下文为空，使用检索结果
                    if not enhanced_context and retrieval_results:
                        enhanced_context = retrieval_results
                    # 如果都有内容，合并结果并去重
                    elif retrieval_results:
                        # 简单去重方法：基于内容的哈希
                        content_hashes = {hash(item.get("content", "")): item for item in enhanced_context}
                        
                        for result in retrieval_results:
                            content_hash = hash(result.get("content", ""))
                            if content_hash not in content_hashes:
                                enhanced_context.append(result)
                                content_hashes[content_hash] = result
                                
                    logger.info(f"检索完成，共找到 {len(retrieval_results)} 个相关结果")
                except Exception as e:
                    logger.error(f"检索过程出错: {str(e)}")
                    # 检索失败时继续使用原始上下文
            
            # 准备系统提示
            system_prompt = self._generate_system_prompt(role, language)
            
            # 根据意图调整系统提示
            if intent:
                if intent == "command":
                    system_prompt += "\n\nThe user is trying to issue a command. Focus on providing clear instructions or responding to the command."
                elif intent == "learning_question":
                    system_prompt += "\n\nThe user is asking a learning-oriented question. Focus on providing educational content and explanations."
                elif intent == "information_seeking":
                    system_prompt += "\n\nThe user is seeking specific information. Focus on providing accurate and concise facts."
                elif intent == "casual_chat":
                    system_prompt += "\n\nThe user is engaging in casual conversation. Maintain a friendly and conversational tone."
            
            # 准备消息列表
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # 添加场景信息（如果有）
            if scene:
                scene_description = f"""
                Current conversation scene: {scene['name']}
                
                Description: {scene['description']}
                
                Difficulty level: {scene['difficulty']}
                
                Please engage with the user in this context.
                """
                messages.append({"role": "system", "content": scene_description})
            
            # 添加相关上下文
            if enhanced_context:
                context_text = "Relevant information:\n\n"
                # 按相关性排序（如果有排名信息）
                sorted_context = sorted(enhanced_context, key=lambda x: x.get("rank", 999))
                
                for i, item in enumerate(sorted_context):
                    context_text += f"[Document {i+1}]\n"
                    context_text += f"Content: {item['content']}\n"
                    
                    # 添加元数据信息（如果有）
                    if "metadata" in item:
                        meta = item["metadata"]
                        if "filename" in meta:
                            context_text += f"Source: {meta['filename']}\n"
                    
                    # 添加关键词匹配信息（如果有）
                    if "keyword_matches" in item and item["keyword_matches"]:
                        matches = item["keyword_matches"]
                        match_text = "Keywords: " + ", ".join([m["text"] for m in matches])
                        context_text += f"{match_text}\n"
                        
                    context_text += "\n"
                
                messages.append({"role": "system", "content": context_text})
            
            # 添加提取的实体信息
            if entities:
                entities_text = "Detected entities in the query:\n"
                for entity in entities:
                    entities_text += f"- {entity['text']} ({entity['type']})\n"
                messages.append({"role": "system", "content": entities_text})
            
            # 添加对话历史
            for message in conversation_history:
                messages.append(message)
                
            # 添加用户问题
            user_message = {"role": "user", "content": query}
            messages.append(user_message)
            
            # 添加思考链指令（如果启用高级功能）
            if self.enable_advanced_features:
                think_prompt = f"""
                Before answering, please think step-by-step about the user's question. 
                Begin your response with your thinking process, followed by '{self.thinking_separator}', 
                and then your actual answer to the user.
                """
                messages.append({"role": "system", "content": think_prompt})
            
            # 调用OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                stream=False
            )
            
            # 提取回答
            full_answer = response.choices[0].message.content.strip()
            
            # 处理带思考链的回答
            thinking = ""
            answer = full_answer
            
            if self.enable_advanced_features and self.thinking_separator in full_answer:
                parts = full_answer.split(self.thinking_separator, 1)
                thinking = parts[0].strip()
                answer = parts[1].strip()
            
            # 更新对话历史
            conversation_history.append(user_message)
            conversation_history.append({"role": "assistant", "content": answer})
            
            # 限制对话历史长度
            if len(conversation_history) > 20:  # 保留最近10轮对话
                conversation_history = conversation_history[-20:]
            
            # 限制查询历史长度
            if len(self.query_history[conversation_id]) > 10:
                self.query_history[conversation_id] = self.query_history[conversation_id][-10:]
                
            self.conversations[conversation_id] = conversation_history
            
            # 生成推荐问题
            recommended_questions = []
            if self.enable_advanced_features:
                recommended_questions = self.generate_recommended_questions(
                    user_question=query,
                    model_answer=answer,
                    retrieved_content=retrieval_results
                )
            
            return {
                "answer": answer,
                "thinking": thinking,
                "conversation_id": conversation_id,
                "retrieval_results": retrieval_results,
                "recommended_questions": recommended_questions,
                "intent": intent,
                "entities": entities,
                "language": language
            }
            
        except Exception as e:
            logger.error(f"生成回答失败: {str(e)}")
            raise RuntimeError(f"生成回答失败: {str(e)}")
    
    def generate_recommended_questions(self, 
                                     user_question: str, 
                                     model_answer: str, 
                                     retrieved_content: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        根据用户提问和检索到的内容生成推荐问题
        
        Args:
            user_question: 用户问题
            model_answer: 模型回答
            retrieved_content: 检索内容
        
        Returns:
            推荐问题列表
        """
        try:
            # 判断 retrieved_content 是否为空
            if not retrieved_content:
                formatted_references = "无相关上下文信息"
            else:
                # 格式化参考内容
                formatted_references = "\n".join([
                    f"[{i+1}] {item.get('content', '')}" 
                    for i, item in enumerate(retrieved_content)
                ])
    
            # 构造提示词
            prompt = f"""
            请根据以下用户提问、助手回答和检索到的内容，生成3个相关的推荐后续问题：
            
            用户提问：{user_question}
            
            助手回答：{model_answer}
            
            检索内容：{formatted_references}
    
            要求：
            1. 问题应与当前对话上下文相关，有助于拓展或深入讨论
            2. 问题应清晰简洁，便于继续对话
            3. 避免重复已回答的内容
            4. 如果对话涉及英语学习，可以围绕词汇、语法或场景用法提问
            5. 返回一个JSON对象，包含一个字段"recommended_questions"，值为问题列表
    
            输出格式示例：
            {{
              "recommended_questions": [
                "问题1描述",
                "问题2描述",
                "问题3描述"
              ]
            }}
            
            请严格按照上述格式返回JSON对象。
            """
            
            # 调用大模型生成推荐问题
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                stream=False,
            )
    
            # 提取生成的推荐问题
            if response.choices:
                response_text = response.choices[0].message.content
                try:
                    # 解析JSON响应
                    response_json = json.loads(response_text)
                    recommended_questions = response_json.get("recommended_questions", [])
                    return recommended_questions
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON response for recommended questions.")
                    return []
            return []
            
        except Exception as e:
            logger.error(f"生成推荐问题失败: {str(e)}")
            return []
    
    def generate_session_name(self, user_question: str) -> str:
        """
        根据用户问题生成会话名称
        
        Args:
            user_question: 用户问题
            
        Returns:
            会话名称
        """
        try:
            prompt = f"""
            请根据以下用户提问，生成一个简洁且具有代表性的会话名称：
            
            用户提问：{user_question}
    
            要求：
            1. 会话名称应简洁明了，不超过15个字符
            2. 能够概括用户提问的主题
            3. 返回一个JSON对象，包含一个字段"session_name"，值为生成的会话名称
    
            输出格式示例：
            {{
              "session_name": "会话名称内容"
            }}
    
            请严格按照上述格式返回JSON对象。
            """
            
            # 调用大模型生成会话名称
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                stream=False,
            )
    
            # 提取生成的会话名称
            if response.choices:
                response_text = response.choices[0].message.content
                try:
                    # 解析JSON响应
                    response_json = json.loads(response_text)
                    session_name = response_json.get("session_name")
                    return session_name if session_name else user_question[:15]
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON response for session name.")
                    return user_question[:15]
            return user_question[:15]
            
        except Exception as e:
            logger.error(f"生成会话名称失败: {str(e)}")
            return user_question[:15] if user_question else "新对话"
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        获取对话历史
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            对话历史列表
        """
        return self.conversations.get(conversation_id, [])
    
    def clear_conversation(self, conversation_id: str) -> None:
        """
        清除对话历史
        
        Args:
            conversation_id: 对话ID
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            # 同时清除查询历史
            if conversation_id in self.query_history:
                del self.query_history[conversation_id]
            logger.info(f"对话 {conversation_id} 已清除")
    
    def process_feedback(self, conversation_id: str, message_id: str, feedback: str) -> None:
        """
        处理用户反馈
        
        Args:
            conversation_id: 对话ID
            message_id: 消息ID
            feedback: 反馈内容
        """
        # 此处可实现反馈处理逻辑，如记录反馈、调整模型等
        logger.info(f"收到对话 {conversation_id} 消息 {message_id} 的反馈: {feedback}")
            
    def detect_language(self, text: str) -> str:
        """
        检测文本语言
        
        Args:
            text: 文本内容
            
        Returns:
            语言代码（"en"/"zh"等）
        """
        try:
            messages = [
                {"role": "system", "content": "Identify the language of the following text. Respond with the language code only (e.g., 'en' for English, 'zh' for Chinese)."},
                {"role": "user", "content": text}
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=10,
                temperature=0.1
            )
            
            language_code = response.choices[0].message.content.strip().lower()
            
            # 标准化语言代码
            if language_code in ["en", "english"]:
                return "en"
            elif language_code in ["zh", "chinese", "zh-cn"]:
                return "zh"
            else:
                return language_code
                
        except Exception as e:
            logger.error(f"语言检测失败: {str(e)}")
            return "en"  # 默认返回英语 