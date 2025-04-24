"""
场景服务模块
负责管理对话场景、场景提取和场景定制
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, List, Optional
import uuid
import re

logger = logging.getLogger(__name__)

class SceneService:
    """场景服务类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化场景服务
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.scenes_dir = config.get("scenes_dir", "data/scenes")
        
        # 创建场景目录
        os.makedirs(self.scenes_dir, exist_ok=True)
        
        # 场景缓存
        self._scenes_cache = {}
        self._last_cache_update = None
        
        # 默认场景
        self.default_scenes = [
            {
                "id": "daily_conversation",
                "name": "日常对话",
                "description": "日常生活中的常见对话场景，适合练习基本的社交表达和日常用语。",
                "difficulty": "beginner",
                "category": "daily",
                "language": "english",
                "example_topics": ["greeting", "weather", "hobbies", "family", "food"],
                "example_questions": [
                    "How's the weather today?",
                    "What do you like to do in your free time?",
                    "Can you tell me about your family?",
                    "What's your favorite food?"
                ]
            },
            {
                "id": "travel_abroad",
                "name": "旅游出行",
                "description": "出国旅游中的各种场景，包括订票、入住酒店、问路、购物和餐厅点餐等。",
                "difficulty": "intermediate",
                "category": "travel",
                "language": "english",
                "example_topics": ["airport", "hotel", "restaurant", "shopping", "transportation"],
                "example_questions": [
                    "Could you recommend a good hotel in the city center?",
                    "How can I get to the museum from here?",
                    "I'd like to book a table for two tonight.",
                    "Do you have this in a different size?"
                ]
            },
            {
                "id": "business_meeting",
                "name": "商务会议",
                "description": "商务环境中的会议场景，包括介绍、讨论、谈判和演示等。",
                "difficulty": "advanced",
                "category": "business",
                "language": "english",
                "example_topics": ["presentation", "negotiation", "collaboration", "project management"],
                "example_questions": [
                    "Could you walk us through your proposal?",
                    "What are the key deliverables for this project?",
                    "I'd like to discuss the terms of our agreement.",
                    "How do you plan to address these challenges?"
                ]
            },
            {
                "id": "academic_discussion",
                "name": "学术讨论",
                "description": "学术环境中的讨论场景，适合练习学术英语和专业术语表达。",
                "difficulty": "advanced",
                "category": "academic",
                "language": "english",
                "example_topics": ["research", "thesis", "publication", "conference"],
                "example_questions": [
                    "Could you elaborate on your research methodology?",
                    "What are the implications of these findings?",
                    "How does your work contribute to the existing literature?",
                    "Can you explain the theoretical framework you're using?"
                ]
            },
            {
                "id": "job_interview",
                "name": "求职面试",
                "description": "求职面试场景，包括自我介绍、回答专业问题和询问公司情况等。",
                "difficulty": "intermediate",
                "category": "career",
                "language": "english",
                "example_topics": ["self-introduction", "work experience", "skills", "career goals"],
                "example_questions": [
                    "Can you tell me about yourself?",
                    "What are your strengths and weaknesses?",
                    "Why are you interested in this position?",
                    "Where do you see yourself in five years?"
                ]
            },
            {
                "id": "healthcare",
                "name": "医疗健康",
                "description": "医疗场所中的对话场景，包括看病、描述症状和了解治疗方案等。",
                "difficulty": "intermediate",
                "category": "healthcare",
                "language": "english",
                "example_topics": ["symptoms", "diagnosis", "treatment", "medication"],
                "example_questions": [
                    "What symptoms are you experiencing?",
                    "How long have you been feeling this way?",
                    "Have you taken any medication?",
                    "Do you have any allergies?"
                ]
            },
            {
                "id": "shopping",
                "name": "购物消费",
                "description": "购物场景中的对话，包括询问价格、讨价还价和退换货等。",
                "difficulty": "beginner",
                "category": "daily",
                "language": "english",
                "example_topics": ["price", "bargaining", "payment", "refund"],
                "example_questions": [
                    "How much does this cost?",
                    "Do you have this in other colors?",
                    "Can I try this on?",
                    "What's your return policy?"
                ]
            },
            {
                "id": "restaurant",
                "name": "餐厅用餐",
                "description": "餐厅场景中的对话，包括预订座位、点餐、支付和投诉等。",
                "difficulty": "beginner",
                "category": "daily",
                "language": "english",
                "example_topics": ["reservation", "ordering", "payment", "complaint"],
                "example_questions": [
                    "I'd like to make a reservation for tonight.",
                    "What do you recommend on the menu?",
                    "Could I have the bill, please?",
                    "This dish isn't what I ordered."
                ]
            },
            {
                "id": "hotel_accommodation",
                "name": "酒店住宿",
                "description": "酒店住宿场景中的对话，包括预订、入住、询问设施和退房等。",
                "difficulty": "intermediate",
                "category": "travel",
                "language": "english",
                "example_topics": ["booking", "check-in", "facilities", "check-out"],
                "example_questions": [
                    "I'd like to book a room for next weekend.",
                    "What time is check-out?",
                    "Is breakfast included in the price?",
                    "Do you have a gym or swimming pool?"
                ]
            },
            {
                "id": "transportation",
                "name": "交通出行",
                "description": "交通出行场景中的对话，包括问路、乘坐公共交通和打车等。",
                "difficulty": "beginner",
                "category": "travel",
                "language": "english",
                "example_topics": ["directions", "public transport", "taxi", "ticket booking"],
                "example_questions": [
                    "How do I get to the city center from here?",
                    "When does the next train to London leave?",
                    "Could you take me to this address, please?",
                    "Is there a direct bus to the airport?"
                ]
            }
        ]
    
    def get_all_scenes(self, refresh_cache: bool = False) -> List[Dict[str, Any]]:
        """
        获取所有场景
        
        Args:
            refresh_cache: 是否刷新缓存
            
        Returns:
            场景列表
        """
        # 检查缓存是否需要刷新
        current_time = datetime.datetime.now()
        if (self._last_cache_update is None or 
            refresh_cache or 
            (current_time - self._last_cache_update).total_seconds() > 300):  # 5分钟缓存
            
            self._refresh_scenes_cache()
        
        # 返回场景列表（默认场景 + 自定义场景）
        all_scenes = self.default_scenes.copy()
        all_scenes.extend(list(self._scenes_cache.values()))
        
        return all_scenes
    
    def _refresh_scenes_cache(self) -> None:
        """刷新场景缓存"""
        self._scenes_cache = {}
        
        # 读取场景目录下的所有JSON文件
        for filename in os.listdir(self.scenes_dir):
            if filename.endswith(".json"):
                try:
                    file_path = os.path.join(self.scenes_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        scene_data = json.load(f)
                        
                    # 检查是否有有效ID
                    if "id" in scene_data:
                        self._scenes_cache[scene_data["id"]] = scene_data
                        
                except Exception as e:
                    logger.error(f"加载场景文件失败 {filename}: {str(e)}")
        
        self._last_cache_update = datetime.datetime.now()
        logger.info(f"刷新场景缓存完成，共加载 {len(self._scenes_cache)} 个自定义场景")
    
    def get_scene_by_id(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取场景
        
        Args:
            scene_id: 场景ID
            
        Returns:
            场景数据，不存在则返回None
        """
        # 先检查默认场景
        for scene in self.default_scenes:
            if scene["id"] == scene_id:
                return scene
        
        # 然后检查自定义场景
        if scene_id in self._scenes_cache:
            return self._scenes_cache[scene_id]
        
        # 如果缓存中没有，尝试直接从文件中读取
        file_path = os.path.join(self.scenes_dir, f"{scene_id}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"读取场景文件失败 {scene_id}: {str(e)}")
        
        return None
    
    def create_scene(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建新场景
        
        Args:
            scene_data: 场景数据
            
        Returns:
            创建后的场景数据
        """
        # 生成唯一ID
        if "id" not in scene_data or not scene_data["id"]:
            scene_data["id"] = f"custom_{uuid.uuid4().hex[:8]}"
        
        # 添加创建时间
        scene_data["created_at"] = datetime.datetime.now().isoformat()
        
        # 验证必要字段
        required_fields = ["name", "description", "difficulty", "category", "language"]
        for field in required_fields:
            if field not in scene_data:
                raise ValueError(f"缺少必要字段: {field}")
        
        # 保存到文件
        file_path = os.path.join(self.scenes_dir, f"{scene_data['id']}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(scene_data, f, ensure_ascii=False, indent=2)
        
        # 更新缓存
        self._scenes_cache[scene_data["id"]] = scene_data
        self._last_cache_update = datetime.datetime.now()
        
        return scene_data
    
    def update_scene(self, scene_id: str, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新场景
        
        Args:
            scene_id: 场景ID
            scene_data: 场景数据
            
        Returns:
            更新后的场景数据
        """
        # 检查场景是否存在
        existing_scene = self.get_scene_by_id(scene_id)
        if not existing_scene:
            raise ValueError(f"场景不存在: {scene_id}")
        
        # 检查是否是默认场景（不允许修改默认场景）
        for scene in self.default_scenes:
            if scene["id"] == scene_id:
                raise ValueError(f"不能修改默认场景: {scene_id}")
        
        # 更新数据，保留ID
        scene_data["id"] = scene_id
        
        # 保留创建时间
        if "created_at" in existing_scene:
            scene_data["created_at"] = existing_scene["created_at"]
        
        # 添加更新时间
        scene_data["updated_at"] = datetime.datetime.now().isoformat()
        
        # 保存到文件
        file_path = os.path.join(self.scenes_dir, f"{scene_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(scene_data, f, ensure_ascii=False, indent=2)
        
        # 更新缓存
        self._scenes_cache[scene_id] = scene_data
        self._last_cache_update = datetime.datetime.now()
        
        return scene_data
    
    def delete_scene(self, scene_id: str) -> bool:
        """
        删除场景
        
        Args:
            scene_id: 场景ID
            
        Returns:
            是否删除成功
        """
        # 检查是否是默认场景（不允许删除默认场景）
        for scene in self.default_scenes:
            if scene["id"] == scene_id:
                raise ValueError(f"不能删除默认场景: {scene_id}")
        
        # 检查是否存在
        file_path = os.path.join(self.scenes_dir, f"{scene_id}.json")
        if not os.path.exists(file_path):
            return False
        
        # 删除文件
        try:
            os.remove(file_path)
            
            # 更新缓存
            if scene_id in self._scenes_cache:
                del self._scenes_cache[scene_id]
                self._last_cache_update = datetime.datetime.now()
            
            return True
        except Exception as e:
            logger.error(f"删除场景文件失败 {scene_id}: {str(e)}")
            return False
    
    def filter_scenes(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        根据条件筛选场景
        
        Args:
            criteria: 筛选条件，可包含difficulty, category, language等
            
        Returns:
            符合条件的场景列表
        """
        all_scenes = self.get_all_scenes()
        filtered_scenes = all_scenes
        
        # 按难度筛选
        if "difficulty" in criteria and criteria["difficulty"]:
            difficulties = criteria["difficulty"]
            if isinstance(difficulties, str):
                difficulties = [difficulties]
            filtered_scenes = [s for s in filtered_scenes if s.get("difficulty") in difficulties]
        
        # 按类别筛选
        if "category" in criteria and criteria["category"]:
            categories = criteria["category"]
            if isinstance(categories, str):
                categories = [categories]
            filtered_scenes = [s for s in filtered_scenes if s.get("category") in categories]
        
        # 按语言筛选
        if "language" in criteria and criteria["language"]:
            languages = criteria["language"]
            if isinstance(languages, str):
                languages = [languages]
            filtered_scenes = [s for s in filtered_scenes if s.get("language") in languages]
        
        # 按关键词筛选
        if "keywords" in criteria and criteria["keywords"]:
            keywords = criteria["keywords"].lower()
            filtered_scenes = [
                s for s in filtered_scenes if (
                    keywords in s.get("name", "").lower() or
                    keywords in s.get("description", "").lower() or
                    any(keywords in topic.lower() for topic in s.get("example_topics", []))
                )
            ]
        
        return filtered_scenes
    
    def extract_scene_from_document(self, document_text: str, document_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        从文档内容中提取场景
        
        Args:
            document_text: 文档文本内容
            document_metadata: 文档元数据
            
        Returns:
            提取的场景数据
        """
        # 基础场景信息
        scene_id = f"doc_{uuid.uuid4().hex[:8]}"
        scene_name = document_metadata.get("title", "自动提取场景")
        
        # 尝试从文件名推断场景类别
        filename = document_metadata.get("filename", "").lower()
        
        category_mapping = {
            "travel": "travel",
            "business": "business",
            "academic": "academic",
            "job": "career",
            "interview": "career",
            "health": "healthcare",
            "medical": "healthcare",
            "shop": "daily",
            "restaurant": "daily",
            "hotel": "travel",
            "transport": "travel"
        }
        
        detected_category = "other"
        for keyword, category in category_mapping.items():
            if keyword in filename:
                detected_category = category
                break
        
        # 尝试从内容推断难度
        word_count = len(re.findall(r'\b\w+\b', document_text))
        avg_word_length = sum(len(word) for word in re.findall(r'\b\w+\b', document_text)) / max(word_count, 1)
        
        difficulty = "intermediate"  # 默认难度
        if avg_word_length < 4.5:
            difficulty = "beginner"
        elif avg_word_length > 6.0:
            difficulty = "advanced"
        
        # 提取示例主题（使用文档中出现的名词短语）
        example_topics = []
        noun_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+){0,2}\b', document_text)
        for phrase in sorted(set(noun_phrases), key=lambda x: document_text.count(x), reverse=True):
            if len(example_topics) < 5:
                example_topics.append(phrase)
        
        # 提取示例问题（查找文档中的问句）
        example_questions = []
        questions = re.findall(r'([A-Z][^.!?]*\?)', document_text)
        for question in questions:
            if (len(question) > 10 and 
                len(question) < 100 and 
                len(example_questions) < 4):
                example_questions.append(question)
        
        # 如果没有提取到足够的问题，添加一些通用问题
        if len(example_questions) < 2:
            example_questions.extend([
                f"Can you tell me more about {example_topics[0]}?" if example_topics else "What can you tell me about this topic?",
                "What are the key points I should know about this?"
            ])
        
        # 构建场景数据
        scene_data = {
            "id": scene_id,
            "name": scene_name,
            "description": self._generate_scene_description(document_text[:500]),
            "difficulty": difficulty,
            "category": detected_category,
            "language": "english",  # 默认语言
            "example_topics": example_topics,
            "example_questions": example_questions,
            "source_document": document_metadata.get("filename", ""),
            "created_at": datetime.datetime.now().isoformat()
        }
        
        return scene_data
    
    def _generate_scene_description(self, document_intro: str) -> str:
        """
        生成场景描述
        
        Args:
            document_intro: 文档前500字符
            
        Returns:
            生成的场景描述
        """
        # 简单实现：使用文档前几句话作为描述
        sentences = re.split(r'[.!?]+', document_intro)
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if valid_sentences:
            description = ". ".join(valid_sentences[:2]) + "."
            return description
        else:
            return "自动从文档提取的对话场景。"
    
    def get_scene_categories(self) -> List[Dict[str, Any]]:
        """
        获取所有场景类别
        
        Returns:
            场景类别列表
        """
        categories = [
            {
                "id": "daily",
                "name": "日常生活",
                "description": "日常生活中的常见对话场景"
            },
            {
                "id": "travel",
                "name": "旅游出行",
                "description": "旅游和出行相关的对话场景"
            },
            {
                "id": "business",
                "name": "商务职场",
                "description": "商务和职场环境中的对话场景"
            },
            {
                "id": "academic",
                "name": "学术教育",
                "description": "学术和教育环境中的对话场景"
            },
            {
                "id": "career",
                "name": "求职就业",
                "description": "求职和就业相关的对话场景"
            },
            {
                "id": "healthcare",
                "name": "医疗健康",
                "description": "医疗和健康相关的对话场景"
            },
            {
                "id": "other",
                "name": "其他",
                "description": "其他类别的对话场景"
            }
        ]
        
        return categories
    
    def get_difficulty_levels(self) -> List[Dict[str, Any]]:
        """
        获取所有难度级别
        
        Returns:
            难度级别列表
        """
        levels = [
            {
                "id": "beginner",
                "name": "初级",
                "description": "适合英语初学者，使用简单词汇和语法"
            },
            {
                "id": "intermediate",
                "name": "中级",
                "description": "适合有一定英语基础的学习者，使用较为复杂的词汇和语法"
            },
            {
                "id": "advanced",
                "name": "高级",
                "description": "适合英语水平较高的学习者，使用复杂词汇和语法，包含专业术语"
            }
        ]
        
        return levels 