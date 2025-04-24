"""
场景定制服务
负责管理和提供对话场景，支持根据行业、难度等条件筛选
"""

import os
import json
import logging
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import uuid
import datetime

logger = logging.getLogger(__name__)

class SceneCustomizationService:
    """场景定制服务类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化场景定制服务
        
        Args:
            config: 配置参数
        """
        self.config = config
        
        # 场景存储路径
        self.scenes_dir = config.get("scenes_dir", "scenes")
        Path(self.scenes_dir).mkdir(exist_ok=True, parents=True)
        
        # 用户自定义场景存储路径
        self.user_scenes_dir = config.get("user_scenes_dir", "user_scenes")
        Path(self.user_scenes_dir).mkdir(exist_ok=True, parents=True)
        
        # 加载预设场景
        self.preset_scenes = self._load_preset_scenes()
        
        # 加载用户自定义场景
        self.user_scenes = self._load_user_scenes()
        
        # 难度级别
        self.difficulty_levels = [
            {"id": "beginner", "name": "初级", "description": "适合英语初学者，使用简单词汇和短句"},
            {"id": "intermediate", "name": "中级", "description": "适合有一定英语基础的学习者，使用中等复杂度的词汇和句型"},
            {"id": "advanced", "name": "高级", "description": "适合英语熟练者，使用复杂词汇和长句"},
            {"id": "expert", "name": "专家级", "description": "适合英语精通者，使用专业词汇和复杂表达"}
        ]
        
        # 行业类别
        self.industry_categories = [
            {"id": "general", "name": "通用", "description": "日常生活和一般对话场景"},
            {"id": "travel", "name": "旅游", "description": "旅游和出行相关场景"},
            {"id": "business", "name": "商务", "description": "商务和职场相关场景"},
            {"id": "education", "name": "教育", "description": "教育和学术相关场景"},
            {"id": "healthcare", "name": "医疗", "description": "医疗和健康相关场景"},
            {"id": "technology", "name": "科技", "description": "科技和IT相关场景"},
            {"id": "entertainment", "name": "娱乐", "description": "娱乐和休闲相关场景"}
        ]
        
        # 场景类型
        self.scene_types = [
            {"id": "dialogue", "name": "对话", "description": "两人或多人对话场景"},
            {"id": "monologue", "name": "独白", "description": "单人演讲或叙述场景"},
            {"id": "interview", "name": "面试", "description": "问答式面试场景"},
            {"id": "presentation", "name": "演示", "description": "演示和讲解场景"},
            {"id": "roleplay", "name": "角色扮演", "description": "特定角色的扮演场景"}
        ]
    
    def _get_scene_path(self, scene_id: str, is_user_scene: bool = False) -> Path:
        """
        获取场景文件路径
        
        Args:
            scene_id: 场景ID
            is_user_scene: 是否为用户自定义场景
            
        Returns:
            场景文件路径
        """
        if is_user_scene:
            return Path(self.user_scenes_dir) / f"{scene_id}.json"
        else:
            return Path(self.scenes_dir) / f"{scene_id}.json"
    
    def _load_preset_scenes(self) -> Dict[str, Any]:
        """
        加载预设场景
        
        Returns:
            预设场景字典
        """
        scenes = {}
        
        # 如果文件夹不存在但在配置中有预设场景，创建默认场景
        if not os.path.exists(self.scenes_dir) and not os.listdir(self.scenes_dir):
            self._create_default_scenes()
            
        # 加载所有场景文件
        scene_files = Path(self.scenes_dir).glob("*.json")
        for scene_file in scene_files:
            try:
                with open(scene_file, "r", encoding="utf-8") as f:
                    scene_data = json.load(f)
                    scenes[scene_data["id"]] = scene_data
            except Exception as e:
                logger.error(f"加载场景文件失败: {scene_file}, 错误: {str(e)}")
                
        return scenes
    
    def _load_user_scenes(self) -> Dict[str, Any]:
        """
        加载用户自定义场景
        
        Returns:
            用户自定义场景字典
        """
        scenes = {}
        
        # 加载所有用户场景文件
        if os.path.exists(self.user_scenes_dir):
            scene_files = Path(self.user_scenes_dir).glob("*.json")
            for scene_file in scene_files:
                try:
                    with open(scene_file, "r", encoding="utf-8") as f:
                        scene_data = json.load(f)
                        scenes[scene_data["id"]] = scene_data
                except Exception as e:
                    logger.error(f"加载用户场景文件失败: {scene_file}, 错误: {str(e)}")
                    
        return scenes
    
    def _create_default_scenes(self) -> None:
        """
        创建默认场景
        """
        # 确保目录存在
        Path(self.scenes_dir).mkdir(exist_ok=True)
        
        # 默认场景数据
        default_scenes = [
            {
                "id": "daily_conversation",
                "name": "日常对话",
                "description": "一般日常生活中的简单对话，涵盖问候、天气、爱好等话题",
                "difficulty": "beginner",
                "industry": "general",
                "type": "dialogue",
                "topics": ["greeting", "weather", "hobbies", "family", "food"],
                "sample_questions": [
                    "How's the weather today?",
                    "What are your hobbies?",
                    "Tell me about your family.",
                    "What's your favorite food?"
                ],
                "key_phrases": [
                    "Nice to meet you",
                    "How are you doing",
                    "What do you like to do",
                    "Tell me about yourself"
                ],
                "created_at": datetime.datetime.now().isoformat(),
                "is_preset": True
            },
            {
                "id": "travel_airport",
                "name": "机场对话",
                "description": "在机场中常见的对话场景，包括值机、安检、登机等环节",
                "difficulty": "intermediate",
                "industry": "travel",
                "type": "dialogue",
                "topics": ["check-in", "security", "boarding", "customs", "baggage"],
                "sample_questions": [
                    "Where is the check-in counter?",
                    "How many bags can I check in?",
                    "What time is my flight boarding?",
                    "Where can I find my gate number?"
                ],
                "key_phrases": [
                    "boarding pass",
                    "security checkpoint",
                    "departure gate",
                    "customs declaration"
                ],
                "created_at": datetime.datetime.now().isoformat(),
                "is_preset": True
            },
            {
                "id": "business_meeting",
                "name": "商务会议",
                "description": "商务会议场景，包括会议安排、项目讨论、数据分析等环节",
                "difficulty": "advanced",
                "industry": "business",
                "type": "dialogue",
                "topics": ["meeting", "project", "proposal", "negotiation", "presentation"],
                "sample_questions": [
                    "Could you give us an update on the project?",
                    "What are the key findings from your analysis?",
                    "How do you propose we address this issue?",
                    "When can we expect the final deliverables?"
                ],
                "key_phrases": [
                    "quarterly results",
                    "action items",
                    "moving forward",
                    "on the same page"
                ],
                "created_at": datetime.datetime.now().isoformat(),
                "is_preset": True
            },
            {
                "id": "job_interview",
                "name": "求职面试",
                "description": "求职面试场景，包括自我介绍、经历讨论、技能评估等环节",
                "difficulty": "advanced",
                "industry": "business",
                "type": "interview",
                "topics": ["introduction", "experience", "skills", "strengths", "career goals"],
                "sample_questions": [
                    "Tell me about yourself.",
                    "What experience do you have in this field?",
                    "What are your greatest strengths and weaknesses?",
                    "Where do you see yourself in five years?"
                ],
                "key_phrases": [
                    "work experience",
                    "team player",
                    "problem solving",
                    "career development"
                ],
                "created_at": datetime.datetime.now().isoformat(),
                "is_preset": True
            },
            {
                "id": "restaurant_dining",
                "name": "餐厅用餐",
                "description": "餐厅用餐场景，包括点餐、付款、特殊要求等环节",
                "difficulty": "intermediate",
                "industry": "general",
                "type": "dialogue",
                "topics": ["ordering", "menu", "payment", "reservation", "special requests"],
                "sample_questions": [
                    "Could I see the menu, please?",
                    "What's today's special?",
                    "How would you like your steak cooked?",
                    "Could we have the bill, please?"
                ],
                "key_phrases": [
                    "table for two",
                    "menu recommendations",
                    "food allergies",
                    "split the bill"
                ],
                "created_at": datetime.datetime.now().isoformat(),
                "is_preset": True
            }
        ]
        
        # 写入默认场景文件
        for scene in default_scenes:
            scene_path = self._get_scene_path(scene["id"])
            with open(scene_path, "w", encoding="utf-8") as f:
                json.dump(scene, f, ensure_ascii=False, indent=2)
                
        # 加载到内存
        self.preset_scenes = {scene["id"]: scene for scene in default_scenes}
    
    def get_all_scenes(self) -> List[Dict[str, Any]]:
        """
        获取所有场景
        
        Returns:
            场景列表
        """
        all_scenes = list(self.preset_scenes.values()) + list(self.user_scenes.values())
        return sorted(all_scenes, key=lambda x: x["name"])
    
    def get_scene_by_id(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取场景
        
        Args:
            scene_id: 场景ID
            
        Returns:
            场景数据
        """
        # 先查找预设场景
        if scene_id in self.preset_scenes:
            return self.preset_scenes[scene_id]
            
        # 再查找用户自定义场景
        if scene_id in self.user_scenes:
            return self.user_scenes[scene_id]
            
        return None
    
    def filter_scenes(self, 
                    difficulty: Optional[str] = None,
                    industry: Optional[str] = None,
                    scene_type: Optional[str] = None,
                    search_text: Optional[str] = None,
                    include_preset: bool = True,
                    include_user: bool = True) -> List[Dict[str, Any]]:
        """
        筛选场景
        
        Args:
            difficulty: 难度级别ID
            industry: 行业类别ID
            scene_type: 场景类型ID
            search_text: 搜索文本
            include_preset: 是否包含预设场景
            include_user: 是否包含用户自定义场景
            
        Returns:
            场景列表
        """
        # 收集要筛选的场景
        scenes_to_filter = []
        
        if include_preset:
            scenes_to_filter.extend(self.preset_scenes.values())
            
        if include_user:
            scenes_to_filter.extend(self.user_scenes.values())
            
        # 应用过滤条件
        filtered_scenes = scenes_to_filter
        
        # 按难度筛选
        if difficulty:
            filtered_scenes = [s for s in filtered_scenes if s.get("difficulty") == difficulty]
            
        # 按行业筛选
        if industry:
            filtered_scenes = [s for s in filtered_scenes if s.get("industry") == industry]
            
        # 按类型筛选
        if scene_type:
            filtered_scenes = [s for s in filtered_scenes if s.get("type") == scene_type]
            
        # 按文本搜索
        if search_text:
            search_pattern = re.compile(re.escape(search_text), re.IGNORECASE)
            
            def match_text(scene):
                """在场景名称和描述中搜索文本"""
                name = scene.get("name", "")
                desc = scene.get("description", "")
                topics = ", ".join(scene.get("topics", []))
                return (search_pattern.search(name) is not None or 
                        search_pattern.search(desc) is not None or
                        search_pattern.search(topics) is not None)
            
            filtered_scenes = [s for s in filtered_scenes if match_text(s)]
            
        # 按名称排序
        return sorted(filtered_scenes, key=lambda x: x["name"])
    
    def create_scene(self, scene_data: Dict[str, Any], is_user_scene: bool = True) -> str:
        """
        创建新场景
        
        Args:
            scene_data: 场景数据
            is_user_scene: 是否为用户自定义场景
            
        Returns:
            场景ID
        """
        # 生成唯一ID
        scene_id = scene_data.get("id")
        if not scene_id:
            scene_id = str(uuid.uuid4())
            
        # 构建完整场景数据
        complete_scene = {
            "id": scene_id,
            "name": scene_data.get("name", "新场景"),
            "description": scene_data.get("description", ""),
            "difficulty": scene_data.get("difficulty", "intermediate"),
            "industry": scene_data.get("industry", "general"),
            "type": scene_data.get("type", "dialogue"),
            "topics": scene_data.get("topics", []),
            "sample_questions": scene_data.get("sample_questions", []),
            "key_phrases": scene_data.get("key_phrases", []),
            "created_at": datetime.datetime.now().isoformat(),
            "is_preset": not is_user_scene
        }
        
        # 保存场景文件
        scene_path = self._get_scene_path(scene_id, is_user_scene)
        with open(scene_path, "w", encoding="utf-8") as f:
            json.dump(complete_scene, f, ensure_ascii=False, indent=2)
            
        # 更新内存中的场景数据
        if is_user_scene:
            self.user_scenes[scene_id] = complete_scene
        else:
            self.preset_scenes[scene_id] = complete_scene
            
        return scene_id
    
    def update_scene(self, scene_id: str, scene_data: Dict[str, Any]) -> bool:
        """
        更新场景
        
        Args:
            scene_id: 场景ID
            scene_data: 场景数据
            
        Returns:
            是否成功
        """
        # 检查场景是否存在
        existing_scene = self.get_scene_by_id(scene_id)
        if not existing_scene:
            return False
            
        # 确定是否为用户自定义场景
        is_user_scene = scene_id in self.user_scenes
        
        # 如果是预设场景但非管理员尝试更新，则拒绝
        if not is_user_scene and not scene_data.get("is_admin", False):
            return False
            
        # 更新场景数据
        updated_scene = existing_scene.copy()
        updated_scene.update({
            "name": scene_data.get("name", existing_scene["name"]),
            "description": scene_data.get("description", existing_scene.get("description", "")),
            "difficulty": scene_data.get("difficulty", existing_scene.get("difficulty", "intermediate")),
            "industry": scene_data.get("industry", existing_scene.get("industry", "general")),
            "type": scene_data.get("type", existing_scene.get("type", "dialogue")),
            "topics": scene_data.get("topics", existing_scene.get("topics", [])),
            "sample_questions": scene_data.get("sample_questions", existing_scene.get("sample_questions", [])),
            "key_phrases": scene_data.get("key_phrases", existing_scene.get("key_phrases", [])),
            "updated_at": datetime.datetime.now().isoformat()
        })
        
        # 保存场景文件
        scene_path = self._get_scene_path(scene_id, is_user_scene)
        with open(scene_path, "w", encoding="utf-8") as f:
            json.dump(updated_scene, f, ensure_ascii=False, indent=2)
            
        # 更新内存中的场景数据
        if is_user_scene:
            self.user_scenes[scene_id] = updated_scene
        else:
            self.preset_scenes[scene_id] = updated_scene
            
        return True
    
    def delete_scene(self, scene_id: str, is_admin: bool = False) -> bool:
        """
        删除场景
        
        Args:
            scene_id: 场景ID
            is_admin: 是否为管理员
            
        Returns:
            是否成功
        """
        # 检查场景是否存在
        if scene_id in self.preset_scenes:
            # 只有管理员才能删除预设场景
            if not is_admin:
                return False
                
            # 删除预设场景
            scene_path = self._get_scene_path(scene_id)
            try:
                os.remove(scene_path)
                del self.preset_scenes[scene_id]
                return True
            except Exception as e:
                logger.error(f"删除预设场景失败: {scene_id}, 错误: {str(e)}")
                return False
                
        elif scene_id in self.user_scenes:
            # 删除用户自定义场景
            scene_path = self._get_scene_path(scene_id, True)
            try:
                os.remove(scene_path)
                del self.user_scenes[scene_id]
                return True
            except Exception as e:
                logger.error(f"删除用户场景失败: {scene_id}, 错误: {str(e)}")
                return False
                
        # 场景不存在
        return False
    
    def get_difficulty_levels(self) -> List[Dict[str, Any]]:
        """
        获取难度级别列表
        
        Returns:
            难度级别列表
        """
        return self.difficulty_levels
    
    def get_industry_categories(self) -> List[Dict[str, Any]]:
        """
        获取行业类别列表
        
        Returns:
            行业类别列表
        """
        return self.industry_categories
    
    def get_scene_types(self) -> List[Dict[str, Any]]:
        """
        获取场景类型列表
        
        Returns:
            场景类型列表
        """
        return self.scene_types
    
    def extract_scene_from_document(self, document_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        从文档中提取场景
        
        Args:
            document_text: 文档文本
            metadata: 文档元数据
            
        Returns:
            场景数据
        """
        # 从文档标题生成场景名称
        name = metadata.get("title", "自动生成场景")
        
        # 提取前200个字符作为描述
        description = document_text[:200] + "..." if len(document_text) > 200 else document_text
        
        # 基于文本长度和复杂度推断难度
        word_count = len(document_text.split())
        complex_word_pattern = re.compile(r'\b\w{8,}\b')  # 8个或更多字符的单词
        complex_words = len(complex_word_pattern.findall(document_text))
        
        if word_count < 300:
            difficulty = "beginner"
        elif word_count < 800:
            difficulty = "intermediate"
        else:
            difficulty = "advanced" if complex_words > 20 else "intermediate"
            
        # 尝试确定行业类别
        industry_keywords = {
            "business": ["company", "market", "business", "corporate", "management", "finance", "strategy"],
            "travel": ["travel", "tourism", "hotel", "flight", "destination", "vacation", "tour"],
            "education": ["school", "university", "education", "learning", "student", "teacher", "academic"],
            "healthcare": ["health", "medical", "doctor", "patient", "disease", "treatment", "hospital"],
            "technology": ["technology", "software", "hardware", "digital", "computer", "internet", "data"]
        }
        
        # 计算每个行业的关键词匹配度
        industry_scores = {}
        for industry, keywords in industry_keywords.items():
            score = sum(1 for kw in keywords if kw.lower() in document_text.lower())
            industry_scores[industry] = score
            
        # 选择匹配度最高的行业
        if industry_scores:
            max_industry = max(industry_scores.items(), key=lambda x: x[1])
            industry = max_industry[0] if max_industry[1] > 0 else "general"
        else:
            industry = "general"
            
        # 提取可能的对话主题
        topics = []
        for sentence in document_text.split("."):
            words = sentence.strip().split()
            if 3 <= len(words) <= 10:  # 适当长度的句子可能是主题
                topics.append(" ".join(words))
        
        # 限制主题数量
        topics = topics[:5]
        
        # 提取示例问题（以问号结尾的句子）
        sample_questions = []
        question_pattern = re.compile(r'([^.!?]+\?)')
        questions = question_pattern.findall(document_text)
        
        for q in questions[:5]:  # 最多5个问题
            q = q.strip()
            if 10 <= len(q) <= 100:  # 适当长度的问题
                sample_questions.append(q)
        
        # 提取关键短语
        key_phrases = []
        phrases = re.findall(r'\b(\w+\s+\w+\s+\w+)\b', document_text)  # 三个单词的短语
        phrase_freq = {}
        
        for phrase in phrases:
            phrase = phrase.lower()
            if phrase in phrase_freq:
                phrase_freq[phrase] += 1
            else:
                phrase_freq[phrase] = 1
                
        # 选择最常见的短语
        sorted_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)
        key_phrases = [phrase for phrase, freq in sorted_phrases[:5]]
        
        # 构建场景数据
        scene_data = {
            "name": name,
            "description": description,
            "difficulty": difficulty,
            "industry": industry,
            "type": "dialogue",  # 默认为对话类型
            "topics": topics,
            "sample_questions": sample_questions,
            "key_phrases": key_phrases,
            "source_document": metadata.get("filename", ""),
            "is_auto_generated": True
        }
        
        return scene_data
    
    def recommend_scenes_for_user(self, user_level: str, preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        为用户推荐场景
        
        Args:
            user_level: 用户级别
            preferences: 用户偏好
                {
                    "preferred_industries": [],
                    "preferred_topics": [],
                    "avoided_topics": []
                }
                
        Returns:
            推荐场景列表
        """
        # 根据用户级别筛选难度适合的场景
        level_to_difficulties = {
            "beginner": ["beginner"],
            "intermediate": ["beginner", "intermediate"],
            "advanced": ["intermediate", "advanced"],
            "expert": ["advanced", "expert"]
        }
        
        suitable_difficulties = level_to_difficulties.get(user_level, ["beginner", "intermediate"])
        
        # 筛选符合难度的场景
        suitable_scenes = []
        for scene in self.get_all_scenes():
            if scene.get("difficulty") in suitable_difficulties:
                suitable_scenes.append(scene)
                
        # 如果用户有行业偏好，增加匹配行业的场景分数
        preferred_industries = preferences.get("preferred_industries", [])
        
        # 为场景计算推荐分数
        scene_scores = {}
        for scene in suitable_scenes:
            score = 0
            
            # 难度匹配加分
            if scene.get("difficulty") == user_level:
                score += 10
                
            # 行业偏好加分
            if scene.get("industry") in preferred_industries:
                score += 5
                
            # 主题偏好加分
            preferred_topics = preferences.get("preferred_topics", [])
            scene_topics = scene.get("topics", [])
            
            for topic in preferred_topics:
                if any(topic.lower() in t.lower() for t in scene_topics):
                    score += 3
                    
            # 避免主题减分
            avoided_topics = preferences.get("avoided_topics", [])
            
            for topic in avoided_topics:
                if any(topic.lower() in t.lower() for t in scene_topics):
                    score -= 5
                    
            scene_scores[scene["id"]] = score
            
        # 按分数排序
        sorted_scenes = sorted(suitable_scenes, key=lambda x: scene_scores.get(x["id"], 0), reverse=True)
        
        # 返回前10个推荐场景
        return sorted_scenes[:10] 