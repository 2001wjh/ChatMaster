"""
进度追踪服务
负责记录和分析用户学习进度，提供个性化学习建议
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, List, Optional
import uuid
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class ProgressTrackingService:
    """进度追踪服务类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化进度追踪服务
        
        Args:
            config: 配置参数
        """
        self.config = config
        
        # 数据存储路径
        self.data_dir = config.get("progress_data_dir", "user_progress")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 加载用户进度数据
        self.user_data = self._load_user_data()
        
        # 技能类别
        self.skill_categories = [
            "vocabulary",  # 词汇量
            "grammar",     # 语法
            "fluency",     # 流利度
            "pronunciation", # 发音
            "comprehension" # 理解能力
        ]
        
        # 难度级别
        self.difficulty_levels = ["beginner", "intermediate", "advanced", "expert"]
        
    def _get_user_data_path(self, user_id: str) -> str:
        """
        获取用户数据文件路径
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户数据文件路径
        """
        return os.path.join(self.data_dir, f"{user_id}.json")
    
    def _load_user_data(self) -> Dict[str, Any]:
        """
        加载所有用户的进度数据
        
        Returns:
            用户数据字典
        """
        user_data = {}
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                user_id = filename[:-5]  # 移除 .json 后缀
                user_data[user_id] = self._load_single_user_data(user_id)
        return user_data
        
    def _load_single_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        加载单个用户的进度数据
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户数据字典
        """
        file_path = self._get_user_data_path(user_id)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"加载用户数据失败: {user_id}")
                return self._create_default_user_data()
        return self._create_default_user_data()
    
    def _save_user_data(self, user_id: str) -> None:
        """
        保存用户数据
        
        Args:
            user_id: 用户ID
        """
        file_path = self._get_user_data_path(user_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.user_data[user_id], f, ensure_ascii=False, indent=2)
    
    def _create_default_user_data(self) -> Dict[str, Any]:
        """
        创建默认用户数据
        
        Returns:
            默认用户数据
        """
        return {
            "created_at": datetime.datetime.now().isoformat(),
            "last_active": datetime.datetime.now().isoformat(),
            "sessions": [],
            "skills": {
                category: {
                    "score": 0,
                    "progress": [],
                    "strengths": [],
                    "weaknesses": []
                } for category in self.skill_categories
            },
            "statistics": {
                "total_sessions": 0,
                "total_messages": 0,
                "total_duration": 0,
                "average_response_time": 0,
                "scene_distribution": {},
                "difficulty_distribution": {}
            },
            "recommendations": [],
            "current_level": "beginner"
        }
    
    def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户进度数据
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户进度数据
        """
        if user_id not in self.user_data:
            self.user_data[user_id] = self._create_default_user_data()
            self._save_user_data(user_id)
        
        # 更新最后活动时间
        self.user_data[user_id]["last_active"] = datetime.datetime.now().isoformat()
        self._save_user_data(user_id)
        
        return self.user_data[user_id]
    
    def record_session(self, 
                      user_id: str, 
                      session_data: Dict[str, Any]) -> str:
        """
        记录会话数据
        
        Args:
            user_id: 用户ID
            session_data: 会话数据
                {
                    "scene": "场景名称",
                    "difficulty": "难度级别",
                    "start_time": "开始时间",
                    "end_time": "结束时间",
                    "messages": [...],
                    "feedback": {...}
                }
                
        Returns:
            会话ID
        """
        # 获取或创建用户数据
        if user_id not in self.user_data:
            self.user_data[user_id] = self._create_default_user_data()
        
        # 生成会话ID
        session_id = str(uuid.uuid4())
        
        # 计算会话时长
        try:
            start_time = datetime.datetime.fromisoformat(session_data["start_time"])
            end_time = datetime.datetime.fromisoformat(session_data["end_time"])
            duration = (end_time - start_time).total_seconds()
        except (ValueError, KeyError):
            duration = 0
            logger.warning("计算会话时长失败")
        
        # 构建会话记录
        session_record = {
            "id": session_id,
            "scene": session_data.get("scene", "unknown"),
            "difficulty": session_data.get("difficulty", "beginner"),
            "start_time": session_data.get("start_time", datetime.datetime.now().isoformat()),
            "end_time": session_data.get("end_time", datetime.datetime.now().isoformat()),
            "duration": duration,
            "message_count": len(session_data.get("messages", [])),
            "feedback": session_data.get("feedback", {})
        }
        
        # 添加到用户数据
        self.user_data[user_id]["sessions"].append(session_record)
        
        # 更新统计数据
        stats = self.user_data[user_id]["statistics"]
        stats["total_sessions"] += 1
        stats["total_messages"] += session_record["message_count"]
        stats["total_duration"] += duration
        
        # 更新场景分布
        scene = session_record["scene"]
        if scene in stats["scene_distribution"]:
            stats["scene_distribution"][scene] += 1
        else:
            stats["scene_distribution"][scene] = 1
        
        # 更新难度分布
        difficulty = session_record["difficulty"]
        if difficulty in stats["difficulty_distribution"]:
            stats["difficulty_distribution"][difficulty] += 1
        else:
            stats["difficulty_distribution"][difficulty] = 1
        
        # 计算平均响应时间
        if stats["total_sessions"] > 0:
            stats["average_response_time"] = stats["total_duration"] / stats["total_sessions"]
        
        # 保存数据
        self._save_user_data(user_id)
        
        # 分析会话数据并更新技能评分
        self._analyze_session(user_id, session_record, session_data.get("messages", []))
        
        return session_id
    
    def _analyze_session(self, 
                        user_id: str, 
                        session_record: Dict[str, Any],
                        messages: List[Dict[str, Any]]) -> None:
        """
        分析会话数据，更新用户技能评分
        
        Args:
            user_id: 用户ID
            session_record: 会话记录
            messages: 消息列表
        """
        if not messages:
            return
        
        # 提取用户消息
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            return
        
        # 分析技能表现
        # 这里使用简单的算法，实际项目中可以使用更复杂的语言模型进行评估
        
        # 模拟技能评分更新
        difficulty_factor = {
            "beginner": 1.0,
            "intermediate": 1.2,
            "advanced": 1.5,
            "expert": 2.0
        }.get(session_record["difficulty"], 1.0)
        
        # 更新所有技能类别的分数
        for category in self.skill_categories:
            # 模拟评分计算
            # 实际项目中应使用ML模型评估用户表现
            base_score = np.random.normal(0.7, 0.15)  # 随机生成0.4~1.0之间的分数
            score = min(1.0, max(0.0, base_score * difficulty_factor))
            
            # 获取当前技能状态
            skill_data = self.user_data[user_id]["skills"][category]
            
            # 记录进步情况
            skill_data["progress"].append({
                "session_id": session_record["id"],
                "score": score,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # 保留最新的20条记录
            if len(skill_data["progress"]) > 20:
                skill_data["progress"] = skill_data["progress"][-20:]
            
            # 计算平均分数
            if skill_data["progress"]:
                avg_score = sum(item["score"] for item in skill_data["progress"]) / len(skill_data["progress"])
                skill_data["score"] = avg_score
        
        # 更新用户水平
        self._update_user_level(user_id)
        
        # 生成推荐
        self._generate_recommendations(user_id)
        
        # 保存数据
        self._save_user_data(user_id)
    
    def _update_user_level(self, user_id: str) -> None:
        """
        更新用户水平
        
        Args:
            user_id: 用户ID
        """
        # 计算所有技能的平均分
        skills = self.user_data[user_id]["skills"]
        avg_score = sum(skill["score"] for skill in skills.values()) / len(skills)
        
        # 根据平均分确定水平
        if avg_score < 0.4:
            level = "beginner"
        elif avg_score < 0.6:
            level = "intermediate"
        elif avg_score < 0.8:
            level = "advanced"
        else:
            level = "expert"
        
        # 更新用户水平
        self.user_data[user_id]["current_level"] = level
    
    def _generate_recommendations(self, user_id: str) -> None:
        """
        生成学习建议
        
        Args:
            user_id: 用户ID
        """
        recommendations = []
        skills = self.user_data[user_id]["skills"]
        
        # 找出最弱的技能领域
        weakest_skill = min(skills.items(), key=lambda x: x[1]["score"])
        
        # 根据最弱技能生成建议
        skill_recommendations = {
            "vocabulary": [
                "每天学习10个新单词，使用词汇记忆软件",
                "阅读英文新闻增加词汇量",
                "使用单词卡片记忆高频词汇"
            ],
            "grammar": [
                "学习高级语法结构和用法",
                "练习不同时态的使用",
                "使用语法检查工具改进句子结构"
            ],
            "fluency": [
                "每天进行英语对话练习",
                "听英语播客提高语感",
                "朗读英文文章提高流利度"
            ],
            "pronunciation": [
                "使用语音识别软件练习发音",
                "模仿英语母语者的发音和语调",
                "录制自己的英语发音并分析"
            ],
            "comprehension": [
                "观看英语电影提高听力理解",
                "参加英语阅读俱乐部",
                "尝试不同口音的英语听力材料"
            ]
        }
        
        # 添加针对最弱技能的建议
        for rec in skill_recommendations[weakest_skill[0]]:
            recommendations.append({
                "type": "skill_improvement",
                "target_skill": weakest_skill[0],
                "content": rec,
                "created_at": datetime.datetime.now().isoformat()
            })
        
        # 添加基于当前水平的建议
        level_recommendations = {
            "beginner": [
                "使用初级英语学习应用如Duolingo",
                "学习基础日常对话表达",
                "观看带字幕的简单英语节目"
            ],
            "intermediate": [
                "尝试不看字幕观看英语节目",
                "参加英语角练习口语",
                "使用英语写日记练习表达"
            ],
            "advanced": [
                "阅读英语原版书籍",
                "观看专业领域的英语讲座",
                "使用高级词汇和表达方式"
            ],
            "expert": [
                "尝试接触不同地区的英语口音",
                "阅读学术论文拓展专业词汇",
                "参与英语辩论提升思维能力"
            ]
        }
        
        # 添加基于当前水平的建议
        level = self.user_data[user_id]["current_level"]
        for rec in level_recommendations[level]:
            recommendations.append({
                "type": "level_based",
                "target_level": level,
                "content": rec,
                "created_at": datetime.datetime.now().isoformat()
            })
        
        # 保留最新的10条建议
        self.user_data[user_id]["recommendations"] = recommendations[:10]
    
    def get_learning_path(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户学习路径和建议
        
        Args:
            user_id: 用户ID
            
        Returns:
            学习路径和建议
        """
        if user_id not in self.user_data:
            self.user_data[user_id] = self._create_default_user_data()
            self._save_user_data(user_id)
        
        user_data = self.user_data[user_id]
        
        # 获取当前水平
        current_level = user_data["current_level"]
        
        # 获取技能评分
        skills = user_data["skills"]
        skill_scores = {k: v["score"] for k, v in skills.items()}
        
        # 获取建议
        recommendations = user_data["recommendations"]
        
        # 构建适合当前水平的场景列表
        scene_recommendations = []
        
        # 初学者场景
        beginner_scenes = ["简单日常对话", "自我介绍", "数字和时间", "简单购物对话"]
        
        # 中级场景
        intermediate_scenes = ["旅游问路", "餐厅点餐", "机场对话", "酒店预订"]
        
        # 高级场景
        advanced_scenes = ["商务会议", "学术讨论", "医疗咨询", "求职面试"]
        
        # 专家级场景
        expert_scenes = ["商业谈判", "学术演讲", "专业领域讨论", "法律咨询"]
        
        # 根据用户水平推荐场景
        if current_level == "beginner":
            scene_recommendations = beginner_scenes
        elif current_level == "intermediate":
            scene_recommendations = intermediate_scenes + beginner_scenes[:2]
        elif current_level == "advanced":
            scene_recommendations = advanced_scenes + intermediate_scenes[:2]
        else:  # expert
            scene_recommendations = expert_scenes + advanced_scenes[:2]
        
        # 构建学习路径
        learning_path = {
            "current_level": current_level,
            "skill_scores": skill_scores,
            "recommendations": recommendations,
            "recommended_scenes": scene_recommendations,
            "next_level": self._get_next_level(current_level),
            "progress_percentage": self._calculate_level_progress(user_id)
        }
        
        return learning_path
    
    def _get_next_level(self, current_level: str) -> str:
        """
        获取下一级别
        
        Args:
            current_level: 当前级别
            
        Returns:
            下一级别
        """
        levels = self.difficulty_levels
        current_index = levels.index(current_level) if current_level in levels else 0
        next_index = min(current_index + 1, len(levels) - 1)
        return levels[next_index]
    
    def _calculate_level_progress(self, user_id: str) -> float:
        """
        计算当前级别的完成进度百分比
        
        Args:
            user_id: 用户ID
            
        Returns:
            完成进度百分比
        """
        user_data = self.user_data[user_id]
        skills = user_data["skills"]
        avg_score = sum(skill["score"] for skill in skills.values()) / len(skills)
        
        # 级别分界点
        level_thresholds = {
            "beginner": 0.0,
            "intermediate": 0.4,
            "advanced": 0.6,
            "expert": 0.8
        }
        
        current_level = user_data["current_level"]
        current_threshold = level_thresholds[current_level]
        next_level = self._get_next_level(current_level)
        next_threshold = level_thresholds[next_level]
        
        # 如果已经是最高级别
        if current_level == next_level:
            return min(100.0, (avg_score - current_threshold) / (1.0 - current_threshold) * 100)
        
        # 计算进度
        progress = (avg_score - current_threshold) / (next_threshold - current_threshold)
        
        # 确保在0-100之间
        return min(100.0, max(0.0, progress * 100)) 