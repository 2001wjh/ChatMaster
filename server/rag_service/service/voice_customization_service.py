"""
语音定制服务
负责处理语音定制功能，支持选择不同口音和语速的语音合成
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class VoiceCustomizationService:
    """语音定制服务类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化语音定制服务
        
        Args:
            config: 配置参数
        """
        self.config = config
        
        # 初始化TTS配置
        self.tts_config = config.get("tts_config", {})
        
        # 支持的语音类型
        self.available_voices = [
            {
                "id": "en-US-female-1",
                "name": "美式英语 (女声 1)",
                "language": "en-US",
                "gender": "female",
                "accent": "american",
                "style": "neutral"
            },
            {
                "id": "en-US-female-2",
                "name": "美式英语 (女声 2)",
                "language": "en-US",
                "gender": "female",
                "accent": "american",
                "style": "friendly"
            },
            {
                "id": "en-US-male-1",
                "name": "美式英语 (男声 1)",
                "language": "en-US",
                "gender": "male",
                "accent": "american",
                "style": "neutral"
            },
            {
                "id": "en-US-male-2",
                "name": "美式英语 (男声 2)",
                "language": "en-US",
                "gender": "male",
                "accent": "american",
                "style": "formal"
            },
            {
                "id": "en-GB-female-1",
                "name": "英式英语 (女声 1)",
                "language": "en-GB",
                "gender": "female",
                "accent": "british",
                "style": "neutral"
            },
            {
                "id": "en-GB-male-1",
                "name": "英式英语 (男声 1)",
                "language": "en-GB",
                "gender": "male",
                "accent": "british",
                "style": "neutral"
            },
            {
                "id": "en-AU-female-1",
                "name": "澳式英语 (女声 1)",
                "language": "en-AU",
                "gender": "female",
                "accent": "australian",
                "style": "neutral"
            },
            {
                "id": "en-AU-male-1",
                "name": "澳式英语 (男声 1)",
                "language": "en-AU",
                "gender": "male",
                "accent": "australian",
                "style": "neutral"
            },
            {
                "id": "en-IN-female-1",
                "name": "印度英语 (女声 1)",
                "language": "en-IN",
                "gender": "female",
                "accent": "indian",
                "style": "neutral"
            },
            {
                "id": "en-IN-male-1",
                "name": "印度英语 (男声 1)",
                "language": "en-IN",
                "gender": "male",
                "accent": "indian",
                "style": "neutral"
            }
        ]
        
        # 支持的语速
        self.available_speeds = [
            {"id": "x-slow", "name": "极慢", "rate": 0.6},
            {"id": "slow", "name": "慢速", "rate": 0.8},
            {"id": "normal", "name": "正常", "rate": 1.0},
            {"id": "fast", "name": "快速", "rate": 1.2},
            {"id": "x-fast", "name": "极快", "rate": 1.5}
        ]
        
        # 默认配置
        self.default_voice = "en-US-female-1"
        self.default_speed = "normal"
        self.default_pitch = 1.0
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        获取可用的语音列表
        
        Returns:
            语音列表
        """
        return self.available_voices
    
    def get_available_speeds(self) -> List[Dict[str, Any]]:
        """
        获取可用的语速列表
        
        Returns:
            语速列表
        """
        return self.available_speeds
    
    def get_voice_by_id(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取语音配置
        
        Args:
            voice_id: 语音ID
            
        Returns:
            语音配置
        """
        for voice in self.available_voices:
            if voice["id"] == voice_id:
                return voice
        return None
    
    def get_speed_by_id(self, speed_id: str) -> Optional[Dict[str, Any]]:
        """
        根据ID获取语速配置
        
        Args:
            speed_id: 语速ID
            
        Returns:
            语速配置
        """
        for speed in self.available_speeds:
            if speed["id"] == speed_id:
                return speed
        return None
    
    def create_tts_config(self, 
                        voice_id: Optional[str] = None, 
                        speed_id: Optional[str] = None,
                        pitch: Optional[float] = None) -> Dict[str, Any]:
        """
        创建TTS配置
        
        Args:
            voice_id: 语音ID
            speed_id: 语速ID
            pitch: 音高
            
        Returns:
            TTS配置
        """
        # 使用默认值
        selected_voice_id = voice_id or self.default_voice
        selected_speed_id = speed_id or self.default_speed
        selected_pitch = pitch or self.default_pitch
        
        # 获取语音和语速配置
        voice_config = self.get_voice_by_id(selected_voice_id) or self.get_voice_by_id(self.default_voice)
        speed_config = self.get_speed_by_id(selected_speed_id) or self.get_speed_by_id(self.default_speed)
        
        # 创建配置
        tts_config = {
            "voice_id": voice_config["id"],
            "language": voice_config["language"],
            "gender": voice_config["gender"],
            "accent": voice_config["accent"],
            "style": voice_config["style"],
            "speed": speed_config["rate"],
            "pitch": selected_pitch
        }
        
        return tts_config
    
    def get_voice_by_criteria(self, 
                            gender: Optional[str] = None, 
                            accent: Optional[str] = None,
                            style: Optional[str] = None) -> Dict[str, Any]:
        """
        根据条件筛选语音
        
        Args:
            gender: 性别 (male/female)
            accent: 口音 (american/british/australian/indian)
            style: 风格 (neutral/friendly/formal)
            
        Returns:
            最佳匹配的语音配置
        """
        # 为每个语音计算匹配度
        best_match = None
        best_score = -1
        
        for voice in self.available_voices:
            score = 0
            
            # 检查性别
            if gender and voice["gender"] == gender:
                score += 1
                
            # 检查口音
            if accent and voice["accent"] == accent:
                score += 2
                
            # 检查风格
            if style and voice["style"] == style:
                score += 1
                
            # 更新最佳匹配
            if score > best_score:
                best_score = score
                best_match = voice
        
        # 如果没有匹配项，返回默认语音
        if best_match is None:
            return self.get_voice_by_id(self.default_voice)
            
        return best_match
    
    def apply_voice_effects(self, tts_config: Dict[str, Any], 
                          effects: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用语音效果
        
        Args:
            tts_config: TTS配置
            effects: 效果参数
                {
                    "emphasis": 浮点数 (0.0-2.0, 强调程度),
                    "clarity": 浮点数 (0.0-2.0, 清晰度),
                    "breathiness": 浮点数 (0.0-1.0, 气息感)
                }
                
        Returns:
            修改后的TTS配置
        """
        # 复制配置
        modified_config = tts_config.copy()
        
        # 添加效果参数
        if effects:
            modified_config["effects"] = effects
            
            # 适当调整速度和音高
            if "emphasis" in effects and effects["emphasis"] > 1.0:
                # 增加强调时稍微降低语速
                modified_config["speed"] = modified_config.get("speed", 1.0) * 0.9
                
            if "clarity" in effects and effects["clarity"] > 1.0:
                # 增加清晰度时稍微降低语速
                modified_config["speed"] = modified_config.get("speed", 1.0) * 0.95
                
        return modified_config
    
    def get_user_voice_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户语音偏好
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户语音偏好
        """
        # 在实际应用中，这里应该从数据库中加载用户偏好
        # 这里返回一个默认配置
        return {
            "voice_id": self.default_voice,
            "speed_id": self.default_speed,
            "pitch": self.default_pitch,
            "effects": {}
        }
    
    def save_user_voice_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """
        保存用户语音偏好
        
        Args:
            user_id: 用户ID
            preferences: 语音偏好
            
        Returns:
            是否成功
        """
        # 在实际应用中，这里应该将用户偏好保存到数据库
        logger.info(f"保存用户 {user_id} 的语音偏好: {preferences}")
        return True 