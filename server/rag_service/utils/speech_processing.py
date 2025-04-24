"""
语音处理模块
集成语音识别(ASR)和语音合成(TTS)功能
"""

import os
import json
import logging
import tempfile
import subprocess
import base64
from typing import Dict, Any, List, Optional, Tuple, Union
import requests

logger = logging.getLogger(__name__)

class SpeechProcessingService:
    """语音处理服务类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化语音处理服务
        
        Args:
            config: 配置参数
        """
        self.config = config
        
        # ASR配置
        self.asr_service_url = config.get("asr_service_url", "http://localhost:10095")
        self.asr_timeout = config.get("asr_timeout", 30)
        self.enable_advanced_asr = config.get("enable_advanced_asr", True)
        
        # TTS配置
        self.tts_service_url = config.get("tts_service_url", "http://localhost:6006")
        self.tts_timeout = config.get("tts_timeout", 30)
        self.enable_advanced_tts = config.get("enable_advanced_tts", True)
        
        # 语音合成默认配置
        self.default_voice = config.get("default_voice", "en_female_1")
        self.default_speed = config.get("default_speed", 1.0)
        self.default_pitch = config.get("default_pitch", 1.0)
        
        # 音频格式
        self.audio_format = config.get("audio_format", "wav")
        self.sample_rate = config.get("sample_rate", 16000)
        
        # 临时文件目录
        self.temp_dir = config.get("temp_dir", tempfile.gettempdir())
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info("初始化语音处理服务")
    
    def speech_to_text(self, audio_file_path: str) -> Dict[str, Any]:
        """
        语音识别(ASR)
        
        Args:
            audio_file_path: 音频文件路径
            
        Returns:
            识别结果，包含文本、置信度等信息
        """
        try:
            logger.info(f"开始语音识别: {audio_file_path}")
            
            if not os.path.exists(audio_file_path):
                raise ValueError(f"音频文件不存在: {audio_file_path}")
            
            # 检查和准备音频文件
            prepared_audio = self._prepare_audio_for_asr(audio_file_path)
            
            # 调用FunASR服务
            response = self._call_asr_service(prepared_audio)
            
            logger.info(f"语音识别完成")
            return response
        
        except Exception as e:
            logger.error(f"语音识别失败: {str(e)}")
            return {
                "status": "error",
                "text": "",
                "error_message": str(e)
            }
    
    def _prepare_audio_for_asr(self, audio_file_path: str) -> str:
        """
        准备音频文件用于ASR
        
        Args:
            audio_file_path: 原始音频文件路径
            
        Returns:
            处理后的音频文件路径
        """
        # 获取文件扩展名
        file_ext = os.path.splitext(audio_file_path)[1].lower()
        
        # 如果不是WAV格式，先转换
        if file_ext != ".wav":
            output_path = os.path.join(self.temp_dir, f"temp_{os.path.basename(audio_file_path)}.wav")
            
            # 使用ffmpeg转换音频格式
            try:
                cmd = [
                    "ffmpeg", "-y", "-i", audio_file_path, 
                    "-ar", str(self.sample_rate), 
                    "-ac", "1", 
                    output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                return output_path
            except subprocess.CalledProcessError as e:
                logger.error(f"音频转换失败: {e.stderr.decode('utf-8')}")
                raise RuntimeError(f"音频转换失败: {e.stderr.decode('utf-8')}")
        
        return audio_file_path
    
    def _call_asr_service(self, audio_file_path: str) -> Dict[str, Any]:
        """
        调用ASR服务
        
        Args:
            audio_file_path: 处理后的音频文件路径
            
        Returns:
            识别结果
        """
        # 读取音频文件
        with open(audio_file_path, 'rb') as f:
            audio_content = f.read()
        
        # 调用FunASR HTTP API
        try:
            url = f"{self.asr_service_url}/recognize"
            
            files = {
                'audio': (os.path.basename(audio_file_path), audio_content),
            }
            
            params = {
                'sample_rate': self.sample_rate,
                'language': 'en',  # 默认英语识别
                'enable_punctuation': 'true' if self.enable_advanced_asr else 'false',
                'enable_timestamp': 'true' if self.enable_advanced_asr else 'false'
            }
            
            response = requests.post(url, files=files, params=params, timeout=self.asr_timeout)
            response.raise_for_status()
            
            result = response.json()
            
            # 返回统一格式的结果
            return {
                "status": "success",
                "text": result.get("text", ""),
                "confidence": result.get("confidence", 0.0),
                "timestamps": result.get("timestamps", []),
                "words": result.get("words", [])
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"调用ASR服务失败: {str(e)}")
            raise RuntimeError(f"调用ASR服务失败: {str(e)}")
    
    def text_to_speech(self, text: str, voice_id: str = None, 
                      speed: float = None, pitch: float = None) -> Dict[str, Any]:
        """
        语音合成(TTS)
        
        Args:
            text: 要合成的文本
            voice_id: 声音ID
            speed: 语速（0.5-2.0）
            pitch: 音调（0.5-2.0）
            
        Returns:
            合成结果，包含音频数据、格式等信息
        """
        try:
            logger.info(f"开始语音合成")
            
            # 使用默认值
            voice_id = voice_id or self.default_voice
            speed = speed or self.default_speed
            pitch = pitch or self.default_pitch
            
            # 调用ChatTTS服务
            response = self._call_tts_service(text, voice_id, speed, pitch)
            
            logger.info(f"语音合成完成")
            return response
            
        except Exception as e:
            logger.error(f"语音合成失败: {str(e)}")
            return {
                "status": "error",
                "audio_data": None,
                "error_message": str(e)
            }
    
    def _call_tts_service(self, text: str, voice_id: str, 
                         speed: float, pitch: float) -> Dict[str, Any]:
        """
        调用TTS服务
        
        Args:
            text: 要合成的文本
            voice_id: 声音ID
            speed: 语速
            pitch: 音调
            
        Returns:
            合成结果
        """
        try:
            url = f"{self.tts_service_url}/synthesize"
            
            data = {
                "text": text,
                "voice_id": voice_id,
                "speed": speed,
                "pitch": pitch,
                "format": self.audio_format,
                "sample_rate": self.sample_rate
            }
            
            response = requests.post(url, json=data, timeout=self.tts_timeout)
            response.raise_for_status()
            
            result = response.json()
            
            # 返回统一格式的结果
            return {
                "status": "success",
                "audio_data": result.get("audio_base64", ""),
                "format": self.audio_format,
                "sample_rate": self.sample_rate,
                "duration": result.get("duration", 0.0)
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"调用TTS服务失败: {str(e)}")
            raise RuntimeError(f"调用TTS服务失败: {str(e)}")
    
    def save_audio_to_file(self, audio_data: str, output_path: str) -> str:
        """
        将Base64编码的音频数据保存到文件
        
        Args:
            audio_data: Base64编码的音频数据
            output_path: 输出文件路径
            
        Returns:
            保存的文件路径
        """
        try:
            # 解码Base64
            audio_bytes = base64.b64decode(audio_data)
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # 写入文件
            with open(output_path, 'wb') as f:
                f.write(audio_bytes)
            
            return output_path
            
        except Exception as e:
            logger.error(f"保存音频文件失败: {str(e)}")
            raise RuntimeError(f"保存音频文件失败: {str(e)}")
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        获取可用的语音列表
        
        Returns:
            语音列表，包含ID、名称、性别、语言等信息
        """
        try:
            url = f"{self.tts_service_url}/voices"
            
            response = requests.get(url, timeout=self.tts_timeout)
            response.raise_for_status()
            
            result = response.json()
            return result.get("voices", [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"获取语音列表失败: {str(e)}")
            return [
                {
                    "id": "en_female_1",
                    "name": "English Female 1",
                    "gender": "female",
                    "language": "en-US"
                },
                {
                    "id": "en_male_1",
                    "name": "English Male 1",
                    "gender": "male",
                    "language": "en-US"
                }
            ]  # 返回默认值 