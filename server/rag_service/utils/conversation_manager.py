"""
对话管理模块
实现拆分式对话管理、对话状态跟踪和外部记忆机制
"""

import re
import json
import logging
import datetime
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class DialoguePolicy:
    """对话策略类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化对话策略
        
        Args:
            config: 配置参数
        """
        self.config = config
        
        # 对话阶段
        self.dialogue_stages = [
            "greeting",           # 问候阶段
            "topic_exploration",  # 话题探索
            "deeper_discussion",  # 深入讨论
            "clarification",      # 问题澄清
            "summary",            # 总结
            "closing"             # 结束
        ]
        
        # 策略规则
        self.policy_rules = {
            "greeting": {
                "next_stages": ["topic_exploration"],
                "max_turns": 2,
                "required_info": []
            },
            "topic_exploration": {
                "next_stages": ["deeper_discussion", "clarification"],
                "max_turns": 5,
                "required_info": ["topic", "user_interest"]
            },
            "deeper_discussion": {
                "next_stages": ["clarification", "summary"],
                "max_turns": 10,
                "required_info": ["topic", "user_knowledge"]
            },
            "clarification": {
                "next_stages": ["deeper_discussion", "summary"],
                "max_turns": 3,
                "required_info": ["unclear_points"]
            },
            "summary": {
                "next_stages": ["closing", "topic_exploration"],
                "max_turns": 2,
                "required_info": ["key_points"]
            },
            "closing": {
                "next_stages": ["greeting"],
                "max_turns": 2,
                "required_info": []
            }
        }
        
        # 各阶段的回复模板
        self.response_templates = {
            "greeting": [
                "你好！今天想聊些什么？",
                "很高兴见到你！有什么我可以帮助你的吗？",
                "嗨！准备好开始我们的英语练习了吗？"
            ],
            "topic_exploration": [
                "这个话题很有趣，你能告诉我更多关于{topic}的事情吗？",
                "我想更多地了解你对{topic}的看法。",
                "让我们深入探讨{topic}，你有什么具体的问题吗？"
            ],
            "deeper_discussion": [
                "关于{topic}，一个重要的方面是{aspect}。你怎么看？",
                "在{topic}中，{key_point}是一个关键点。这与你的经验一致吗？",
                "我们可以从{perspective}角度来看待{topic}。这给你带来了新的思考吗？"
            ],
            "clarification": [
                "我想确认一下，你是说{unclear_point}吗？",
                "让我们澄清一下：{clarification_question}",
                "我不太确定你的意思，你能解释一下{unclear_point}吗？"
            ],
            "summary": [
                "总结一下我们的讨论：{summary_points}",
                "今天我们讨论了以下几点：{summary_points}",
                "我们探讨了关于{topic}的几个方面：{summary_points}"
            ],
            "closing": [
                "很高兴与你交流！有什么其他问题吗？",
                "我们的讨论很有成效。下次聊天再见！",
                "感谢你的参与！希望这次对话对你有帮助。"
            ]
        }
    
    def determine_next_action(self, dialogue_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据当前对话状态确定下一步行动
        
        Args:
            dialogue_state: 当前对话状态
            
        Returns:
            下一步行动的决策
        """
        current_stage = dialogue_state.get("current_stage", "greeting")
        turns_in_stage = dialogue_state.get("turns_in_stage", 0)
        available_stages = self.policy_rules[current_stage]["next_stages"]
        
        # 检查是否需要转换阶段
        stage_max_turns = self.policy_rules[current_stage]["max_turns"]
        
        decision = {
            "should_change_stage": False,
            "next_stage": current_stage,
            "response_type": "normal"
        }
        
        # 如果当前阶段已达到最大轮次，则考虑转换阶段
        if turns_in_stage >= stage_max_turns:
            decision["should_change_stage"] = True
            decision["next_stage"] = available_stages[0]  # 默认取第一个下一阶段
        
        # 特殊情况处理
        user_query = dialogue_state.get("last_query", "")
        
        # 检测问题或请求澄清
        if "?" in user_query and current_stage != "clarification":
            decision["should_change_stage"] = True
            decision["next_stage"] = "clarification"
            decision["response_type"] = "question_answering"
        
        # 检测总结请求
        if any(kw in user_query.lower() for kw in ["summarize", "summary", "conclude", "总结", "小结"]):
            decision["should_change_stage"] = True
            decision["next_stage"] = "summary"
            decision["response_type"] = "summarization"
        
        # 检测结束信号
        if any(kw in user_query.lower() for kw in ["goodbye", "bye", "thanks", "thank you", "再见", "谢谢"]):
            decision["should_change_stage"] = True
            decision["next_stage"] = "closing"
            decision["response_type"] = "farewell"
        
        return decision
    
    def get_response_template(self, stage: str) -> str:
        """
        获取指定阶段的回复模板
        
        Args:
            stage: 对话阶段
            
        Returns:
            回复模板
        """
        templates = self.response_templates.get(stage, self.response_templates["greeting"])
        return np.random.choice(templates)


class DialogueStateTracker:
    """对话状态跟踪器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化状态跟踪器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.dialogue_history_max_turns = config.get("dialogue_history_max_turns", 10)
        self.enable_topic_tracking = config.get("enable_topic_tracking", True)
        
        # 实体和槽位类型
        self.slot_types = {
            "topic": None,          # 当前讨论的主题
            "user_interest": None,  # 用户感兴趣的内容
            "user_knowledge": None, # 用户知识水平
            "unclear_points": [],   # 需要澄清的点
            "key_points": []        # 讨论的关键点
        }
    
    def initialize_state(self, conversation_id: str) -> Dict[str, Any]:
        """
        初始化对话状态
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            初始化的对话状态
        """
        return {
            "conversation_id": conversation_id,
            "current_stage": "greeting",
            "turns_in_stage": 0,
            "total_turns": 0,
            "dialogue_history": [],
            "slots": dict(self.slot_types),
            "last_query": "",
            "last_response": "",
            "topics": [],
            "entities": [],
            "last_updated": datetime.datetime.now().isoformat()
        }
    
    def update_state(self, state: Dict[str, Any], user_query: str, 
                    system_response: str, extracted_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新对话状态
        
        Args:
            state: 当前对话状态
            user_query: 用户查询
            system_response: 系统回复
            extracted_info: 从对话中提取的信息
            
        Returns:
            更新后的对话状态
        """
        # 更新对话轮次
        state["turns_in_stage"] += 1
        state["total_turns"] += 1
        
        # 更新对话历史
        state["dialogue_history"].append({
            "turn": state["total_turns"],
            "user": user_query,
            "system": system_response,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # 限制对话历史长度
        if len(state["dialogue_history"]) > self.dialogue_history_max_turns:
            state["dialogue_history"] = state["dialogue_history"][-self.dialogue_history_max_turns:]
        
        # 更新最后查询和回复
        state["last_query"] = user_query
        state["last_response"] = system_response
        
        # 更新槽位信息
        if "slots" in extracted_info:
            for slot, value in extracted_info["slots"].items():
                if slot in state["slots"]:
                    state["slots"][slot] = value
        
        # 更新实体
        if "entities" in extracted_info:
            state["entities"].extend(extracted_info["entities"])
        
        # 更新主题
        if self.enable_topic_tracking and "topics" in extracted_info:
            for topic in extracted_info["topics"]:
                if topic not in state["topics"]:
                    state["topics"].append(topic)
        
        # 更新时间戳
        state["last_updated"] = datetime.datetime.now().isoformat()
        
        return state
    
    def extract_information(self, text: str) -> Dict[str, Any]:
        """
        从文本中提取信息
        
        Args:
            text: 输入文本
            
        Returns:
            提取的信息
        """
        result = {
            "slots": {},
            "entities": [],
            "topics": []
        }
        
        # 简单实现：基于规则提取信息
        
        # 1. 提取主题（简化为句子中最重要的名词短语）
        # 在实际实现中，可以使用更复杂的NLP技术
        topic_candidates = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if topic_candidates:
            result["topics"] = topic_candidates[:2]  # 取前两个作为主题候选
            result["slots"]["topic"] = topic_candidates[0]
        
        # 2. 检测用户兴趣
        interest_patterns = [
            r'I am interested in (.+?)\.', 
            r'I like (.+?)\.', 
            r'I enjoy (.+?)\.', 
            r'I want to learn about (.+?)\.', 
            r'I\'m curious about (.+?)\.'
        ]
        
        for pattern in interest_patterns:
            matches = re.findall(pattern, text)
            if matches:
                result["slots"]["user_interest"] = matches[0]
                break
        
        # 3. 检测不清楚的点（问题）
        if '?' in text:
            unclear_points = re.findall(r'(.+?\?)', text)
            if unclear_points:
                result["slots"]["unclear_points"] = unclear_points
        
        return result


class ExternalMemory:
    """外部记忆机制"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化外部记忆
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.max_memory_items = config.get("max_memory_items", 100)
        self.memory_decay_factor = config.get("memory_decay_factor", 0.95)  # 记忆衰减因子
        self.max_context_window = config.get("max_context_window", 3000)    # 上下文窗口最大长度
        
        # 记忆存储
        self.semantic_memory = {}  # 语义记忆：长期知识
        self.episodic_memory = {}  # 情景记忆：具体对话内容
        self.working_memory = {}   # 工作记忆：当前对话的活跃信息
    
    def add_episodic_memory(self, conversation_id: str, memory_item: Dict[str, Any]) -> None:
        """
        添加情景记忆
        
        Args:
            conversation_id: 对话ID
            memory_item: 记忆项，包含对话内容、时间戳等
        """
        if conversation_id not in self.episodic_memory:
            self.episodic_memory[conversation_id] = []
        
        # 添加记忆项
        memory_item["created_at"] = datetime.datetime.now().isoformat()
        memory_item["importance"] = memory_item.get("importance", 0.5)  # 默认重要性
        memory_item["recall_count"] = 0
        
        self.episodic_memory[conversation_id].append(memory_item)
        
        # 限制记忆大小
        if len(self.episodic_memory[conversation_id]) > self.max_memory_items:
            # 按重要性排序，保留重要的记忆
            self.episodic_memory[conversation_id] = sorted(
                self.episodic_memory[conversation_id], 
                key=lambda x: x["importance"], 
                reverse=True
            )[:self.max_memory_items]
    
    def update_working_memory(self, conversation_id: str, information: Dict[str, Any]) -> None:
        """
        更新工作记忆
        
        Args:
            conversation_id: 对话ID
            information: 当前对话的活跃信息
        """
        if conversation_id not in self.working_memory:
            self.working_memory[conversation_id] = {}
        
        # 更新工作记忆
        for key, value in information.items():
            self.working_memory[conversation_id][key] = value
    
    def retrieve_relevant_memories(self, conversation_id: str, query: str, 
                                 max_items: int = 5) -> List[Dict[str, Any]]:
        """
        检索相关记忆
        
        Args:
            conversation_id: 对话ID
            query: 查询文本
            max_items: 最大返回记忆项数量
            
        Returns:
            相关的记忆项列表
        """
        if conversation_id not in self.episodic_memory:
            return []
        
        memories = self.episodic_memory[conversation_id]
        
        # 简单实现：基于关键词匹配检索相关记忆
        # 在实际实现中可以使用向量相似度等复杂方法
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        scored_memories = []
        
        for memory in memories:
            if "content" in memory:
                memory_words = set(re.findall(r'\b\w+\b', memory["content"].lower()))
                # 计算相关性分数
                common_words = query_words.intersection(memory_words)
                if common_words:
                    score = len(common_words) / len(query_words)
                    # 考虑记忆的重要性和新近性
                    created_at = datetime.datetime.fromisoformat(memory["created_at"])
                    days_old = (datetime.datetime.now() - created_at).days + 1
                    recency_factor = 1.0 / (days_old ** 0.5)  # 平方根衰减
                    
                    final_score = score * memory["importance"] * recency_factor
                    scored_memories.append((memory, final_score))
        
        # 按分数排序并返回前N个
        relevant_memories = [m for m, s in sorted(scored_memories, key=lambda x: x[1], reverse=True)[:max_items]]
        
        # 更新检索计数
        for memory in relevant_memories:
            memory["recall_count"] += 1
            # 被频繁检索的记忆增加重要性
            memory["importance"] = min(1.0, memory["importance"] + 0.05)
        
        return relevant_memories
    
    def get_compressed_context(self, conversation_id: str, 
                             max_tokens: int = None) -> str:
        """
        获取压缩的上下文
        
        Args:
            conversation_id: 对话ID
            max_tokens: 最大上下文长度（字符数），默认使用配置的值
            
        Returns:
            压缩的上下文文本
        """
        if max_tokens is None:
            max_tokens = self.max_context_window
        
        if conversation_id not in self.episodic_memory:
            return ""
        
        memories = self.episodic_memory[conversation_id]
        
        # 按时间排序
        sorted_memories = sorted(memories, key=lambda x: x["created_at"])
        
        # 提取关键记忆
        key_memories = [m for m in sorted_memories if m["importance"] > 0.7]
        
        # 提取最近记忆
        recent_memories = sorted_memories[-10:]
        
        # 合并并去重
        all_memories = []
        memory_ids = set()
        
        for memory in key_memories + recent_memories:
            if "id" in memory and memory["id"] not in memory_ids:
                all_memories.append(memory)
                memory_ids.add(memory["id"])
        
        # 构建上下文文本
        context_parts = []
        
        # 添加工作记忆（当前话题等）
        if conversation_id in self.working_memory:
            working_mem = self.working_memory[conversation_id]
            if "topic" in working_mem and working_mem["topic"]:
                context_parts.append(f"Current topic: {working_mem['topic']}")
            if "user_interest" in working_mem and working_mem["user_interest"]:
                context_parts.append(f"User is interested in: {working_mem['user_interest']}")
        
        # 添加记忆内容
        for memory in all_memories:
            if "content" in memory:
                context_parts.append(memory["content"])
        
        # 拼接上下文并限制长度
        context = "\n".join(context_parts)
        if len(context) > max_tokens:
            # 简单截断
            context = context[:max_tokens] + "..."
        
        return context
    
    def forget_memories(self, conversation_id: str, days_threshold: int = 30) -> None:
        """
        遗忘旧记忆
        
        Args:
            conversation_id: 对话ID
            days_threshold: 超过多少天的记忆会被遗忘
        """
        if conversation_id not in self.episodic_memory:
            return
        
        current_time = datetime.datetime.now()
        updated_memories = []
        
        for memory in self.episodic_memory[conversation_id]:
            created_at = datetime.datetime.fromisoformat(memory["created_at"])
            days_old = (current_time - created_at).days
            
            # 应用遗忘机制
            if days_old <= days_threshold:
                # 按照记忆衰减因子降低重要性
                memory["importance"] *= (self.memory_decay_factor ** days_old)
                
                # 只保留重要性仍然足够高的记忆
                if memory["importance"] > 0.1:
                    updated_memories.append(memory)
            # 非常重要的记忆即使超过时间也会保留
            elif memory["importance"] > 0.8:
                memory["importance"] *= 0.5  # 大幅降低重要性，但仍然保留
                updated_memories.append(memory)
        
        self.episodic_memory[conversation_id] = updated_memories


class ConversationManager:
    """对话管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化对话管理器
        
        Args:
            config: 配置参数
        """
        self.config = config
        
        # 初始化组件
        self.dialogue_policy = DialoguePolicy(config)
        self.state_tracker = DialogueStateTracker(config)
        self.external_memory = ExternalMemory(config)
        
        # 对话状态存储
        self.dialogue_states = {}
    
    def initialize_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        初始化新对话
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            对话状态
        """
        if conversation_id not in self.dialogue_states:
            state = self.state_tracker.initialize_state(conversation_id)
            self.dialogue_states[conversation_id] = state
        
        return self.dialogue_states[conversation_id]
    
    def process_user_input(self, conversation_id: str, user_query: str, 
                         system_response: str) -> Dict[str, Any]:
        """
        处理用户输入
        
        Args:
            conversation_id: 对话ID
            user_query: 用户查询
            system_response: 系统回复
            
        Returns:
            对话管理结果，包含对话状态和策略决策
        """
        # 获取或初始化对话状态
        state = self.dialogue_states.get(conversation_id)
        if state is None:
            state = self.initialize_conversation(conversation_id)
        
        # 从文本中提取信息
        extracted_info = self.state_tracker.extract_information(user_query)
        
        # 更新对话状态
        state = self.state_tracker.update_state(state, user_query, system_response, extracted_info)
        
        # 决定下一步行动
        policy_decision = self.dialogue_policy.determine_next_action(state)
        
        # 如果需要转换阶段
        if policy_decision["should_change_stage"]:
            state["current_stage"] = policy_decision["next_stage"]
            state["turns_in_stage"] = 0
        
        # 更新状态
        self.dialogue_states[conversation_id] = state
        
        # 更新外部记忆
        memory_item = {
            "id": f"{conversation_id}-{state['total_turns']}",
            "type": "dialogue_turn",
            "content": f"User: {user_query}\nSystem: {system_response}",
            "importance": 0.5,  # 默认重要性
            "metadata": {
                "turn": state["total_turns"],
                "stage": state["current_stage"]
            }
        }
        
        # 检测重要对话轮次
        if policy_decision["response_type"] in ["question_answering", "summarization"]:
            memory_item["importance"] = 0.8  # 提高重要对话的重要性
        
        self.external_memory.add_episodic_memory(conversation_id, memory_item)
        
        # 更新工作记忆
        self.external_memory.update_working_memory(conversation_id, {
            "topic": state["slots"]["topic"],
            "user_interest": state["slots"]["user_interest"],
            "current_stage": state["current_stage"]
        })
        
        return {
            "state": state,
            "policy_decision": policy_decision,
            "next_template": self.dialogue_policy.get_response_template(state["current_stage"])
        }
    
    def get_relevant_context(self, conversation_id: str, query: str, 
                           max_items: int = 5) -> List[Dict[str, Any]]:
        """
        获取相关上下文记忆
        
        Args:
            conversation_id: 对话ID
            query: 查询文本
            max_items: 最大返回记忆项数量
            
        Returns:
            相关的记忆项列表
        """
        return self.external_memory.retrieve_relevant_memories(conversation_id, query, max_items)
    
    def get_conversation_summary(self, conversation_id: str) -> str:
        """
        获取对话摘要
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            对话摘要
        """
        state = self.dialogue_states.get(conversation_id)
        if not state:
            return "No conversation found."
        
        # 简单实现：汇总主要对话内容
        history = state["dialogue_history"]
        if not history:
            return "No dialogue history available."
        
        topics = state["topics"]
        topic_str = ", ".join(topics) if topics else "No specific topics"
        
        turns = len(history)
        first_turn = history[0]["timestamp"] if history else "Unknown"
        last_turn = history[-1]["timestamp"] if history else "Unknown"
        
        summary = f"Conversation with {turns} turns, from {first_turn} to {last_turn}.\n"
        summary += f"Topics discussed: {topic_str}\n\n"
        
        # 添加关键对话轮次
        summary += "Key exchanges:\n"
        
        # 查找重要的对话轮次（前3轮和最近3轮）
        key_turns = []
        if turns > 0:
            key_turns.extend(history[:min(3, turns)])
        if turns > 6:
            key_turns.extend(history[-3:])
        
        for turn in key_turns:
            summary += f"Turn {turn['turn']}:\n"
            summary += f"User: {turn['user']}\n"
            summary += f"System: {turn['system']}\n\n"
        
        return summary
    
    def clear_conversation(self, conversation_id: str) -> None:
        """
        清除对话
        
        Args:
            conversation_id: 对话ID
        """
        if conversation_id in self.dialogue_states:
            del self.dialogue_states[conversation_id]
        
        # 遗忘记忆（不完全删除，应用遗忘机制）
        self.external_memory.forget_memories(conversation_id, days_threshold=0)
        
        # 清除工作记忆
        if conversation_id in self.external_memory.working_memory:
            del self.external_memory.working_memory[conversation_id] 