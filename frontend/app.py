"""
ChatMaster - 智能英语外教系统
集成文字对话、口语交流和RAG功能
"""

import os
import uuid
import json
import time
import tempfile
import datetime
import streamlit as st
import requests
from typing import Dict, Any, List, Optional

# 导入RAG组件
from rag_components import (
    render_knowledge_base_management, 
    render_document_management, 
    get_knowledge_bases,
    rag_query,
    display_rag_result,
    render_enhanced_rag_interface
)

# 导入性能仪表盘
from retrieval_dashboard import main as retrieval_dashboard_main

# 配置页面
st.set_page_config(
    page_title="ChatMaster - 英语外教系统",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4())
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "current_view" not in st.session_state:
    st.session_state.current_view = "chat"
if "selected_kb_id" not in st.session_state:
    st.session_state.selected_kb_id = None
if "selected_kb_name" not in st.session_state:
    st.session_state.selected_kb_name = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "submit_question" not in st.session_state:
    st.session_state.submit_question = False
if "interaction_mode" not in st.session_state:
    st.session_state.interaction_mode = "text"  # 'text' 或 'speech'

# API相关配置
API_BASE = "http://localhost:8000"  # 后端API基础URL
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# 保存和加载对话历史相关函数
def load_session_history():
    """加载会话历史"""
    try:
        with open("sessions.json", "r", encoding="utf-8") as f:
            sessions = json.load(f)
        return sessions
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_session_history():
    """保存会话历史"""
    with open("sessions.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.sessions, f, ensure_ascii=False, indent=2)

def generate_session_name(question):
    """根据问题生成会话名称"""
    if not question:
        return "新对话"
    # 简单截取前15个字符作为会话名称
    return question[:15] + "..." if len(question) > 15 else question

def save_current_session():
    """保存当前会话"""
    if not st.session_state.messages:
        return
    
    # 生成一个会话名称
    first_question = next((msg["content"] for msg in st.session_state.messages if msg["role"] == "user"), "新对话")
    session_name = generate_session_name(first_question)
    
    # 保存会话
    st.session_state.sessions[st.session_state.current_session_id] = {
        "name": session_name,
        "messages": st.session_state.messages,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 保存到文件
    save_session_history()

def load_session(session_id):
    """加载指定会话"""
    if session_id in st.session_state.sessions:
        st.session_state.messages = st.session_state.sessions[session_id]["messages"]
        st.session_state.current_session_id = session_id
        st.experimental_rerun()

def delete_session(session_id):
    """删除指定会话"""
    if session_id in st.session_state.sessions:
        del st.session_state.sessions[session_id]
        save_session_history()
        if session_id == st.session_state.current_session_id:
            st.session_state.messages = []
            st.session_state.current_session_id = str(uuid.uuid4())
        st.experimental_rerun()

def generate_recommended_questions(question, answer, context=None):
    """生成推荐问题"""
    # 这里可以使用更复杂的逻辑生成推荐问题
    # 简单实现：返回一些固定的问题
    return [
        "如何提高英语口语流利度？",
        "有哪些常用的英语日常对话表达？",
        "英语学习有什么好的方法推荐？"
    ]

# 渲染侧边栏
def render_sidebar():
    st.sidebar.title("ChatMaster 设置")
    
    # 视图切换
    st.sidebar.subheader("功能选择")
    view_options = {
        "chat": "对话练习",
        "enhanced_retrieval": "增强型检索",
        "performance_dashboard": "性能仪表盘",
        "knowledge_base_management": "知识库管理",
        "document_management": "文档管理 (需先选择知识库)"
    }
    
    # 如果没有选择知识库，则禁用文档管理选项
    disabled_options = ["document_management"] if not st.session_state.selected_kb_id else []
    
    # 创建一个列表用于显示
    view_options_list = list(view_options.keys())
    view_labels = list(view_options.values())
    
    # 设置默认索引
    default_idx = view_options_list.index(st.session_state.current_view) if st.session_state.current_view in view_options_list else 0
    
    selected_view = st.sidebar.radio(
        "选择功能:",
        options=view_options_list,
        format_func=lambda x: view_options[x] + (" (需先选择知识库)" if x in disabled_options else ""),
        index=default_idx,
        disabled=False
    )
    
    # 如果选择了文档管理但没有选择知识库，提示用户
    if selected_view == "document_management" and not st.session_state.selected_kb_id:
        st.sidebar.warning("请先在知识库管理中选择或创建一个知识库")
    else:
        st.session_state.current_view = selected_view
    
    # 对话模式
    if st.session_state.current_view == "chat":
        st.sidebar.subheader("交互设置")
        
        # 切换交互模式
        interaction_mode = st.sidebar.radio(
            "交互模式:",
            options=["text", "speech"],
            format_func=lambda x: "文字对话" if x == "text" else "语音交流",
            index=0 if st.session_state.interaction_mode == "text" else 1
        )
        st.session_state.interaction_mode = interaction_mode
        
        # 选择对话场景
        st.sidebar.subheader("对话场景")
        
        # 默认场景
        default_scenes = [
            "日常对话", "旅游出行", "商务会议", "学术讨论", 
            "求职面试", "医疗健康", "购物消费", "餐厅用餐",
            "酒店住宿", "交通出行"
        ]
        
        # 获取知识库作为自定义场景
        knowledge_bases = get_knowledge_bases()
        kb_options = [{"id": kb["id"], "name": kb["name"]} for kb in knowledge_bases]
        
        # 场景选择
        scene_type = st.sidebar.radio(
            "选择场景类型:",
            options=["default", "custom"],
            format_func=lambda x: "默认场景" if x == "default" else "知识库场景",
            index=0
        )
        
        if scene_type == "default":
            selected_scene = st.sidebar.selectbox(
                "选择默认场景:",
                options=default_scenes
            )
            # 清除已选知识库
            st.session_state.selected_kb_id = None
        else:
            if kb_options:
                kb_names = [kb["name"] for kb in kb_options]
                kb_ids = [kb["id"] for kb in kb_options]
                
                selected_kb_index = st.sidebar.selectbox(
                    "选择知识库:",
                    range(len(kb_names)),
                    format_func=lambda i: kb_names[i]
                )
                
                # 保存选择的知识库ID
                st.session_state.selected_kb_id = kb_ids[selected_kb_index]
                st.session_state.selected_kb_name = kb_names[selected_kb_index]
            else:
                st.sidebar.warning("没有可用的知识库。请先创建一个知识库。")
                st.session_state.selected_kb_id = None
        
        # 难度设置
        st.sidebar.subheader("难度设置")
        difficulty = st.sidebar.select_slider(
            "选择难度:",
            options=["初级", "中级", "高级"],
            value="中级"
        )
        
        # 模型选择
        st.sidebar.subheader("模型设置")
        model = st.sidebar.selectbox(
            "选择模型:",
            options=["gpt-3.5-turbo", "gpt-4"],
            index=0
        )
        
        # 会话历史管理
        st.sidebar.subheader("会话管理")
        if st.sidebar.button("保存当前会话"):
            save_current_session()
            st.sidebar.success("会话已保存！")
        
        # 显示历史会话
        if st.session_state.sessions:
            st.sidebar.subheader("历史会话")
            sessions_list = list(st.session_state.sessions.items())
            sessions_list.sort(key=lambda x: x[1]["timestamp"], reverse=True)
            
            for session_id, session in sessions_list:
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    if st.button(f"{session['name']}", key=f"load_{session_id}"):
                        load_session(session_id)
                with col2:
                    if st.button("删除", key=f"delete_{session_id}"):
                        delete_session(session_id)
        
        # 新对话按钮
        if st.sidebar.button("开始新对话"):
            st.session_state.messages = []
            st.session_state.current_session_id = str(uuid.uuid4())
            st.experimental_rerun()
    
    # API设置
    with st.sidebar.expander("API设置", expanded=False):
        api_key = st.text_input("OpenAI API Key:", value=OPENAI_API_KEY, type="password")
        if api_key != OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = api_key
    
    # 显示关于信息
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**ChatMaster** - 智能英语外教系统\n\n"
        "基于人工智能的英语口语学习助手。\n\n"
        "© 2023 ChatMaster 团队"
    )

# 渲染语音交互界面
def render_speech_interaction():
    st.header("英语口语练习")
    
    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 用户输入 - 语音模式
    st.subheader("语音输入")
    
    # 模拟麦克风录音功能
    col1, col2 = st.columns(2)
    with col1:
        if st.button("开始录音", use_container_width=True):
            st.info("正在录音...请对着麦克风说话")
            # 这里添加实际的录音代码
            # 模拟录音过程
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # 模拟语音识别
            user_input = "这是用户语音输入的模拟文本。在实际应用中，这里会是ASR的结果。"
            
            # 添加用户消息
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # 添加助手响应
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "这是助手的回复。在实际应用中，这里会是LLM生成的回复。"
            })
            
            st.experimental_rerun()
    
    with col2:
        if st.button("取消录音", use_container_width=True):
            st.warning("录音已取消")
    
    # 文本输入备选
    with st.expander("文本输入 (备选)", expanded=False):
        user_input = st.text_area("输入文本消息:", height=100)
        if st.button("发送"):
            # 添加用户消息
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # 处理回复逻辑
            # 如果选择了知识库，使用RAG
            if st.session_state.selected_kb_id:
                with st.spinner("AI思考中..."):
                    # 调用RAG查询
                    result = rag_query(
                        question=user_input,
                        knowledge_base_id=st.session_state.selected_kb_id,
                        conversation_id=st.session_state.current_session_id
                    )
                    
                    if not result.get("error", False):
                        answer = result["data"]["answer"]
                        # 添加助手响应
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        # 添加错误消息
                        st.session_state.messages.append({"role": "assistant", "content": f"抱歉，发生了错误: {result['message']}"})
            else:
                # 使用常规LLM对话
                # 模拟助手响应
                with st.spinner("AI思考中..."):
                    time.sleep(1)  # 模拟API调用延迟
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "这是助手的回复。在实际应用中，这里会是LLM生成的回复。"
                    })
            
            st.experimental_rerun()

# 渲染文本聊天界面
def render_text_chat():
    st.header("英语对话练习")
    
    # 显示当前会话信息
    if st.session_state.selected_kb_id:
        st.info(f"当前使用知识库: {st.session_state.selected_kb_name}")
    
    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 如果是助手消息，显示推荐问题
            if message["role"] == "assistant" and message == st.session_state.messages[-1]:
                # 生成推荐问题
                if "recommended_questions" not in st.session_state:
                    last_user_msg = next((msg["content"] for msg in reversed(st.session_state.messages) if msg["role"] == "user"), "")
                    st.session_state.recommended_questions = generate_recommended_questions(
                        last_user_msg, message["content"]
                    )
                
                # 显示推荐问题
                if st.session_state.recommended_questions:
                    st.markdown("**你可能想问:**")
                    cols = st.columns(len(st.session_state.recommended_questions))
                    for i, question in enumerate(st.session_state.recommended_questions):
                        with cols[i]:
                            if st.button(question, key=f"rec_{i}"):
                                st.session_state.user_input = question
                                st.session_state.submit_question = True
    
    # 用户输入
    user_input = st.chat_input("输入你的问题...")
    
    # 处理推荐问题点击
    if st.session_state.submit_question:
        user_input = st.session_state.user_input
        st.session_state.user_input = ""
        st.session_state.submit_question = False
    
    if user_input:
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 清除之前的推荐问题
        if "recommended_questions" in st.session_state:
            del st.session_state.recommended_questions
        
        # 如果选择了知识库，使用RAG
        if st.session_state.selected_kb_id:
            with st.spinner("AI思考中..."):
                # 调用RAG查询
                result = rag_query(
                    question=user_input,
                    knowledge_base_id=st.session_state.selected_kb_id,
                    conversation_id=st.session_state.current_session_id
                )
                
                if not result.get("error", False):
                    # 添加助手响应
                    st.session_state.messages.append({"role": "assistant", "content": result["data"]["answer"]})
                    
                    # 保存推荐问题
                    st.session_state.recommended_questions = result["data"].get("recommended_questions", [])
                else:
                    # 添加错误消息
                    st.session_state.messages.append({"role": "assistant", "content": f"抱歉，发生了错误: {result['message']}"})
        else:
            # 使用常规LLM对话
            with st.spinner("AI思考中..."):
                # 这里应该是调用OpenAI API的代码
                # 模拟响应
                time.sleep(1)
                response = "这是助手的回复。在实际应用中，这里会是LLM生成的回复。"
                
                # 添加助手响应
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # 模拟推荐问题
                st.session_state.recommended_questions = generate_recommended_questions(user_input, response)
        
        st.experimental_rerun()

# 主函数
def main():
    # 加载会话历史
    if not st.session_state.sessions:
        loaded_sessions = load_session_history()
        if loaded_sessions:
            st.session_state.sessions = loaded_sessions
    
    # 渲染侧边栏
    render_sidebar()
    
    # 根据当前视图渲染页面
    if st.session_state.current_view == "chat":
        # 渲染聊天页面
        st.title("ChatMaster 英语对话练习")
        
        # 根据交互模式渲染不同的界面
        if st.session_state.interaction_mode == "speech":
            render_speech_interaction()
        else:
            render_text_chat()
    
    elif st.session_state.current_view == "enhanced_retrieval":
        # 渲染增强型检索界面
        render_enhanced_rag_interface()
        
    elif st.session_state.current_view == "performance_dashboard":
        # 渲染性能仪表盘
        retrieval_dashboard_main()
            
    elif st.session_state.current_view == "knowledge_base_management":
        # 渲染知识库管理页面
        render_knowledge_base_management()
        
    elif st.session_state.current_view == "document_management":
        # 渲染文档管理页面
        if st.session_state.selected_kb_id and st.session_state.selected_kb_name:
            render_document_management(
                kb_id=st.session_state.selected_kb_id,
                kb_name=st.session_state.selected_kb_name
            )
        else:
            st.warning("请先选择一个知识库")
            if st.button("前往知识库管理"):
                st.session_state.current_view = "knowledge_base_management"
                st.experimental_rerun()

if __name__ == "__main__":
    main()
