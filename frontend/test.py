import streamlit as st
import logging
import requests
import json

# ========================
# 配置和初始化
# ========================

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置页面的基本配置
st.set_page_config(page_title="ChatMaster", layout="wide")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = ""

# ========================
# 侧边栏 - 设置选项
# ========================

def get_available_models():
    """从服务器获取可用的模型列表。"""
    api_url = "http://localhost:23333/v1/models"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            models = [model['id'] for model in data['data']]
            print(f"Available models: {models}")
            return models
        else:
            st.error("无法获取可用的模型列表")
            return []
    except Exception as e:
        st.error(f"请求失败：{e}")
        return []

def render_sidebar():
    """渲染侧边栏，提供设置选项。"""
    with st.sidebar:
        st.header("设置")
        # 获取可用模型列表
        available_models = get_available_models()
        if not available_models:
            st.stop()
        # 模型选择器
        st.session_state.model_choice = st.selectbox(
            "选择对话模型",
            options=available_models,
            index=0
        )
        # 语言选择器
        st.session_state.language_choice = st.selectbox(
            "选择对话语言",
            options=["中文", "English"],
            index=0  # 默认选择中文
        )
        # 交流模式选择器
        st.session_state.interaction_mode = st.selectbox(
            "选择交流模式",
            options=["口语交流", "文字对话"],
            index=1  # 默认文字对话
        )
        st.write("选择不同的模式以获得个性化体验。")

# ========================
# 处理系统提示消息
# ========================

def update_system_prompt():
    """根据语言选择更新系统提示消息。"""
    if st.session_state.language_choice == "中文":
        st.session_state.system_prompt = "你必须使用中文与用户对话。"
    else:
        st.session_state.system_prompt = "You must respond in English to the user."

    # 更新或插入 system 消息
    if not st.session_state.messages or st.session_state.messages[0]["role"] != "system":
        st.session_state.messages.insert(0, {"role": "system", "content": st.session_state.system_prompt})
    else:
        st.session_state.messages[0]["content"] = st.session_state.system_prompt

# ========================
# 渲染口语交流模式界面
# ========================

def render_speech_interaction():
    """渲染口语交流模式的界面。"""
    st.title("🗣️ ChatMaster - 口语交流模式")
    st.image("../assert/image/数字人.png", caption="准备开始口语交流", width=600)
    st.info("语音对话功能尚在开发中，敬请期待！")
    # TODO: 集成语音输入和输出功能

# ========================
# 渲染文字对话模式界面
# ========================

def render_text_chat():
    """渲染文字对话模式的界面。"""
    st.title("💬 ChatMaster - 你的专属外教")

    # 显示历史对话（跳过 system 消息）
    for message in st.session_state.messages[1:]:  # 第一个是 system 消息
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入框
    prompt = st.chat_input("请输入你的问题")
    if prompt:
        handle_user_input(prompt)

# ========================
# 处理用户输入
# ========================

def handle_user_input(user_input):
    """处理用户输入，生成 AI 回复。"""
    # 将用户输入添加到会话状态
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 仅发送最近的对话记录，或者根据需要截断对话历史
    messages = st.session_state.messages[-10:]  # 只取最近的10条对话

    # 将 messages 列表转换为字符串形式的 prompt
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            prompt += f"System: {content}\n"
        elif role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
    prompt += "Assistant: "

    # 调用本地 FastAPI 接口，启用非流式输出
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # 设置 FastAPI 服务器的 URL
        api_url = "http://localhost:23333/v1/chat/completions"

        # 准备请求数据，将 prompt 作为 messages 字段发送
        payload = {
            "model": st.session_state.model_choice,
            "messages": prompt,  # 将字符串形式的 prompt 作为 messages 发送
            "temperature": 0.7,
            "top_p": 1.0,
            "n": 1,
            "logprobs": False,
            "stream": False,  # 使用非流式输出
            "max_tokens": 512,
            "repetition_penalty": 1.0,
            "stop": None,
            "top_k": 50,
            "ignore_eos": False,
            "skip_special_tokens": True,
            "tool_choice": "none",
            "tools": None
        }

        # 发送请求并接收响应
        try:
            response = requests.post(api_url, json=payload)
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response headers: {response.headers}")
            result = response.json()
            logger.info(f"Response JSON: {result}")

            if response.status_code == 200:
                # 从响应中获取助手的回复
                assistant_message = result['choices'][0]['message']['content']
                message_placeholder.markdown(assistant_message)
                # 将 AI 回复添加到会话状态
                st.session_state.messages.append({"role": "assistant", "content": assistant_message})
            else:
                # 处理错误信息
                error_message = result.get('message', '未知错误')
                st.error(f"服务器返回错误：{error_message}")
                logger.error(f"服务器返回错误：{error_message}")
        except Exception as e:
            st.error(f"请求失败：{e}")
            logger.error(f"请求失败：{e}")

# ========================
# 主函数
# ========================

def main():
    """应用的主函数。"""
    # 渲染侧边栏
    render_sidebar()

    # 更新系统提示消息
    update_system_prompt()

    # 根据交流模式渲染界面
    if st.session_state.interaction_mode == "口语交流":
        render_speech_interaction()
    elif st.session_state.interaction_mode == "文字对话":
        render_text_chat()
    else:
        st.error(f"未识别的交流模式：{st.session_state.interaction_mode}")

if __name__ == "__main__":
    main()
