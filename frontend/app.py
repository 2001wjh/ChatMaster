# 文件名：app.py

import streamlit as st
from openai import OpenAI
import logging

# ========================
# 配置和初始化
# ========================

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置页面的基本配置
st.set_page_config(page_title="ChatMaster", layout="wide")

# 检查并设置 OpenAI API 密钥
if "OPENAI_API_KEY" not in st.secrets:
    st.error("请在 Streamlit Secrets 中配置 OPENAI_API_KEY。")
    st.stop()

try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error("初始化 OpenAI 客户端失败，请检查 API 密钥。")
    logger.exception("OpenAI 客户端初始化失败：%s", e)
    st.stop()

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = ""

# ========================
# 侧边栏 - 设置选项
# ========================

def render_sidebar():
    """渲染侧边栏，提供设置选项。"""
    with st.sidebar:
        st.header("设置")
        # 模型选择器
        st.session_state.model_choice = st.selectbox(
            "选择对话模型",
            options=["gpt-3.5-turbo", "gpt-4"],
            index=0
        )
        # 语言选择器
        st.session_state.language_choice = st.selectbox(
            "选择对话语言",
            options=["中文", "English"],
            index=1  # 默认选择 English
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

def handle_user_input(prompt):
    """处理用户输入，生成 AI 回复。"""
    # 将用户输入添加到会话状态
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 OpenAI 接口生成回复
    with st.chat_message("assistant"):
            # 准备请求参数
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

            # 调用 OpenAI 接口，启用流式输出
            stream = client.chat.completions.create(
                model=st.session_state.model_choice,
                messages=messages,
                stream=True,
            )

            # 使用 st.write_stream 处理流式输出并获取响应
            response = st.write_stream(stream)

    # 将 AI 回复添加到会话状态
    st.session_state.messages.append({"role": "assistant", "content": response})


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
        st.error("未识别的交流模式。")

if __name__ == "__main__":
    main()
