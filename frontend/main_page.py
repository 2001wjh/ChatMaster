from openai import OpenAI
import interface.streamlit as st

# 设置页面的宽度和基础样式
st.set_page_config(page_title="ChatMaster", layout="wide")

# 设置API密钥
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 添加侧边栏，用户可以选择交流模式
with st.sidebar:
    st.header("设置")
    # 模型选择器
    model_choice = st.selectbox(
        "选择对话模型",
        options=["gpt-3.5-turbo", "gpt-4"],
        index=0
    )
    # 语言选择器
    language_choice = st.selectbox(
        "选择对话语言",
        options=["中文", "English"],
        index=1  # 默认选择English
    )
    # 增加界面切换选项
    interaction_mode = st.selectbox(
        "选择交流模式",
        options=["口语交流", "文字对话"],
        index=1  # 默认文字对话
    )

    st.write("选择不同的模式以获得个性化体验。")

# 根据用户选择的交流模式渲染不同的界面
if interaction_mode == "口语交流":
    # 显示数字人图片，准备用于口语交流
    # st.image("./assert/image/数字人.png", caption="准备开始口语交流", use_column_width=True)
    st.image("../assert/image/数字人.png", caption="准备开始口语交流", width=800)
    
    # 这里你可以集成一个语音输入功能
    st.write("请使用语音对话功能（未来可添加语音识别接口）。")

elif interaction_mode == "文字对话":
    # 设置自定义标题
    st.title("🦜🔗 ChatMaster - 你的专属英文外教")

    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 在用户输入前，插入 system 提示消息，但不显示在UI中
    if language_choice == "中文":
        system_prompt = "你必须使用中文与用户对话"
    else:
        system_prompt = "You must respond in English to the user."

    # 如果没有 system 消息或需要更新，先更新 system 消息
    if not st.session_state.messages or st.session_state.messages[0]["role"] != "system":
        st.session_state.messages.insert(0, {"role": "system", "content": system_prompt})
    else:
        st.session_state.messages[0]["content"] = system_prompt

    # 显示历史对话（跳过 system 消息）
    for message in st.session_state.messages:
        if message["role"] != "system":  # 不显示系统消息
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 用户输入框
    if prompt := st.chat_input("请输入你的问题"):
        # 将用户输入添加到会话状态
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 生成 AI 回复
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=model_choice,
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
