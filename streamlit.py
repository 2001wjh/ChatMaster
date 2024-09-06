from openai import OpenAI
import streamlit as st

# 设置页面的宽度和基础样式
st.set_page_config(page_title="ChatMaster", layout="wide")



# 设置API密钥
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 添加侧边栏，用户可以选择交流模式
with st.sidebar:
    st.header("设置")
    # 模型选择器
    model_choice = st.selectbox(
        "选择模型",
        options=["gpt-3.5-turbo", "gpt-4"],
        index=0
    )
    # 语言选择器
    language_choice = st.selectbox(
        "选择语言",
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
    st.image("./assert/image/数字人.png", caption="准备开始口语交流", width=500)
    
    # 这里你可以集成一个语音输入功能
    st.write("请使用语音对话功能（未来可添加语音识别接口）。")

elif interaction_mode == "文字对话":
    # 设置自定义标题
    st.title("🦜🔗 ChatMaster - 你的专属英文外教")

    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 显示历史对话
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 用户输入框
    if prompt := st.chat_input("请输入你的问题"):
        # 如果选择了中文，添加提示让模型回答中文
        if language_choice == "中文":
            prompt = f"请用中文回答: {prompt}"

        # 将用户输入添加到会话状态
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 实时显示用户输入
        with st.chat_message("user"):
            st.markdown(prompt)

        # 添加加载状态，显示“AI正在思考中...”
        with st.chat_message("assistant"):
            assistant_placeholder = st.empty()

            # 准备消息记录
            messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

            # 调用OpenAI API，流式获取回答
            stream = client.chat.completions.create(
                model=model_choice,  # 动态选择模型
                messages=messages,
                temperature=0.5,
                stream=True
            )

            # 实时接收并流式显示AI的回答
            full_response = ""
            for chunk in stream:
                content = chunk['choices'][0]['delta'].get('content', '')
                full_response += content
                assistant_placeholder.markdown(full_response)

        # 保存AI的完整回答
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# 添加进阶功能：显示请求计数与Token花费
with st.sidebar:
    if st.session_state.messages:
        total_tokens = sum(len(msg['content']) for msg in st.session_state.messages)
        st.write(f"已使用Token数量: {total_tokens}")


# from openai import OpenAI
# import streamlit as st

# # 设置页面的宽度和基础样式
# st.set_page_config(page_title="ChatMaster", layout="wide")

# # 设置自定义标题
# st.title("🦜🔗 ChatMaster - 你的专属英文外教")

# # 添加侧边栏，用户可以选择模型和语言
# with st.sidebar:
#     st.header("设置")
#     # 在侧边栏添加数字人图片
#     st.image("./assert/image/数字人.png", caption="AI 外教", use_column_width=True)
    
#     # 模型选择器和语言选择器等控件
#     model_choice = st.selectbox(
#         "选择模型",
#         options=["gpt-3.5-turbo", "gpt-4"],
#         index=0
#     )
#     language_choice = st.selectbox(
#         "选择语言",
#         options=["中文", "English"],
#         index=1
#     )
#     st.write("选择不同的模型和语言以获得个性化体验。")

# # 设置API密钥
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# # 初始化会话状态
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # 显示历史对话
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # 用户输入框
# if prompt := st.chat_input("请输入你的问题"):
#     # 如果选择了中文，添加提示让模型回答中文
#     if language_choice == "中文":
#         prompt = f"请用中文回答: {prompt}"
    
#     # 将用户输入添加到会话状态
#     st.session_state.messages.append({"role": "user", "content": prompt})
    
#     # 实时显示用户输入
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # 添加加载状态，显示“AI正在思考中...”
#     with st.chat_message("assistant"):
#         assistant_placeholder = st.empty()

#         # 准备消息记录
#         messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

#         # 调用OpenAI API，流式获取回答
#         stream = client.chat.completions.create(
#             model=model_choice,  # 动态选择模型
#             messages=messages,
#             temperature=0.5,
#             stream=True
#         )

#         # 实时接收并流式显示AI的回答
#         full_response = ""
#         for chunk in stream:
#             content = chunk['choices'][0]['delta'].get('content', '')
#             full_response += content
#             assistant_placeholder.markdown(full_response)  # 动态更新界面

#     # 保存AI的完整回答
#     st.session_state.messages.append({"role": "assistant", "content": full_response})

# # 添加进阶功能：显示请求计数与Token花费
# with st.sidebar:
#     if st.session_state.messages:
#         total_tokens = sum(len(msg['content']) for msg in st.session_state.messages)
#         st.write(f"已使用Token数量: {total_tokens}")



# # from openai import OpenAI
# # import streamlit as st

# # st.title("🦜🔗 ChatMaster")

# # client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# # if "openai_model" not in st.session_state:
# #     st.session_state["openai_model"] = "gpt-3.5-turbo"

# # if "messages" not in st.session_state:
# #     st.session_state.messages = []

# # for message in st.session_state.messages:
# #     with st.chat_message(message["role"]):
# #         st.markdown(message["content"])

# # if prompt := st.chat_input("What is up?"):
# #     st.session_state.messages.append({"role": "user", "content": prompt})
# #     with st.chat_message("user"):
# #         st.markdown(prompt)

# #     with st.chat_message("assistant"):
# #         stream = client.chat.completions.create(
# #             model=st.session_state["openai_model"],
# #             messages=[
# #                 {"role": m["role"], "content": m["content"]}
# #                 for m in st.session_state.messages
# #             ],
# #             stream=True,
# #         )
# #         response = st.write_stream(stream)
# #     st.session_state.messages.append({"role": "assistant", "content": response})