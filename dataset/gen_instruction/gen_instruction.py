import os
import openai
import gradio as gr
from dotenv import load_dotenv, find_dotenv

# 定义从消息中获取回复的函数
def get_completion(messages, model="gpt-4o"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        if response and response.choices:
            message = response.choices[0].message
            if message:
                return message.content
            else:
                print("没有返回消息内容。")
        else:
            print("无效的回复或未返回选择。")
    except Exception as e:
        print(f"发生错误: {e}")
    return "对不起，发生了错误。"

# 从文件中读取系统提示内容的函数
def read_system_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 定义处理聊天输入的函数
def chat(user_input):
    # 指定系统提示文件的路径
    PROMPT_FILE_ROOT = r"../prompt_template/friends.txt"
    PROMPT_FILE_ROOT = r"../prompt_template/the_big_bang_theory.txt"

    # 读取系统提示内容
    system_prompt = read_system_prompt(PROMPT_FILE_ROOT)
    # print(system_prompt_content)

    # 定义初始上下文
    context = [
        {'role': 'system', 'content': system_prompt}
    ]

    print(f"用户输入: {user_input}")
    context.append({'role': 'user', 'content': user_input})
    response = get_completion(context)
    context.append({'role': 'assistant', 'content': response})
    print(f"AI 回复: {response}")
    return response

if __name__ == "__main__":
    # os.environ["no_proxy"] = "localhost,127.0.0.1,::1"

    # 加载环境变量
    _ = load_dotenv(find_dotenv())

    client = openai.OpenAI()

    FLAGGING_DIR = r"./gen_dataset"

    # 创建 Gradio 界面
    interface = gr.Interface(
        fn=chat,
        inputs=gr.Textbox(lines=7, label="请输入你的消息:"),
        outputs=gr.Textbox(label="回复:"),
        flagging_dir=FLAGGING_DIR,
        title="ChatMaster",
        description="熟悉《老友记》剧情，对角色性格和对话风格有深刻理解。"
    )

    # 启动界面
    interface.launch()