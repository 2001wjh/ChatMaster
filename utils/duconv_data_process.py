import json
import re

# 去除句子中中英文字符与标点符号之间的多余空格
def remove_extra_spaces(text):
    if isinstance(text, str):
        return ''.join(text.split())  # 删除所有多余空格
    return text

# 处理每行json数据，提取并清理conversation字段
def process_line(line):
    try:
        data = json.loads(line)
        conversation = data.get("conversation", [])
        cleaned_conversation = [remove_extra_spaces(sentence) for sentence in conversation]
        return {"conversation": cleaned_conversation}
    except json.JSONDecodeError:
        print(f"Error decoding line: {line}")
        return None

# 转换为微调格式的函数
def convert_to_finetune_format(conversation_list):
    finetune_conversation = []
    
    # 逐行处理对话内容，确保不重复
    for i in range(0, len(conversation_list), 2):  # 每次跳两行，避免重复
        input_sentence = conversation_list[i].strip()  # 当前句子作为输入
        if i + 1 < len(conversation_list):  # 确保有下一句
            output_sentence = conversation_list[i + 1].strip()  # 下一句作为输出
        else:
            output_sentence = ""  # 如果没有下一句，留空

        finetune_conversation.append({
            "input": input_sentence,
            "output": output_sentence
        })

    return finetune_conversation

# 主函数：读取txt文件，清理数据，保存为json，并转换为微调格式
def process_and_convert(input_txt_file, cleaned_output_file, finetune_output_file):
    cleaned_data = []

    # 读取txt文件并处理每一行
    with open(input_txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            cleaned_line = process_line(line.strip())  # 逐行读取并处理
            if cleaned_line:  # 只添加有效的conversation数据
                cleaned_data.append(cleaned_line)

    # # 保存清理后的conversation到新的JSON文件
    # with open(cleaned_output_file, 'w', encoding='utf-8') as f:
    #     json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
    
    # print(f"Cleaned conversation data saved to {cleaned_output_file}")

    # 将清理后的数据转换为微调格式
    finetune_data = []

    # 遍历每个conversation并转换格式
    for item in cleaned_data:
        conversation = item.get("conversation", [])
        finetune_conversation = convert_to_finetune_format(conversation)
        finetune_data.append({"conversation": finetune_conversation})

    # 保存转换后的数据为新的JSON文件
    with open(finetune_output_file, 'w', encoding='utf-8') as f:
        json.dump(finetune_data, f, ensure_ascii=False, indent=4)
    
    print(f"Finetune-ready conversation data saved to {finetune_output_file}")

# 示例调用
input_txt_file = '../dataset/initial_data/duconv_train.txt'  # 输入txt文件路径
cleaned_output_file = '../dataset/gen_dataset/cleaned_conversations.json'  # 清理后的JSON文件路径
finetune_output_file = '../dataset/gen_dataset/duconv_dataset.json'  # 微调格式的JSON文件路径

process_and_convert(input_txt_file, cleaned_output_file, finetune_output_file)
