import json

# 读取输入的json文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 定义转换函数，确保input和output不重复，并保留对话划分
def convert_format(data, use_character_as_input=True):
    all_conversations = []

    # 处理data中的每个项（每个项是一个对话场景）
    for item in data:
        dialogues = item.get("dialogues", [])  # 获取dialogues字段
        conversation_list = []  # 用于保存当前对话场景的所有input-output对

        # 遍历dialogues，并确保input和output不重复
        for i in range(0, len(dialogues) - 1, 2):  # 每次跳过一对对话，input和output相邻
            if use_character_as_input:
                # 定义input为当前角色(character)，output为下一句的台词(text)
                input_sentence = dialogues[i].get("character", "") + ": " + dialogues[i].get("text", "")
                output_sentence = dialogues[i + 1].get("character", "") + ": " + dialogues[i + 1].get("text", "")
            else:
                # 定义input和output均为text
                input_sentence = dialogues[i].get("text", "")
                output_sentence = dialogues[i + 1].get("text", "")

            conversation_entry = {
                "input": input_sentence,
                "output": output_sentence
            }
            conversation_list.append(conversation_entry)

        # 每个对话场景的conversation保留独立结构
        all_conversations.append({"conversation": conversation_list})

    # 返回最终的所有对话
    return all_conversations

# 保存结果到JSON文件
def save_json(data, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 主函数
def process_json_file(input_file, output_file, use_character_as_input=True):
    # 加载json文件
    input_data = load_json(input_file)
    
    # 格式转换
    output_data = convert_format(input_data, use_character_as_input)
    
    # 保存结果
    save_json(output_data, output_file)
    print(f"转换完成，结果已保存到 {output_file}")

# 示例用法
input_file_path = '../initial_data/friends_data.json'  # 你的输入json文件路径
output_file_path = '../gen_dataset/friends_datasets.json'  # 生成的输出json文件路径

# 调用主函数
process_json_file(input_file_path, output_file_path, use_character_as_input=False)  # 可以将use_character_as_input设置为False


# import json

# # 定义一个函数将多个部分拼接为system字段的内容
# def create_system_field(scene, role_num, difficulty, difficulty_description):
#     # 自定义拼接方式，可以根据需要调整
#     return f"Scene: {scene} | Role Count: {role_num} | Difficulty: {difficulty} | Description: {difficulty_description}"

# def convert_to_xtuner_format(data):
#     xtuner_data = []

#     for item in data:
#         # 拼接system字段内容
#         system = create_system_field(
#             scene=item["scene"],
#             role_num=item["role_num"],
#             difficulty=item["difficulty"],
#             difficulty_description=item["difficulty_description"]
#         )

#         # 构建conversation部分
#         conversation = []
#         num = 0
        
#         # 添加对话的每一轮
#         for dialogue in item["dialogues"]:
#             character = dialogue["character"]
#             text = dialogue["text"]
            
#             # 将每个角色的对话填入conversation中
#             if num == 0:
#                 num += 1
#                 conversation.append({
#                     "system": system,
#                     "input": character,  # 角色名作为input
#                     "output": text  # 角色的文本作为output
#                 })
#             else:
#                 conversation.append({
#                     "input": character,  # 角色名作为input
#                     "output": text  # 角色的文本作为output
#                 })
       
        
#         # 将conversation添加到xtuner_data
#         xtuner_data.append({
#             "conversation": conversation
#         })
    
#     return xtuner_data

# # 读取JSON文件
# with open('../initial_data/friends_data.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # 转换为XTuner格式
# xtuner_data = convert_to_xtuner_format(data)

# # 输出为json文件
# with open('../gen_dataset/xtuner_data_friends.json', 'w', encoding='utf-8') as f:
#     json.dump(xtuner_data, f, ensure_ascii=False, indent=4)

# print("转换完成，数据已保存到 gen_dataset 目录下的 xtuner_data_friends.json 文件中")


# def find_missing_fields(data):
#     missing_entries = []  # 用于存储缺少字段的对话条目

#     for idx, item in enumerate(data):
#         dialogues = item.get("dialogues", [])
        
#         for dialogue_idx, dialogue in enumerate(dialogues):
#             missing_fields = []
            
#             # 检查character字段
#             if "character" not in dialogue:
#                 missing_fields.append("character")
            
#             # 检查text字段
#             if "text" not in dialogue:
#                 missing_fields.append("text")
            
#             # 如果有缺失的字段，将信息添加到 missing_entries
#             if missing_fields:
#                 missing_entries.append({
#                     "scene": item.get("scene", "Unknown"),
#                     "dialogue_index": dialogue_idx,
#                     "missing_fields": missing_fields
#                 })

#     return missing_entries

# # 读取JSON文件
# with open('./friends_data.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # 查找缺少字段的对话条目
# missing_data = find_missing_fields(data)

# # 输出缺少字段的对话条目信息
# if missing_data:
#     print("以下对话条目缺少字段:")
#     for entry in missing_data:
#         print(f"Scene: {entry['scene']}, Dialogue Index: {entry['dialogue_index']}, Missing Fields: {', '.join(entry['missing_fields'])}")
# else:
#     print("所有对话条目都完整，没有缺少字段。")
