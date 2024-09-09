import json

# 定义一个函数将多个部分拼接为system字段的内容
def create_system_field(scene, role_num, difficulty, difficulty_description):
    # 自定义拼接方式，可以根据需要调整
    return f"Scene: {scene} | Role Count: {role_num} | Difficulty: {difficulty} | Description: {difficulty_description}"

def convert_to_xtuner_format(data):
    xtuner_data = []

    for item in data:
        # 拼接system字段内容
        system = create_system_field(
            scene=item["scene"],
            role_num=item["role_num"],
            difficulty=item["difficulty"],
            difficulty_description=item["difficulty_description"]
        )

        # 构建conversation部分
        conversation = []
        
        # 添加对话的每一轮
        for dialogue in item["dialogues"]:
            character = dialogue["character"]
            text = dialogue["text"]
            
            # 将每个角色的对话填入conversation中
            conversation.append({
                "system": system,  # 拼接好的system字段
                "input": character,  # 角色名作为input
                "output": text  # 角色的文本作为output
            })
        
        # 将conversation添加到xtuner_data
        xtuner_data.append({
            "conversation": conversation
        })
    
    return xtuner_data

# 读取JSON文件
with open('./friends_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换为XTuner格式
xtuner_data = convert_to_xtuner_format(data)

# 输出为json文件
with open('xtuner_data.json', 'w', encoding='utf-8') as f:
    json.dump(xtuner_data, f, ensure_ascii=False, indent=4)

print("转换完成，XTuner格式数据已保存为xtuner_data.json")


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
