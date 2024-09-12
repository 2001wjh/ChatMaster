import json

def process_txt_to_json(txt_file_path, json_file_path):
    conversations = []
    
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        
        for line in lines:
            # 分割每段对话的句子
            dialogue = line.strip().split('__eou__')[:-1]  # 舍弃最后一个__eou__
            dialogue = [d.strip() for d in dialogue if d.strip()]  # 去除空白项
            
            # 确保句子是偶数对，奇数对话舍弃最后一个
            if len(dialogue) % 2 != 0:
                dialogue = dialogue[:-1]
            
            # 每两个句子为一组，A为input，B为output
            conversation = []
            for i in range(0, len(dialogue), 2):
                conversation.append({
                    "input": dialogue[i],
                    "output": dialogue[i+1]
                })
            
            if conversation:
                conversations.append({"conversation": conversation})
    
    # 将结果保存为json文件
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(conversations, json_file, indent=4, ensure_ascii=False)

    print(f"JSON文件已成功保存到 {json_file_path}")

# 使用示例
txt_file_path = '/home/hce/project_llm/initial_data/ijcnlp_dailydialog/dialogues_text.txt'  # 输入的txt文件路径
json_file_path = '/home/hce/project_llm/initial_data/ijcnlp_dailydialog/English_dialog.json'  # 输出的json文件路径

process_txt_to_json(txt_file_path, json_file_path)
