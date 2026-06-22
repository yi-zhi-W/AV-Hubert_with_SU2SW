import json
import os

def transform_json(input_data):
    output_data = []
    for item in input_data:
        # 创建新对象
        new_item = {
            "messages": [
                {
                    "content": item["messages"][0]["content"],
                    "role": "user"
                },
                {
                    "content": item["messages"][1]["content"],
                    "role": "assistant"
                }
            ],
            "audios": item["audios"],
            "language_id": 0
        }
        output_data.append(new_item)
    return output_data

# 使用示例
if __name__ == "__main__":
    # 从文件读取原始JSON
    with open("ug_train.json", "r", encoding="utf-8") as f:
        original_data = json.load(f)
    
    # 转换数据
    transformed_data = transform_json(original_data)
    
    # 写入新文件
    with open("ug_train_id.json", "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=2)