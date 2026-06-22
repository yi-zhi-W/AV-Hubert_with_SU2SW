import json
import os

def transform_json(input_data):
    output_data = []
    current_id = 0
    count = 0
    for item in input_data:
        if item["language_id"]==current_id:
            if count>1500:
                continue
            else:
                count = count+1
                new_item = {
                    "messages": [
                        {
                            "content": item["messages"][0]["content"],
                            "role": "user"
                        },
                        {
                            "content": str(item["language_id"]),
                            "role": "assistant"
                        }
                    ],
                    "audios": item["audios"],
                    "language_id": item["language_id"]
                }
                output_data.append(new_item)
        else:
            count = 1
            current_id = item["language_id"]
            new_item = {
                "messages": [
                    {
                        "content": item["messages"][0]["content"],
                        "role": "user"
                    },
                    {
                        "content": str(item["language_id"]),
                        "role": "assistant"
                    }
                ],
                "audios": item["audios"],
                "language_id": item["language_id"]
            }
            output_data.append(new_item)
        # 创建新对象
        
    return output_data

# 使用示例
if __name__ == "__main__":
    # 从文件读取原始JSON
    with open("language_id_test_before.json", "r", encoding="utf-8") as f:
        original_data = json.load(f)
    
    # 转换数据
    transformed_data = transform_json(original_data)
    
    # 写入新文件
    with open("language_id_test_new.json", "w", encoding="utf-8") as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=2)