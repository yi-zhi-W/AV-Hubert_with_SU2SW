import csv
import json

def csv_to_json(csv_path, json_path):
    # 读取CSV文件
    with open(csv_path, 'r', encoding='utf-8') as f:
        # 使用制表符分隔
        reader = csv.reader(f, delimiter='\t')
        data = []
        
        for row in reader:
            if len(row) != 2:  # 跳过格式错误的行
                continue
            
            audio_path, text = row[0].strip(), row[1].strip()
            
            # 构建消息结构
            messages = [
                {
                    "content": "<audio>Generate the caption in Turkish:",
                    "role": "user"
                },
                {
                    "content": text,
                    "role": "assistant"
                }
            ]
            
            # 构建完整条目
            entry = {
                "messages": messages,
                "audios": [audio_path],
                "language_id": 4
            }
            
            data.append(entry)
    
    # 写入JSON文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 使用示例
csv_to_json("/nfs4/wyz/TR/cv-corpus-17.0-2024-03-15/dev.csv", "/nfs4/wyz/TR/cv-corpus-17.0-2024-03-15/dev.jsonl")