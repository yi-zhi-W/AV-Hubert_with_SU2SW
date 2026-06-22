import json

input_file = "/home/wyz/projects/LLaMA-Factory/data/ug_test.csv"
output_file = "/home/wyz/projects/LLaMA-Factory/data/ug_test_id_full.json"

json_list = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        audio_path, text = line.strip().split('\t', 1)
        new_path = "/home/wyz/data/UG/clips/" + audio_path.split('/')[-1]
        
        json_list.append({
            "messages": [
                {"role": "user", "content": "<audio>Generate the caption in Uyghur:"},
                {"role": "assistant", "content": text}
            ],
            "audios": [new_path],
            "language_id": 0
        })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(json_list, f, indent=2, ensure_ascii=False)