import json
from collections import Counter


# 读取JSON文件（假设文件名为data.json）
with open('ug_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取所有assistant的content
contents = []
for item in data:
    for message in item['messages']:
        if message['role'] == 'assistant':
            contents.append(message['content'])

# 合并所有内容并统计字符
all_content = ''.join(contents)
char_counts = Counter(all_content)

# 按字符出现次数降序排序
sorted_counts = sorted(char_counts.items(), key=lambda x: (-x[1], x[0]))

# 输出结果
print("字符统计结果（按出现次数降序，次数相同按字符升序）：")
num = 1
cha_map = {}
num_map = {}
for char, count in sorted_counts:
    if char==' ':
        continue
    else:
        cha_map[char] = num
        num_map[num] = char
        num = num+1
print(cha_map)
print(num_map)