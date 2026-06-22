import json

def convert_to_unicode_str(text):
    """将字符串转换为 Unicode 十进制值加下划线的形式"""
    unicode_str = ""
    for char in text:
        if char not in [' ']:
            unicode_str = unicode_str+str(ord(char))
            unicode_str = unicode_str+"_"
        else:
            unicode_str = unicode_str+" "
    return unicode_str

# 读取 JSON 文件（假设文件名为 data.json）
with open('ksc_tr_ug_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历处理每个 assistant 的 content
for item in data:
    for message in item['messages']:
        if message['role'] == 'assistant':
            original_content = message['content']
            message['content'] = convert_to_unicode_str(original_content)

# 保存转换后的结果到新文件 converted_data.json
with open('3language_unicode_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)