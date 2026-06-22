import json

def update_language_ids(input_file, output_file=None):
    """
    更新JSON文件中的language_id字段（从0开始递增），保持assistant响应内容不变
    :param input_file: 输入JSON文件路径
    :param output_file: 输出文件路径（默认覆盖原文件）
    """
    # 设置输出路径（默认覆盖原文件）
    if output_file is None:
        output_file = input_file
    
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 更新language_id（从0开始递增）
    for idx, item in enumerate(data):
        # 只修改顶层的language_id字段
        item["language_id"] = idx
    
    # 写回文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"处理完成！共更新 {len(data)} 条记录，输出文件: {output_file}")

# 使用示例
update_language_ids('test_softmax.json')  # 直接修改原文件
# update_language_ids('input.json', 'output.json')  # 保存为新文件