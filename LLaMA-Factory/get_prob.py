import json
import os

def extract_txt_data(txt_path):
    try:
        with open(txt_path, 'r') as file:
            content = file.read()
        
        # 查找第一个Shape块
        start_idx = content.find("Shape: (1, 5)")
        if start_idx == -1:
            raise ValueError(f"在文件 {txt_path} 中未找到Shape标记")
        
        # 定位数据行开始位置
        data_start = content.find("\n", start_idx) + 1
        if data_start == 0:
            raise ValueError(f"在文件 {txt_path} 中Shape标记后没有数据")
        
        # 提取第一行数据
        end_line = content.find("\n", data_start)
        data_line = content[data_start:end_line].strip()
        
        # 分割数据点
        data_points = data_line.split()
        # if len(data_points) < 20:
        #     raise ValueError(f"文件 {txt_path} 中数据点不足，需要至少20个，实际只有{len(data_points)}个")
        
        return [float(x) for x in data_points]
    
    except Exception as e:
        print(f"处理文件 {txt_path} 时出错: {str(e)}")
        return None

def process_json_file(json_path, output_path=None):
    """处理JSON文件并更新language_id字段"""
    if output_path is None:
        output_path = json_path  # 默认覆盖原文件
    
    try:
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 处理每个条目
        for item in data:
            if "language_id" in item:
                # 获取原始language_id值
                orig_id = item["language_id"]
                
                # 构建txt文件路径
                txt_path = "/home/wyz/projects/LLaMA-Factory/softmax_id_logits/"+str(orig_id)+".txt"
                
                # 检查文件是否存在
                if not os.path.exists(txt_path):
                    print(f"警告: 文件 {txt_path} 不存在，跳过处理")
                    continue
                
                # 从txt文件提取数据
                extracted_data = extract_txt_data(txt_path)
                
                if extracted_data is not None:
                    # 更新language_id字段
                    item["language_id"] = extracted_data
                    print(f"成功更新条目 {orig_id} => {extracted_data}")
        
        # 保存更新后的JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成! 结果已保存到: {output_path}")
        return True
    
    except Exception as e:
        print(f"处理JSON文件时出错: {str(e)}")
        return False

# 使用示例
if __name__ == "__main__":
    # 输入JSON文件路径
    input_json = "/home/wyz/projects/LLaMA-Factory/data/test_softmax.json"
    
    # 输出文件路径（可选，不提供则覆盖原文件）
    output_json = "/home/wyz/projects/LLaMA-Factory/data/test_softmax.json"
    
    # 处理文件
    process_json_file(input_json, output_json)
























# import json

# def update_language_ids(input_file, output_file=None):
#     """
#     更新JSON文件中的language_id字段，使其从0开始依次递增
    
#     参数:
#         input_file (str): 输入JSON文件路径
#         output_file (str): 输出JSON文件路径（默认为None，覆盖原文件）
#     """
#     # 默认输出文件为输入文件（覆盖原文件）
#     if output_file is None:
#         output_file = input_file
    
#     # 读取JSON文件
#     with open(input_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     # 验证数据结构
#     if not isinstance(data, list):
#         raise ValueError("JSON文件应包含一个对象列表")
    
#     # 更新language_id
#     for idx, item in enumerate(data):
#         # 确保对象包含language_id字段
#         if 'language_id' not in item:
#             print(f"警告: 索引 {idx} 的对象缺少 language_id 字段，已添加")
#             item['language_id'] = idx
#         else:
#             # 更新为当前索引值
#             item['language_id'] = idx
    
#     # 写入更新后的JSON
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
    
#     print(f"成功更新 {len(data)} 条记录的 language_id")
#     print(f"结果已保存到: {output_file}")

# # 示例使用
# if __name__ == "__main__":
#     # 输入文件路径
#     input_json = "/home/wyz/projects/LLaMA-Factory/data/mtl_5.json"  # 替换为实际文件路径
    
#     # 执行更新（覆盖原文件）
#     # update_language_ids(input_json)
    
#     # 可选：保存到新文件
#     update_language_ids(input_json, "/home/wyz/projects/LLaMA-Factory/data/mtl_5_updated_data.json")