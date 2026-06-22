import torch
import re
from collections import defaultdict
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
import torch
import re
import os
import torch.nn.functional as F

def extract_and_save_gatemtllora_B_w(data, save_dir='./extracted_weights'):
    """
    从 safetensors 数据中提取所有 gatemtllora_B_w 权重并按指定格式保存
    
    参数:
        data: 包含权重张量的字典
        save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 用于存储提取的权重
    extracted_weights = {}
    
    # 正则表达式模式用于匹配键名
    pattern = r'layers\.(\d+)\.(.*?)\.gatemtllora_B_w'
    
    # 提取所有 gatemtllora_B_w 权重
    for key, tensor in data.items():
        if 'gatemtllora_B_w' in key:
            # 使用正则表达式匹配层号和投影类型
            match = re.search(pattern, key)
            if match:
                layer_num = int(match.group(1))  # 转换为整数
                proj_type = match.group(2)
                
                # 存储权重张量，层号加1
                layer_key = f'layers.{layer_num + 1}'
                if layer_key not in extracted_weights:
                    extracted_weights[layer_key] = {}
                extracted_weights[layer_key][proj_type] = tensor
    
    # 为每个投影类型创建单独的文件
    for proj_type in set([pt for layer in extracted_weights.values() for pt in layer.keys()]):
        # 为每行创建两个文件：原始值和softmax值
        for row_idx in range(5):  # 假设每行有5个值
            # 原始值文件
            filename_orig = os.path.join(save_dir, f'{proj_type}_{row_idx}.txt')
            # softmax值文件
            filename_softmax = os.path.join(save_dir, f'{proj_type}_{row_idx}_softmax.txt')
            
            # 保存原始值
            with open(filename_orig, 'w') as f:
                # 按层号排序
                for layer_key in sorted(extracted_weights.keys(), 
                                       key=lambda x: int(x.split('.')[1])):
                    if proj_type in extracted_weights[layer_key]:
                        # 获取权重张量
                        weights = extracted_weights[layer_key][proj_type]
                        
                        # 检查行索引是否有效
                        if row_idx < weights.shape[0]:
                            # 获取指定行的权重值
                            row_weights = weights[row_idx]
                            
                            # 格式化行数据
                            row_values = []
                            for j in range(row_weights.shape[0]):
                                value = row_weights[j].item()
                                # 保留4位小数
                                row_values.append(f"{value:.4f}")
                            
                            # 写入注释行（层标识）
                            layer_num = layer_key.split('.')[1]
                            f.write(f"# count_for_decoder_layer {layer_num}\n")
                            
                            # 写入行数据
                            f.write(", ".join(row_values) + "\n\n")
            
            # 保存softmax值
            with open(filename_softmax, 'w') as f:
                # 按层号排序
                for layer_key in sorted(extracted_weights.keys(), 
                                       key=lambda x: int(x.split('.')[1])):
                    if proj_type in extracted_weights[layer_key]:
                        # 获取权重张量
                        weights = extracted_weights[layer_key][proj_type]
                        
                        # 检查行索引是否有效
                        if row_idx < weights.shape[0]:
                            # 获取指定行的权重值
                            row_weights = weights[row_idx]
                            
                            # 计算softmax
                            softmax_weights = F.softmax(row_weights, dim=0)
                            
                            # 格式化行数据
                            row_values = []
                            for j in range(softmax_weights.shape[0]):
                                value = softmax_weights[j].item()
                                # 保留6位小数（softmax值通常较小）
                                row_values.append(f"{value:.6f}")
                            
                            # 写入注释行（层标识）
                            layer_num = layer_key.split('.')[1]
                            f.write(f"# count_for_decoder_layer {layer_num}\n")
                            
                            # 写入行数据
                            f.write(", ".join(row_values) + "\n\n")
            
            print(f"已保存 {proj_type} 的第 {row_idx} 行权重到 {filename_orig} 和 {filename_softmax}")


# 示例使用
if __name__ == "__main__":
    # 假设您的数据已经加载到变量 `data` 中
    # data = {'base_model.model.language_model.model.layers.0.mlp.down_proj.gatemtllora_B_w': tensor(...), ...}
    
    # 提取并保存权重
    filename = "/home/wyz/projects/LLaMA-Factory/saves/qwen2_audio_instruct-7b/gatemtllora_5L_no_lambda_1ep/adapter_model.safetensors"
    data = safe_load_file(filename, device="cpu")
    extract_and_save_gatemtllora_B_w(data)