from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
import torch

filename = "/home/wyz/projects/LLaMA-Factory/saves/qwen2_audio_instruct-7b/gatemtllora_5L_language_id_compress/adapter_model.safetensors"
adapters_weights = safe_load_file(filename, device="cpu")
print(adapters_weights)

# 文件路径
# file_path = "/home/wyz/projects/LLaMA-Factory/saves/qwen2_audio_instruct-7b/mtllora_az/adapter_model.safetensors"

# try:
#     # 安全打开文件（避免内存溢出）
#     with safe_open(file_path, framework="pt", device="cpu") as f:
#         # 获取所有张量键名
#         all_keys = f.keys()
#         print("文件中的张量键名:", all_keys)
#         if len(all_keys) > 0:
#             sample_key = all_keys[0]
#             tensor = f.get_tensor(sample_key)
#             print(f"\n示例张量 '{sample_key}' 的前10个元素值:")
#             print(tensor.flatten()[:10].numpy())  # 展平后取前10个值
#             sample_key = "base_model.model.language_model.model.layers.9.self_attn.v_proj.mtllora_B_0.weight"
#             tensor = f.get_tensor(sample_key)
#             print(f"\n示例张量 '{sample_key}' 的前10个元素值:")
#             print(tensor.flatten()[:10].numpy())  # 展平后取前10个值
#             sample_key = "base_model.model.language_model.model.layers.9.self_attn.v_proj.mtllora_B_w"
#             tensor = f.get_tensor(sample_key)
#             print(f"\n示例张量 '{sample_key}' 的前10个元素值:")
#             print(tensor.flatten()[:10].numpy())  # 展平后取前10个值
#             sample_key = "base_model.model.language_model.model.layers.9.self_attn.q_proj.mtllora_B_1"
#             tensor = f.get_tensor(sample_key)
#             print(f"\n示例张量 '{sample_key}' 的前10个元素值:")
#             print(tensor.flatten()[:10].numpy())  # 展平后取前10个值
#             sample_key = "model.language_model.model.layers.9.self_attn.v_proj.mtllora_B_0"
#             tensor = f.get_tensor(sample_key)
#             print(f"\n示例张量 '{sample_key}' 的前10个元素值:")
#             print(tensor.flatten()[:10].numpy())  # 展平后取前10个值
#             sample_key = "model.language_model.model.layers.9.self_attn.v_proj.mtllora_B_w"
#             tensor = f.get_tensor(sample_key)
#             print(f"\n示例张量 '{sample_key}' 的前10个元素值:")
#             print(tensor.flatten()[:10].numpy())  # 展平后取前10个值

# except Exception as e:
#     print(f"读取文件失败: {str(e)}")



