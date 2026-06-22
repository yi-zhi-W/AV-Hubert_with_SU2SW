import librosa
from vllm import LLM, SamplingParams

# 1. 初始化 vLLM 推理引擎
# (首次运行会自动从 HuggingFace/ModelScope 下载权重)
llm = LLM(model="/nfs4/wyz/merged_qwen2audio/lora_5l_merged", trust_remote_code=True)

# 2. 读取音频文件和采样率
audio_path = "/home/wyz/data/UG/clips/common_voice_ug_36990048.mp3"
# sr=None 代表保留音频的原始采样率，vLLM 内部的特征提取器会自动将其重采样到所需的 16kHz
audio_data, sample_rate = librosa.load(audio_path, sr=None)

# 3. 构建包含特殊音频 token 的 prompt
# 注意：占位符必须与你要分析的音频严格对应
prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>\nGenerate the caption in Uyghur:"

# 4. 配置生成采样的参数
sampling_params = SamplingParams(temperature=0.2, max_tokens=256)

# 5. 将文本和多模态音频数据组合，送入模型推理
outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {
        "audio": (audio_data, sample_rate)  # 严格遵守输入格式：(NumPy数组, 采样率)
    }
}, sampling_params=sampling_params)

# 6. 提取并打印生成的回复
for o in outputs:
    print(o.outputs[0].text)