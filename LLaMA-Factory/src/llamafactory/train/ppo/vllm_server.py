import os
from vllm import LLM, SamplingParams

# 只让 vLLM 看到 GPU1 (3090)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

llm = LLM(
    model="/nfs4/wyz/merged_qwen2audio/lora_5l_merged",
    max_model_len = 2048,
    trust_remote_code=True,
    gpu_memory_utilization=0.8,
    dtype="bfloat16",
    limit_mm_per_prompt={"audio": 1},
)

sampling_params = SamplingParams(
    max_tokens=512,
    temperature=0.2,
    top_p=0.95
)


def generate_batch(prompts, audio_features):
    outputs = llm.generate(
        [
            {
                "prompt": p,
                "multi_modal_data": {
                    "audio": (a, 16000)
                }
            }
            for p, a in zip(prompts, audio_features)
        ],
        sampling_params
    )

    return [o.text for o in outputs]
