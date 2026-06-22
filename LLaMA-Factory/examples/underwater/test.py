import json
from sentence_transformers import SentenceTransformer as SBert
from sentence_transformers.util import cos_sim

# 1. 解析 JSONL 文件并提取数据
jsonl_file_path = '/home/wyz/projects/LLaMA-Factory/saves/underwater_captions/dpo_caption_lora_caption_train_caption_train_sameprompt/preprocessed_res.jsonl'  # 请替换为你的实际文件路径
print(jsonl_file_path)
sentences1 = []
sentences2 = []

with open(jsonl_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line.strip())
        sentences1.append(data['predict'])
        sentences2.append(data['label'])

# 2. 加载模型 
# 注意：你提供的数据是中文，建议使用支持中文的模型，如 'paraphrase-multilingual-MiniLM-L12-v2' 
# 或 'shibing624/text2vec-base-chinese'。如果坚持用原来的英文模型，结果可能不准。
print("正在加载模型...")

model = SBert("google/embeddinggemma-300m")
# model = SBert('roberta-large-nli-stsb-mean-tokens') 

# 3. 对句子进行编码
print("正在进行文本编码...")
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

# 4. 计算余弦相似度矩阵
print("正在计算相似度...")
cosine_scores = cos_sim(embeddings1, embeddings2)

# 5. 提取每对 (predict, label) 的得分并计算平均值
# cosine_scores 是 N x N 矩阵，使用 .diag() 提取对角线（即对应行的数据）
pair_scores = cosine_scores.diag()

# 计算平均分
average_score = pair_scores.mean().item()

print("\n--- 相似度计算结果 ---")
for i, score in enumerate(pair_scores):
    print(f"第 {i+1} 条数据相似度得分: {score.item():.4f}")

print("-" * 25)
print(f"【总平均相似度得分】: {average_score:.4f}")