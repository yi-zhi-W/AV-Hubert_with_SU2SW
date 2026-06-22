#!/bin/bash

# 多适配器多数据集评估脚本（使用两个数组实现）
# 文件名: run_full_evaluation_dual_arrays.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 定义要评估的适配器路径数组
adapters=(
    "/home/wyz/projects/LLaMA-Factory/saves/ppo/0226/train_ppo_uyghur_ppo_whiten_rewards"  # path1
    # "/home/wyz/projects/LLaMA-Factory/saves/ppo/0114_real_bpe/ppo_1000_samewith0106_ug_train_id_5000"  # path2
    # "/home/wyz/projects/LLaMA-Factory/saves/ppo/0112/ug_train_id_5000_kl_1"  # path3
)

# 定义对应的微调类型数组
finetuning_types=(
    "lora"  # 对应第一个适配器
    # "lora"  # 对应第二个适配器
    # "lora"  # 对应第三个适配器
)

# 检查数组长度是否匹配
if [ ${#adapters[@]} -ne ${#finetuning_types[@]} ]; then
    echo "❌ 错误：适配器路径数组和微调类型数组长度不匹配！"
    echo "适配器数量: ${#adapters[@]}, 微调类型数量: ${#finetuning_types[@]}"
    exit 1
fi

# 定义要评估的数据集列表
datasets=("ug_test_id")
# datasets=("tr_test_id")

# 获取当前时间戳用于日志记录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="evaluation_log_${TIMESTAMP}.txt"

# 开始日志记录
{
echo "============================================================"
echo "🚀 开始多适配器多数据集评估任务（双数组实现）"
echo "📅 开始时间: $(date)"
echo "💻 GPU: $CUDA_VISIBLE_DEVICES"
echo "📁 适配器数量: ${#adapters[@]}"
echo "📊 数据集数量: ${#datasets[@]}"
echo "🔧 微调类型映射:"
for i in "${!adapters[@]}"; do
    adapter_name=$(basename "${adapters[$i]}")
    echo "    - $adapter_name → ${finetuning_types[$i]}"
done
echo "📝 日志文件: $LOG_FILE"
echo "============================================================"

# 计数器
total_tasks=$(( ${#adapters[@]} * ${#datasets[@]} ))
completed_tasks=0

# 外层循环：适配器路径
for i in "${!adapters[@]}"; do
    adapter_path="${adapters[$i]}"
    finetuning_type="${finetuning_types[$i]}"
    
    # 提取适配器名称用于输出目录
    adapter_name=$(basename "$adapter_path")
    
    echo ""
    echo "****************************************************************"
    echo "🔧 开始处理适配器: $adapter_name (索引: $i)"
    echo "📂 路径: $adapter_path"
    echo "⚙️ 微调类型: $finetuning_type"
    echo "****************************************************************"
    
    # 内层循环：数据集
    for dataset in "${datasets[@]}"; do
        ((completed_tasks++))
        echo ""
        echo "============================================================"
        echo "🔁 任务进度: $completed_tasks/$total_tasks"
        echo "📊 开始评估数据集: $dataset"
        echo "🔄 适配器: $adapter_name"
        echo "⚙️ 微调类型: $finetuning_type"
        echo "============================================================"
        
        # 设置基于适配器、微调类型和数据集的输出目录
        output_dir="/home/wyz/projects/LLaMA-Factory/saves/ppo_results/0226/${adapter_name}_${finetuning_type}_${dataset}_do_sample"
        
        echo "📂 输出目录: $output_dir"
        echo "⏳ 开始时间: $(date +"%T")"
        
        # 执行模型评估命令（使用对应的微调类型）
        llamafactory-cli train \
            --stage sft \
            --do_predict \
            --model_name_or_path "/nfs4/wyz/merged_qwen2audio/lora_5l_merged" \
            --adapter_name_or_path "$adapter_path" \
            --eval_dataset "$dataset" \
            --dataset_dir "./data" \
            --template "qwen2_audio" \
            --finetuning_type "$finetuning_type" \
            --output_dir "$output_dir" \
            --overwrite_cache \
            --overwrite_output_dir \
            --cutoff_len 1024 \
            --preprocessing_num_workers 4 \
            --per_device_eval_batch_size 1 \
            --predict_with_generate \
            --max_samples 1000
        
        # 检查命令执行状态
        if [ $? -eq 0 ]; then
            echo "✅ 评估成功! 适配器: $adapter_name, 类型: $finetuning_type, 数据集: $dataset"
        else
            echo "❌ 评估失败! 适配器: $adapter_name, 类型: $finetuning_type, 数据集: $dataset"
            # 可以选择继续执行下一个任务或退出
            # exit 1  # 取消注释此行会在失败时退出整个脚本
        fi
        
        echo "⏱️ 结束时间: $(date +"%T")"
        echo "============================================================"
        sleep 1  # 添加短暂延迟
    done
done

echo ""
echo "============================================================"
echo "🎉 所有评估任务已完成!"
echo "📅 结束时间: $(date)"
echo "✅ 成功任务: $completed_tasks/$total_tasks"
echo "============================================================"

} | tee "$LOG_FILE"  # 同时输出到终端和日志文件

# 添加桌面通知（如果可用）
if command -v notify-send &> /dev/null; then
    notify-send "评估任务完成" "已完成 $completed_tasks/$total_tasks 个评估任务\n适配器数量: ${#adapters[@]}"
fi

echo "📝 完整日志已保存到: $LOG_FILE"