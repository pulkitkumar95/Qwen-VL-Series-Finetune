#!/bin/bash

# Example evaluation script for Qwen-VL models
# Usage: bash scripts/eval_sft.sh

# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="Qwen/Qwen3-VL-4B-Instruct"

export HF_HOME=/fs/cfar-projects/actionloc/hub_261202
export TRITON_CACHE_DIR=/scratch1/pulkit/cache
mkdir -p /scratch1/pulkit/cache

export PYTHONPATH=src:$PYTHONPATH

# Evaluation settings
BATCH_PER_DEVICE=1
OUTPUT_DIR="output/eval_results"

# If evaluating a LoRA model, uncomment and set the path:
# LORA_WEIGHT_PATH="output/test_train/checkpoint-1000"

python src/train/eval_sft.py \
    --model_id $MODEL_NAME \
    --eval_path /fs/vulcan-projects/motion_llm/pulkit/Flash-VStream/Flash-VStream-Qwen/data/llava-video-178k/sampled_test.json \
    --image_folder /fs/cfar-projects/actionloc/data/llava_video_9k \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size $BATCH_PER_DEVICE \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --compute_perplexity True \
    --save_predictions True \
    --max_new_tokens 512 \
    --do_sample False \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k 50 \
    --video_min_pixels $((64 * 32 * 32)) \
    --video_max_pixels $((64 * 32 * 32)) \
    --nframes 64 \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    $(if [ -n "$LORA_WEIGHT_PATH" ]; then echo "--lora_enable True --lora_weight_path $LORA_WEIGHT_PATH"; fi)

