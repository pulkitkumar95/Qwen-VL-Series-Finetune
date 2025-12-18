#!/bin/bash

# Evaluation script for the model trained with finetune_video_nexus_llava_video.sh
# This evaluates on the training data

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

# export PATH="$PATH:/scratch/zt1/project/abhinav2-prj/user/pulkit/python_packages_new_img/bin"
# export PYTHONUSERBASE=/scratch/zt1/project/abhinav2-prj/user/pulkit/python_packages_new_img

MODEL_NAME="Qwen/Qwen3-VL-4B-Instruct"

export HF_HOME=/fs/cfar-projects/actionloc
export TRITON_CACHE_DIR=/scratch1/pulkit/cache
mkdir -p /scratch1/pulkit/cache

export PYTHONPATH=src:$PYTHONPATH

# Path to the checkpoint directory (output_dir from training)
OUTPUT_DIR="output/qwen3vl4b_1fps_favor_full_text_base"

# Training data paths (same as training script)
EVAL_PATH="/fs/vulcan-projects/motion_llm/pulkit/dataset_utils/final_jsons/favor_llava_style_full_text.json"
IMAGE_FOLDER="/fs/cfar-projects/actionloc/hub/datasets--zl2048--FAVOR/snapshots/2a78953831e41ebd046a7e6e55eb3b6c28f61e9b/videos/FAVOR-Bench"

# Run evaluation
# Note: You can use deepspeed for distributed evaluation, or just python for single GPU
deepspeed --master_port 2952 src/train/eval_sft.py \
    --use_liger False \
    --model_id $MODEL_NAME \
    --eval_path $EVAL_PATH \
    --image_folder $IMAGE_FOLDER \
    --output_dir $OUTPUT_DIR \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --per_device_eval_batch_size 1 \
    --video_max_pixels $((360 * 420)) \
    --fps 1 \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --batch_eval_metrics True \
    --include_for_metrics inputs \
    --eval_dataset_name favor

# For distributed evaluation with deepspeed, use:
# deepspeed --master_port 2952 src/train/eval_sft.py \
#     --use_liger True \
#     --deepspeed scripts/zero3.json \
#     --model_id $MODEL_NAME \
#     --eval_path $EVAL_PATH \
#     --image_folder $IMAGE_FOLDER \
#     --output_dir $OUTPUT_DIR \
#     --bf16 True \
#     --fp16 False \
#     --disable_flash_attn2 False \
#     --per_device_eval_batch_size 2 \
#     --video_max_pixels $((360 * 420)) \
#     --nframes 64 \
#     --lazy_preprocess True \
#     --dataloader_num_workers 0


