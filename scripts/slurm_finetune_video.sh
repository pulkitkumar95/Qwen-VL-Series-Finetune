#!/bin/bash
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --partition=vulcan-scavenger
#SBATCH --mem=255g
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=vulcan46
#SBATCH --job-name=ft_qwen
#SBATCH --output=slurm_out/train_%j.out
#SBATCH --error=slurm_out/train_%j.err

# Evaluation script for the model trained with finetune_video_nexus_llava_video.sh
# This evaluates on the training data

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

# export PATH="$PATH:/scratch/zt1/project/abhinav2-prj/user/pulkit/python_packages_new_img/bin"
# export PYTHONUSERBASE=/scratch/zt1/project/abhinav2-prj/user/pulkit/python_packages_new_img
source /etc/profile.d/modules.sh
module load cuda/12.8.1 
cd /fs/vulcan-projects/motion_llm/pulkit/Qwen-VL-Series-Finetune
source /fs/vulcan-projects/motion_llm/miniconda/bin/activate
conda activate qwen3

# MODEL_NAME="Qwen/Qwen3-VL-4B-Instruct"
export HF_HOME=/fs/cfar-projects/actionloc/hub_261202
export TRITON_CACHE_DIR=/scratch1/pulkit/cache
mkdir -p /scratch1/pulkit/cache

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=2
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# If your dataset is mixed with images and videos, you need to use zero2.
# If you want to set the min pixels and max pixels for Qwen3-VL, You should set as (N * 32 * 32)

deepspeed src/train/train_sft.py \
    --use_liger_kernel False \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /fs/vulcan-projects/motion_llm/pulkit/Flash-VStream/Flash-VStream-Qwen/data/llava-video-178k/trainset_9k_video.json \
    --image_folder /fs/cfar-projects/actionloc/data/llava_video_9k \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/qwen_25vl_1fps_llava_9k \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_max_pixels $((360 * 420)) \
    --fps 1 \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --dataloader_num_workers 4