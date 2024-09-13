#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

#MODEL_VERSION=vicuna-v1-3-7b
#MODEL_VERSION=llama-2-7b-chat

########### DO NOT CHANGE ###########
########### USE THIS FOR BOTH ###########
PROMPT_VERSION=plain
########### DO NOT CHANGE ###########


# profile
PROFILE_ARGS=
#PROFILE_ARGS="--profiling_warmup_steps 10 --profiling_steps 2 --no_profiling_record_shapes"

MAX_STEPS=
#MAX_STEPS="--max_steps 10"

    #--vision_tower openai/clip-vit-large-patch14 \
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /mnt/ctrl/disk1/personal/kurt/llava/vicuna-13b-v1.5 \
    --version $PROMPT_VERSION \
    --data_path /mnt/ctrl/disk2/mingzhi/llava_dataset/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /mnt/ctrl/disk2/mingzhi/llava_dataset/images \
    --vision_tower /mnt/ctrl/disk1/personal/kurt/llava/clip-vit-large-patch14-336 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --gaudi_config_name ./scripts/gaudi_config.json \
    ${MAX_STEPS} \
    ${PROFILE_ARGS} \
    --throughput_warmup_steps 3 \
    --adjust_throughput \
    --use_habana \
    --use_lazy_mode
