#!/bin/bash

source /opt/conda/bin/activate artml1

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="/home/jovyan/working/art_ml_project/art_and_ml_2024/project1/training_data"
export OUTPUT_DIR="/home/jovyan/working/art_ml_project/art_and_ml_2024/project1/save_model/steps1"

accelerate launch /home/jovyan/working/art_ml_project/project1/diffusers/examples/text_to_image/train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=3 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=1 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR}