#/bin/bash

# $2: gpu number

python3 vipaint.py \
    --inpaint_config=configs/inpainting/inpaint_config.yaml \
    --gpu=$2 \
    --save_dir=$3;