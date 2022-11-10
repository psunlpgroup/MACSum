#! /bin/bash

export WANDB_PROJECT=mac_dial
export OMP_NUM_THREADS=1
# export HF_DATASETS_CACHE="/data/yfz5488/.hf_cache"

learning_rate=3e-5
config_type=async # select from [prompt, vanilla, freeze, scratch, combine, async]
cur_date="$(date +%Y-%m-%d)"

export CUDA_VISIBLE_DEVICES=0
export RUN_NAME=BART-large_qmsum_${config_type}_${cur_date}

#python -m torch.distributed.launch --nproc_per_node 2 --master_port 1234 train.py \
python train.py \
 --seed 42 \
 --cfg PREFIX_TUNING/BART_mac_qmsum_${config_type}.cfg \
 --run_name ${RUN_NAME} \
 --logging_strategy steps\
 --logging_first_step true \
 --logging_steps 4 \
 --evaluation_strategy steps \
 --eval_steps 150 \
 --greater_is_better false \
 --save_strategy steps \
 --save_steps 150 \
 --save_total_limit 1 \
 --load_best_model_at_end \
 --gradient_accumulation_steps 4 \
 --num_train_epochs 300 \
 --learning_rate ${learning_rate} \
 --do_train --do_eval --do_predict\
 --predict_with_generate \
 --output_dir output/${RUN_NAME} \
 --overwrite_output_dir \
 --per_device_train_batch_size 6 \
 --per_device_eval_batch_size 8 \
 --generation_num_beams 4 \
 --generation_max_length 400 \
 --input_max_length 1024 \
 --ddp_find_unused_parameters true \
 --warmup_steps 300 \
 --adam_epsilon 1e-8