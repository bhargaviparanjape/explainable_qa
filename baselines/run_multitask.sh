#!/usr/bin/env bash
REPO_PATH=$1
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch evidence_multitask.py \
    --data_dir $REPO_PATH/data/movies \
    --output_dir $REPO_PATH/out/movies_spanbert_multitask \
    --model_params $REPO_PATH/params/bert_sentiment.json \
    --model_type multitask_bert \
    --overwrite_output_dir \
	--do_eval \
	--do_train \
	--eval_split val \
	--max_seq_length 512 \
	--max_query_length 8 \
	--local_rank -1 \
	--num_train_epochs 20 \
	--wait_step 10 \
    --evaluate_during_training \
    --logging_steps 40 \
	--save_steps 100 \
