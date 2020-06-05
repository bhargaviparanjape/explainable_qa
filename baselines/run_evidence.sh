#!/usr/bin/env bash
REPO_PATH=$1
export CUDA_VISIBLE_DEVICES=2
python -m torch.distributed.launch evidence_train.py \
    --data_dir $REPO_PATH/data/fever \
    --output_dir $REPO_PATH/out/fever_bert_trained_evidence \
    --model_params $REPO_PATH/params/bert_verification.json \
    --model_type bert \
    --overwrite_output_dir \
	--do_eval \
	--eval_split train \
	--max_seq_length 512 \
	--max_query_length 32 \
	--local_rank -1 \
	--num_train_epochs 20 \
	--wait_step 10 \
    --evaluate_during_training \
    --logging_steps 500 \
	--save_steps 1000 \
	--predicted_evidence_file $REPO_PATH/out/fever_bert_evidence.pkl \
