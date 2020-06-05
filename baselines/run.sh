#!/usr/bin/env bash
REPO_PATH=$1
export CUDA_VISIBLE_DEVICES=0,1,2
python -m torch.distributed.launch evidence_deploy.py \
    --data_dir $REPO_PATH/data/boolq \
    --output_dir $REPO_PATH/out/boolq_gold \
    --model_params $REPO_PATH/params/bert_truefalse.json \
    --model_type bert \
    --overwrite_output_dir \
	--do_train \
	--do_eval \
	--eval_split val \
	--max_seq_length 512 \
	--max_query_length 24 \
	--local_rank -1 \
	--num_train_epochs 20 \
	--wait_step 10 \
    --evaluate_during_training \
    --logging_steps 500 \
	--save_steps 10000 \
	--gold_evidence \
