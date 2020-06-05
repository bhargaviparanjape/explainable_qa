#!/usr/bin/env bash
REPO_PATH=$1
export CUDA_VISIBLE_DEVICES=2
python -m torch.distributed.launch ib_train.py \
    --data_dir $REPO_PATH/data/movies \
    --output_dir $REPO_PATH/out/movies_ib_beta_100 \
    --model_params $REPO_PATH/params/ib_sentiment.json \
    --model_type distilbert \
    --overwrite_output_dir \
	--overwrite_cache \
	--do_train \
	--do_eval \
	--eval_split val \
	--max_seq_length 512 \
	--max_query_length 8 \
	--local_rank -1 \
	--num_train_epochs 20 \
	--wait_step 10 \
    --evaluate_during_training \
    --logging_steps 40 \
	--save_steps 1000 \
	--tf_summary \
