#!/usr/bin/env bash
REPO_PATH=$1
export CUDA_VISIBLE_DEVICES=2
python -m torch.distributed.launch ib_train_sentence.py \
    --data_dir $REPO_PATH/data/multirc \
    --output_dir $REPO_PATH/out/multirc_ib_L0_1e-6 \
    --model_params $REPO_PATH/params/gated_truefalse.json \
    --model_type distilbert_hard_gated_sent \
    --overwrite_output_dir \
	--do_train \
    --do_eval \
    --eval_split val \
    --max_seq_length 512 \
    --max_query_length 32 \
    --num_train_epochs 20 \
    --max_num_sentences 15 \
    --wait_step 10 \
    --evaluate_during_training \
    --logging_steps 500 \
    --save_steps 5000 \
    --tf_summary \
    --local_rank -1 \
