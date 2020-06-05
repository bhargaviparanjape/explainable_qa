#!/usr/bin/env bash
REPO_PATH=$1
export CUDA_VISIBLE_DEVICES=2
python -m torch.distributed.launch ib_train_sentence.py \
    --data_dir $REPO_PATH/data/boolq_truncated \
    --output_dir $REPO_PATH/out/boolq_KL_0.1 \
    --model_params $REPO_PATH/params/gated_bool.json \
    --model_type distilbert_gated_sent \
    --overwrite_output_dir \
	--do_train \
    --do_eval \
    --eval_split val \
    --max_seq_length 512 \
    --max_query_length 24 \
    --num_train_epochs 20 \
    --max_num_sentences 25 \
    --wait_step 10 \
    --evaluate_during_training \
    --logging_steps 100 \
    --save_steps 5000 \
    --tf_summary \
    --local_rank -1 \
