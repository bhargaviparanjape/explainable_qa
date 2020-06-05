#!/usr/bin/env bash
REPO_PATH=$1
export CUDA_VISIBLE_DEVICES=2
python -m torch.distributed.launch ib_train_sentence.py \
    --data_dir $REPO_PATH/data/beer \
    --output_dir $REPO_PATH/beer_KL_1.0_zhat \
    --model_params $REPO_PATH/params/gated_regression.json \
    --model_type distilbert_gated_sent \
    --overwrite_output_dir \
    --do_eval \
    --eval_split val \
    --max_seq_length 512 \
    --max_query_length 4 \
    --local_rank -1 \
    --num_train_epochs 20 \
    --max_num_sentences 10 \
    --wait_step 10 \
    --evaluate_during_training \
    --logging_steps 500 \
    --save_steps 5000 \
    --tf_summary \
    --eval_split test \
