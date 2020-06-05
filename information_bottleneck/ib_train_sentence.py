import sys; sys.path.insert(0, "..")
import argparse
import json
import logging
import random
import os
import glob
import pdb
import shutil
from itertools import chain
from typing import Set
from scipy.stats import entropy


import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter
from sklearn.metrics import (accuracy_score, classification_report,
                            precision_recall_curve, roc_auc_score,
                             mean_squared_error, auc, precision_score,
                             recall_score, f1_score)

from eraser.rationale_benchmark.utils import (
    write_jsonl,
    load_datasets,
    load_documents,
    annotations_from_jsonl,
    intern_documents,
    intern_annotations
)

import metrics

from ib_utils import read_examples, convert_examples_to_sentence_features

from transformers import (WEIGHTS_NAME, BertTokenizer, RobertaTokenizer, BertForSequenceClassification, \
                          BertConfig, RobertaConfig, \
                          DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer, \
                          RobertaForSequenceClassification, BertForMultipleChoice, RobertaForMultipleChoice,
                          AdamW, get_linear_schedule_with_warmup)

from bert_explainer import DistilBertExplainer, BertExplainer,  DistilBertSentenceExplainer
from gated_explainer import (DistilBertGatedSentenceExplainer, DistilBertHardGatedSentenceExplainer, \
                             BertGatedSentenceExplainer, BertHardGatedSentenceExplainer, \
                             RobertaGatedSentenceExplainer, RobertaHardGatedSentenceExplainer)

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
# let's make this more or less deterministic (not resistent to restarts)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, RobertaConfig)), ())


MODEL_CLASSES = {
    'bert': (BertConfig, BertExplainer, BertTokenizer),
    'roberta' : (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertExplainer, DistilBertTokenizer),
    'distilbert_sent': (DistilBertConfig, DistilBertSentenceExplainer, DistilBertTokenizer),

    'distilbert_gated_sent': (DistilBertConfig, DistilBertGatedSentenceExplainer, DistilBertTokenizer),
    'distilbert_hard_gated_sent': (DistilBertConfig, DistilBertHardGatedSentenceExplainer, DistilBertTokenizer),

    'bert_gated_sent' : (BertConfig, BertGatedSentenceExplainer, BertTokenizer),
    'bert_hard_gated_sent' : (BertConfig, BertHardGatedSentenceExplainer, BertTokenizer),

    'roberta_gated_sent' : (RobertaConfig, RobertaGatedSentenceExplainer, RobertaTokenizer),
    'roberta_hard_gated_sent' : (RobertaConfig, RobertaHardGatedSentenceExplainer, RobertaTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def load_and_cache_examples(args, model_params, tokenizer, evaluate=False, split="train", output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # only load one split
    input_file = os.path.join(args.data_dir, split)
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}_sent'.format(
        split,
        list(filter(None, model_params["tokenizer_name"].split('/'))).pop(),
        str(args.max_seq_length)))
    if args.gold_evidence:
        cached_features_file += "_goldevidence"

    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        dataset = annotations_from_jsonl(os.path.join(args.data_dir, split + ".jsonl"))

        docids = set(e.docid for e in
                     chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(dataset)))))
        if "boolq" in args.data_dir or "evidence_inference" in args.data_dir:
            docids  = set([ex.docids[0] for ex in dataset])
        if "beer" in args.data_dir or "imdb" in args.data_dir:
            docids= set([ex.annotation_id for ex in dataset])
        if "movies" in args.data_dir:
            docids.add("posR_161.txt")
        documents = load_documents(args.data_dir, docids)

        examples = read_examples(args, model_params, dataset, documents, split)

        features = convert_examples_to_sentence_features(args, model_params, examples=examples,
                                               tokenizer=tokenizer,
                                               max_seq_length=args.max_seq_length,
                                               max_query_length=args.max_query_length,
                                               is_training=not evaluate)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    # Tensorize all features
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.long)
    all_unique_ids = torch.tensor([f.unique_id for f in features], dtype=torch.float)
    all_sentence_starts = torch.tensor([f.sentence_starts for f in features], dtype=torch.long)
    all_sentence_ends = torch.tensor([f.sentence_starts for f in features], dtype=torch.long)
    all_sentence_mask = torch.tensor([f.sentence_mask for f in features], dtype=torch.long)
    all_evidence_labels = torch.tensor([f.evidence_label for f in features], dtype=torch.long)

    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        tensorized_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask,
                                all_unique_ids, all_sentence_starts, all_sentence_ends,
                                all_sentence_mask, all_evidence_labels)
    else:
        if isinstance(features[0].class_label,float):
            all_class_labels = torch.tensor([f.class_label for f in features], dtype=torch.float)
        else:
            all_class_labels = torch.tensor([f.class_label for f in features], dtype=torch.long)
        # required if providing some additional supervision
        tensorized_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_class_labels, all_cls_index, all_p_mask,
                                all_unique_ids, all_sentence_starts, all_sentence_ends,
                                all_sentence_mask, all_evidence_labels)


    if output_examples:
        return tensorized_dataset, examples, features
    return tensorized_dataset, features

def train(args, model_params, train_dataset, model, tokenizer):
    if args.local_rank in [-1, 0] and args.tf_summary:
        tb_writer = SummaryWriter("runs/" + os.path.basename(args.output_dir))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    # TODO: Addiitonal parameters require more learning rate compared to BERT model
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    if args.evaluate_during_training:
        dataset, eval_features = load_and_cache_examples(args, model_params, tokenizer, evaluate=True, split=args.eval_split, output_examples=False)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Keep all eraser annotations at  the ready to go
    annotations = annotations_from_jsonl(os.path.join(args.data_dir, args.eval_split + '.jsonl'))
    true_rationales = list(chain.from_iterable(metrics.Rationale.from_annotation(ann) for ann in annotations))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    class_tr_loss, class_logging_loss = 0.0, 0.0
    info_tr_loss, info_logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    if "beer" in args.data_dir:
        best_f1 = (1e6, 1e6)
    else:
        best_f1 = (-1, -1) # Classfication F1 and accuracy
    wait_step = 0
    stop_training = False
    metric_name = "F1"
    epoch = 0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':       batch[0],
                      'attention_mask':  batch[1],
                      # 'token_type_ids':  None if args.model_type == 'xlm' or args.model_type == 'roberta' else batch[2],
                      'labels': batch[3],
                      'p_mask': batch[5]}
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask':       batch[5]})
            inputs.update({
                "sentence_starts" : batch[7],
                "sentence_ends" : batch[8],
                "sentence_mask": batch[9],
                "evidence_labels" : batch[10],
            })
            if args.anneal and (global_step + 1) % 500 == 0:
                model.module.beta = model.module.beta/2
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            class_loss = outputs[1]
            info_loss = outputs[2]
            lambda0 = outputs[-1].mean().item()
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
                class_loss = class_loss.mean()
                info_loss = info_loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                class_loss = class_loss / args.gradient_accumulation_steps
                info_loss = info_loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            class_tr_loss += class_loss.item()
            info_tr_loss += info_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = predict(args, model_params, model, tokenizer, eval_features, eval_dataloader, true_rationales, tb_writer, global_step)
                        #for key, value in results.items():
                        #    tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        if ("beer" in args.data_dir and best_f1 > results) or ("beer" not in args.data_dir and best_f1 < results):
                            logger.info("Saving model with best %s: %.2f (Acc %.2f) -> %.2f (Acc %.2f) on epoch=%d" % \
                                        (metric_name, best_f1[0] * 100, best_f1[1] * 100, results[0] * 100, results[1] * 100,
                                         epoch))
                            output_dir = os.path.join(args.output_dir, 'best_model')
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                            best_f1 = results
                            wait_step = 0
                            stop_training = False
                        else:
                            wait_step += 1
                            if wait_step == args.wait_step:
                                logger.info("Loosing Patience")
                                stop_training = True
                if args.tf_summary  and global_step % 20 == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar('class_loss', (class_tr_loss - class_logging_loss) / 20, global_step)
                    tb_writer.add_scalar('info_loss', (info_tr_loss - info_logging_loss) / 20, global_step)
                    tb_writer.add_scalar('lambda0', lambda0, global_step)
                    # logger.info("Value of pi %.4f", lambda0)
                    logging_loss = tr_loss
                    class_logging_loss = class_tr_loss
                    info_logging_loss = info_tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if stop_training or args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if stop_training or args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        epoch += 1
    if args.local_rank in [-1, 0] and args.tf_summary:
        tb_writer.close()

    return global_step, tr_loss / global_step

def predict(args, model_params, model, tokenizer, eval_features, eval_dataloader, true_rationales, tb_writer, global_step):
    all_results = []
    all_targets = []
    # all_rationale_targets = []
    all_rationale_results = []
    all_results_dictionary = {}
    results = None
    class_interner = dict((y, x) for (x, y) in enumerate(model_params['classes']))
    class_labels = [k for k, v in sorted(class_interner.items(), key=lambda x: x[1])]
    average_nnz  = 0.0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      # 'token_type_ids': None if args.model_type == 'xlm' or args.model_type == 'roberta' else batch[2]
                      # XLM don't use segment_ids
                      'p_mask': batch[5]
                      }
            example_indices = batch[3]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask': batch[5]})
            inputs.update({
                "sentence_starts" : batch[7],
                "sentence_ends" : batch[8],
                "sentence_mask": batch[9],
                "evidence_labels" : batch[10],
            })
            inputs.update({"evaluate": True})
            outputs = model(**inputs)
            logits = outputs[0]
            nnz = torch.mean(outputs[-1]).item()
            average_nnz += nnz
            hard_rationale_pred = outputs[1]
            if logits.size(-1) > 1:
                hard_preds = to_list(torch.argmax(logits.float(), dim=-1))
            else:
                hard_preds = to_list(logits.squeeze(-1))
            # all_rationale_results.extend(hard_rationale_pred)
            all_results.extend(hard_preds)
        for i, example_index in enumerate(example_indices):
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_targets.extend([eval_feature.class_label])

            # # TODO: Only include unmasked entries since you are counting too many tokens other wise
            # hard_rationale_binary = np.zeros(len(eval_feature.evidence_label))
            # hard_rationale_binary[np.array(to_list(hard_rationale_pred[i]))] = 1
            # all_rationale_results.extend(hard_rationale_binary[:sum(eval_feature.sentence_mask)])
            # all_rationale_targets.extend(eval_feature.evidence_label[:sum(eval_feature.sentence_mask)])
            # Run a tiny loop to also fill up the predicted ratioanales output

            # based on indices selected; get corresponding start and end ; add those tokens to the Rationale list
            key = (eval_feature.annotation_id, eval_feature.doc_id)
            last_sentence = sum(eval_feature.sentence_mask)
            for sentence_no, sentence in enumerate(hard_rationale_pred[i]):
                # batch_index = sentence[0].item()
                # if batch_index != i:
                #     continue
                sentence = sentence.item()
                # sentence chosen is one outside the mask (especially for fever where you are forcing k selections from a 1 hot vector
                if sentence_no >= last_sentence or sentence == -1:
                    continue
                start_token = eval_feature.sentence_starts[sentence_no]
                end_token = eval_feature.sentence_ends[sentence_no]
                # end token is already exclusive, so even it does point to the SEP in the end , so will the gold
                # but they point to tokens and have to be mapped back
                if start_token not in eval_feature.token_to_orig_map:
                    # pdb.set_trace()
                    continue
                orig_start_token = eval_feature.token_to_orig_map[start_token]
                if eval_feature.tokens[end_token] in ["[SEP]", "</s>"]:
                    # last sentence, then end_token points to SEP, we want to get it to point to last word
                    end_token -= 1
                if end_token not in eval_feature.token_to_orig_map:
                    # pdb.set_trace()
                    continue
                orig_end_token = eval_feature.token_to_orig_map[end_token] + 1 # +1 since it needs to be exclusive

                all_rationale_results.append(metrics.Rationale(ann_id=key[0],
                                                               docid=key[1],
                                                               start_token=orig_start_token,
                                                               end_token=orig_end_token))

            all_results_dictionary[key] = hard_preds[i]
    evidence_iou = metrics.partial_match_score(true_rationales, all_rationale_results, [0.1, 0.5, 0.9])
    hard_rationale_metrics = metrics.score_hard_rationale_predictions(true_rationales, all_rationale_results)
    logger.info("Average Non-zero entries: %.4f" % (average_nnz/len(eval_dataloader)))
    logger.info("Rationale Partial score metrics:")
    logger.info(evidence_iou)
    tb_writer.add_scalar('rationale_f1', evidence_iou[0]['macro']['f1'], global_step)
    logger.info("Rationale hard metrics:")
    logger.info(hard_rationale_metrics)
    if isinstance(all_targets[0], float):
        mse = mean_squared_error(all_targets, all_results)
        results=(mse, mse)
    else:
        results = (classification_report(all_targets, all_results, target_names=class_labels, output_dict=True)['weighted avg']['f1-score'], accuracy_score(all_targets, all_results))
    tb_writer.add_scalar('task_perf', results[0], global_step)
    tb_writer.add_scalar('sparsity_rat', average_nnz/len(eval_dataloader), global_step)
    return results

def evaluate(args, model_params, model, tokenizer, prefix="", output_examples=False, split="val"):

    if output_examples:
        dataset, examples, features = load_and_cache_examples(args, model_params, tokenizer, evaluate=True, split=split, output_examples=output_examples)
    else:
        dataset, features = load_and_cache_examples(args, model_params, tokenizer, evaluate=True, split=split, output_examples=output_examples)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Keep all eraser annotations at  the ready to go
    annotations = annotations_from_jsonl(os.path.join(args.data_dir, split + '.jsonl'))
    true_rationales = list(chain.from_iterable(metrics.Rationale.from_annotation(ann) for ann in annotations))

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    all_targets = []
    # all_rationale_targets = []
    all_rationale_results = []
    all_results_dictionary = {}
    results = []
    class_interner = dict((y, x) for (x, y) in enumerate(model_params['classes']))
    class_labels = [k for k, v in sorted(class_interner.items(), key=lambda x: x[1])]

    # Faithfulness evaluation
    all_full_context_logits = []
    all_complement_context_logits = []
    all_rationale_context_logits = []

    # AUPRC for soft scoring
    soft_sentence_scores = []
    pred_sentence_scores = []
    gold_sentence_scores = []
    all_nnz = []
    task_performances = []
    evidence_performances = []
    # Do multiple evaluations for sampling based evaluation;
    if model_params["sampled_eval"]:
        args.num_evaluations = 10

    for run_no in range(args.num_evaluations):
        average_nnz  = 0.0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          # 'token_type_ids': None if args.model_type == 'xlm' or args.model_type == 'roberta' else batch[2]  # XLM don't use segment_ids
                          'p_mask': batch[5]
                          }
                example_indices = batch[3]
                if args.model_type in ['xlnet', 'xlm']:
                    inputs.update({'cls_index': batch[4],
                                   'p_mask':    batch[5]})
                inputs.update({
                    "sentence_starts" : batch[7],
                    "sentence_ends" : batch[8],
                    "sentence_mask": batch[9],
                    "evidence_labels" : batch[10],
                })
                inputs.update({"evaluate" : True})
                inputs.update({"evaluate_faithfulness": True})
                outputs = model(**inputs)
                logits = outputs[0]
                nnz = torch.mean(outputs[-1]).item()
                all_nnz += to_list(outputs[-2])
                average_nnz += nnz
                if logits.size(-1) > 1:
                    hard_preds = to_list(torch.argmax(logits.float(), dim=-1))
                else:
                    hard_preds = to_list(logits.squeeze(-1))
                # all_rationale_results.extend(hard_rationale_pred)

                hard_rationale_pred = outputs[1]
                p_i = outputs[2] # required for AUPRC at sentence level for sentence evaluation;
                all_results.extend(hard_preds)
                all_complement_context_logits.extend(to_list(torch.softmax(outputs[4], dim=-1)))
                all_rationale_context_logits.extend(to_list(torch.softmax(outputs[0], dim=-1)))
                all_full_context_logits.extend(to_list(torch.softmax(outputs[3], dim=-1)))
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_targets.extend([eval_feature.class_label])

                num_sentences = len(eval_feature.sentences)  # (1/0 score for all soft score sentences)
                pred_sentences = [0]*num_sentences

                # based on indices selected; get corresponding start and end ; add those tokens to the Rationale list
                key = (eval_feature.annotation_id, eval_feature.doc_id)
                last_sentence = sum(eval_feature.sentence_mask)

                for sentence_no, sentence in enumerate(hard_rationale_pred[i]):
                    # batch_index = sentence[0].item()
                    # if batch_index != i:
                    #     continue
                    sentence = sentence.item()
                    # sentence chosen is one outside the mask (especially for fever where you are forcing k selections from a 1 hot vector
                    if sentence_no >= last_sentence or sentence == -1:
                        continue
                    pred_sentences[sentence_no] = 1
                    start_token = eval_feature.sentence_starts[sentence_no]
                    end_token = eval_feature.sentence_ends[sentence_no]
                    # end token is already exclusive, so even it does point to the SEP in the end , so will the gold
                    # but they point to tokens and have to be mapped back
                    if start_token not in eval_feature.token_to_orig_map:
                        continue
                    orig_start_token = eval_feature.token_to_orig_map[start_token]
                    if eval_feature.tokens[end_token] in ["[SEP]", "</s>"]:
                        # last sentence, then end_token points to SEP, we want to get it to point to last word
                        end_token -= 1
                    if end_token not in eval_feature.token_to_orig_map:
                        continue
                    orig_end_token = eval_feature.token_to_orig_map[end_token] + 1  # +1 since it needs to be exclusive

                    all_rationale_results.append(metrics.Rationale(ann_id=key[0],
                                                                   docid=key[1],
                                                                   start_token=orig_start_token,
                                                                   end_token=orig_end_token))
                all_results_dictionary[key] = hard_preds[i]

                # create the appropriate sized gold truth vector
                soft_sentence_scores.append((to_list(p_i[i]) + [0] * (num_sentences - args.max_num_sentences))[:num_sentences])
                pred_sentence_scores.append(pred_sentences)
                gold_sentence_scores.append(eval_feature.gold_sentences)

        output_ratioanale_prediction_file = os.path.join(args.output_dir, "rationale_predictions.json")
        file_pointer = open(output_ratioanale_prediction_file, "w+")
        ann_to_rat = metrics._keyed_rationale_from_list(all_rationale_results)
        ann_to_gold = metrics._keyed_rationale_from_list(true_rationales)
        example_dict = dict()
        for ex in examples:
            example_dict[(ex.id, ex.doc_id)] = ex

        evidence_iou = metrics.partial_match_score(true_rationales, all_rationale_results, [0.1, 0.5, 0.9])
        hard_rationale_metrics = metrics.score_hard_rationale_predictions(true_rationales, all_rationale_results)
        if isinstance(all_targets[0], float):
            mse = mean_squared_error(all_targets, all_results)
            results = (mse, mse)
        else:
            results = (classification_report(all_targets, all_results, target_names=class_labels, output_dict=True)[
                           'weighted avg']['f1-score'], accuracy_score(all_targets, all_results))
        task_performances.append(results[0])
        evidence_performances.append((evidence_iou[0]['macro']['f1'], evidence_iou[1]['macro']['f1']))
    for e, feature in enumerate(features):
        unique_id = (feature.annotation_id, feature.doc_id)
        tokens = feature.tokens
        token_level_rationales = ann_to_rat[unique_id]
        doc_tokens = example_dict[unique_id].doc_toks
        if isinstance(feature.class_label, float):
            class_label = feature.class_label
            predicted_class_label = all_results_dictionary[unique_id]
        else:
            class_label = class_labels[feature.class_label]
            predicted_class_label = all_results_dictionary[unique_id]
        # human_rationale = [tokens[i] for i in np.where(np.asarray(feature.evidence_label)== 1)[0]] # This is stale too since there is extra padding into what both models consider
        # query = " ".join([tok for tok in feature.tokens[:64] if tok not in ["[CLS]", "[PAD]", "[SEP]"]])
        rationales = []
        for rationale in token_level_rationales:
            start_token = rationale.start_token
            end_token = rationale.end_token
            rationale_text = doc_tokens[start_token:end_token]
            rationales.append({"start_token": start_token, "end_token": end_token})
        output_dict = {"annotation_id": feature.annotation_id,
                       "classification": predicted_class_label,
                       "rationales": [{"docid": feature.doc_id, "hard_rationale_predictions": rationales}],

                       "gold_classification": class_label,
                       # "gold_rationales" : ann_to_gold[unique_id],
                       # "predicted_rationales" : ann_to_rat[unique_id],
                       "query": example_dict[unique_id].query,
                       "doc_tokens": doc_tokens}
        file_pointer.write(json.dumps(output_dict) + "\n")
        # Print the rationale chunks predicted to file

    file_pointer.close()
    logger.info("Averaged classification results: %.4f" % np.average(task_performances))
    avg_evidence_perf = np.average(np.asarray(evidence_performances).transpose(), 1)
    logger.info("Averaged Evidence results at thesholds 0.1, 0.5: %.4f, %.4f" % (avg_evidence_perf[0], avg_evidence_perf[1]))
    logger.info('sparsity Acheived : %.4f' % (average_nnz/len(eval_dataloader)))
    logger.info('Variance Acheived : %.4f' % (np.asarray(all_nnz).sum(1).std()))
    # AUPRC evaluation
    aucs = []
    #if not all([len(gold_sentence_scores[i]) == len(soft_sentence_scores[i]) and
    #            len(gold_sentence_scores[i]) == len(pred_sentence_scores[i])
    #            for i in range(len(gold_sentence_scores))]):
    for pred, true in zip(soft_sentence_scores, gold_sentence_scores):
        precision, recall, _ = precision_recall_curve(true, pred)
        _auc = auc(recall, precision)
        if not np.isnan(_auc):
            aucs.append(_auc)
        else:
            aucs.append(0.0)
    avg_auc = np.average(aucs)
    # Evaluating sentence level precision like Lie et. al.
    sentence_precision, sentence_recall, sentence_f1 = [], [], []
    for pred, true in zip(pred_sentence_scores, gold_sentence_scores):
        precision, recall, f1 = precision_score(true, pred), recall_score(true, pred), f1_score(true, pred)
        sentence_precision.append(precision)
        sentence_recall.append(recall)
        sentence_f1.append(f1)
    avg_sent_precision = np.average(sentence_precision)
    avg_sent_recall = np.average(sentence_recall)
    avg_sent_f1 = np.average(sentence_f1)

    # Faithfulness Evaluation;
    comprehensiveness_score = np.average([max(all_full_context_logits[i]) - max(all_complement_context_logits[i]) for i in range(len(all_full_context_logits))])
    comprehensiveness_entropies = [entropy(all_full_context_logits[i]) - entropy(all_complement_context_logits[i]) for i in range(len(all_full_context_logits))]
    comprehensiveness_entropy = np.average(comprehensiveness_entropies)
    comprehensiveness_kl = np.average(
        list(entropy(all_complement_context_logits[i], all_full_context_logits[i]) for i in range(len(all_full_context_logits))))
    sufficiency_score = np.average([max(all_full_context_logits[i]) - max(all_rationale_context_logits[i]) for i in range(len(all_full_context_logits))])
    sufficiency_entropies = [entropy(all_full_context_logits[i]) - entropy(all_rationale_context_logits[i]) for i in range(len(all_full_context_logits))]
    sufficiency_entropy = np.average(sufficiency_entropies)
    sufficiency_kl = np.average(
        list(entropy(all_rationale_context_logits[i], all_full_context_logits[i]) for i in range(len(all_full_context_logits))))

    # My comprehensiveness score;
    my_comprehensiveness_score = np.average([max(all_rationale_context_logits[i]) - max(all_complement_context_logits[i])
                                             for i in range(len(all_full_context_logits))])
    my_comprehensiveness_entropies = [entropy(all_rationale_context_logits[i]) - entropy(all_complement_context_logits[i]) for i
                                   in range(len(all_full_context_logits))]
    my_comprehensiveness_entropy = np.average(my_comprehensiveness_entropies)
    my_comprehensiveness_kl = np.average(
        list(entropy(all_complement_context_logits[i], all_rationale_context_logits[i]) for i in
             range(len(all_full_context_logits))))

    logger.info("Comprehensiveness Score : %.4f" %  comprehensiveness_score)
    logger.info("Comprehensiveness Entropy : %.4f" % comprehensiveness_entropy)
    logger.info("Comprehensiveness KL Divergence : %.4f" % comprehensiveness_kl)
    logger.info("Sufficiency Score : %.4f" % sufficiency_score)
    logger.info("Sufficiency Entropy : %.4f" % sufficiency_entropy)
    logger.info("Sufficiency KL Divergence : %.4f" % sufficiency_kl)

    # AUPRC
    logger.info("AUPRC: %.4f" % avg_auc)
    logger.info("Sentence precision, recall, f1: %.4f, %.4f, %.4f" % (avg_sent_precision, avg_sent_recall, avg_sent_f1))
    logger.info("My Comprehensiveness Score : %.4f" % my_comprehensiveness_score)
    logger.info("My Comprehensiveness Entropy : %.4f" % my_comprehensiveness_entropy)
    logger.info("My Comprehensiveness KL Divergence : %.4f" % my_comprehensiveness_kl)
    return results

def main():
    # Required parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', required=True,
                        help='Which directory contains a {train,val,test}.jsonl file?')
    parser.add_argument('--output_dir', dest='output_dir', required=True,
                        help='Where shall we write intermediate models + final data to?')
    parser.add_argument('--model_params', dest='model_params', required=True,
                        help='JSoN file for loading arbitrary model parameters (e.g. optimizers, pre-saved files, etc.')
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_split", type=str, default="val")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--tf_summary", action="store_true", default=False)

    # Input parameters
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_query_length", default=64, type=int)
    parser.add_argument("--truncate", default=False, action="store_true")

    # Input Parameters due to IB
    parser.add_argument("--max_num_sentences", default=50, type=int)
    parser.add_argument("--max_sentence_len", default=30, type=int) # Avoid this excessive padding
    parser.add_argument('--gold_evidence', action="store_true", default=False)
    parser.add_argument('--query_mask', action="store_true", default=False, help="Whether to add query or not")
    parser.add_argument('--low_resource', action="store_true", default=False)
    parser.add_argument('--warm_start', action="store_true", default=False)
    parser.add_argument('--semi_supervised', type=float, default=0.0)
    parser.add_argument('--anneal', action="store_true", default=False)
    parser.add_argument('--num_evaluations', type=int, default=1)

    # Training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--wait_step", default=5, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=100.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    # Logging
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # Multi-GPU
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    args = parser.parse_args()

    # Parse model args json
    with open(args.model_params, 'r') as fp:
        logging.debug(f'Loading model parameters from {args.model_params}')
        model_params = json.load(fp)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(model_params["config_name"] if model_params["config_name"] else model_params["model_name_or_path"])
    tokenizer = tokenizer_class.from_pretrained(model_params["tokenizer_name"] if model_params["tokenizer_name"] else model_params["model_name_or_path"],
                                                do_lower_case=model_params["do_lower_case"])

    # exchange information between config and model parameters as required
    config.num_labels = len(model_params['classes'])
    model_params["max_query_len"] = args.max_query_length
    model_params["warm_start"] = args.warm_start
    model_params["semi_supervised"] = args.semi_supervised
    model = model_class.from_pretrained(model_class, pretrained_model_name_or_path=model_params["model_name_or_path"], from_tf=bool('.ckpt' in model_params["model_name_or_path"]),
                                        config=config, model_params=model_params)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        # TODO: rewrite save pretrained to also dump model params into a json so that thay have to be loaded
        model_to_save.save_pretrained(args.output_dir)
        train_dataset, _ = load_and_cache_examples(args, model_params, tokenizer, evaluate=False, split="train", output_examples=False)
        global_step, tr_loss = train(args, model_params, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        torch.save(model_params, os.path.join(args.output_dir, 'model_params.json'))
        # model = model_class.from_pretrained(model_class, pretrained_model_name_or_path=args.output_dir, config=config, model_params=model_params)
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=model_params["do_lower_case"])
        # model.to(args.device)

    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir + "/best_model"]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(model_class, pretrained_model_name_or_path=checkpoint, config=config, model_params=model_params, force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model_params, model, tokenizer, prefix=global_step, output_examples=True, split=args.eval_split)
            results = {"Best F1":result[0], "Best Accuracy":result[1]}

    logger.info("Results on the split {} : {}".format(args.eval_split, results))
    return results

if __name__ == '__main__':
    main()
