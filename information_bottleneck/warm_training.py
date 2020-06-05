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

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, classification_report

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

from bert_explainer import DistilBertSentenceClassifier

logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
# let's make this more or less deterministic (not resistent to restarts)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, RobertaConfig)), ())


MODEL_CLASSES = {
    'distilbert': (DistilBertConfig, DistilBertSentenceClassifier, DistilBertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def load_and_cache_examples(args, model_params, tokenizer, evaluate=False, split="train", output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # only load one split
    input_file = os.path.join(args.data_dir, split)
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}_{}_sent'.format(
        split,
        list(filter(None, model_params["tokenizer_name"].split('/'))).pop(),
        str(args.max_seq_length),
        str(args.low_resource)))
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
    tb_writer = None
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
        dataset, eval_features = load_and_cache_examples(args, model_params, tokenizer, evaluate=True, split="val", output_examples=False)
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
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_f1 = -1 # Single Measure of sentence classification
    wait_step = 0
    stop_training = False
    metric_name = "IOU Macro F1"
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
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
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
                        if best_f1 < results:
                            logger.info("Saving model with best %s: %.2f -> %.2f on epoch=%d" % \
                                        (metric_name, best_f1 * 100, results * 100, epoch))
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
                    logging_loss = tr_loss

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
    all_rationale_results = []
    all_results = []
    all_targets = []
    results = None
    class_interner = dict((y, x) for (x, y) in enumerate(model_params['classes']))
    class_labels = [k for k, v in sorted(class_interner.items(), key=lambda x: x[1])]
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
            hard_rationale_pred = to_list(torch.sigmoid(logits))
            hard_rationale_pred = [np.where(np.asarray(l) > 0.5)[0] for l in hard_rationale_pred]

            hard_preds = to_list(torch.argmax(logits.float(), dim=-1))
            # all_rationale_results.extend(hard_rationale_pred)
            all_results.extend(hard_preds)

        for i, example_index in enumerate(example_indices):
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_targets.extend([eval_feature.class_label])

            # based on indices selected; get corresponding start and end ; add those tokens to the Rationale list
            key = (eval_feature.annotation_id, eval_feature.doc_id)
            last_sentence = sum(eval_feature.sentence_mask)
            for sentence in hard_rationale_pred[i]:
                if sentence >= last_sentence:
                    continue
                start_token = eval_feature.sentence_starts[sentence]
                end_token = eval_feature.sentence_ends[sentence]
                # end token is already exclusive, so even it does point to the SEP in the end , so will the gold
                # but they point to tokens and have to be mapped back

                orig_start_token = eval_feature.token_to_orig_map[start_token]
                if eval_feature.tokens[end_token] in ["[SEP]", "</s>"]:
                    # last sentence, then end_token points to SEP, we want to get it to point to last word
                    end_token -= 1
                orig_end_token = eval_feature.token_to_orig_map[end_token] + 1 # +1 since it needs to be exclusive

                all_rationale_results.append(metrics.Rationale(ann_id=key[0],
                                                               docid=key[1],
                                                               start_token=orig_start_token,
                                                               end_token=orig_end_token))

    # TODO: What if the rationales  predicted are empty? Return tf-idf ranked rationales (top K)
    evidence_iou = metrics.partial_match_score(true_rationales, all_rationale_results, [0.1, 0.5, 0.9])
    logger.info("Rationale Partial score metrics:")
    logger.info(evidence_iou)
    tb_writer.add_scalar('rationale_f1', evidence_iou[0]['macro']['f1'], global_step)
    # returning for threshold 0.1
    results = (
    classification_report(all_targets, all_results, target_names=class_labels, output_dict=True)['weighted avg'][
        'f1-score'], accuracy_score(all_targets, all_results))
    logger.info(" Task Performance = %f", results[0])
    return evidence_iou[0]['macro']['f1']

def evaluate(args, model_params, model, tokenizer, prefix="", output_examples=False, split="val"):

    # Keep all eraser annotations at  the ready to go
    annotations = annotations_from_jsonl(os.path.join(args.data_dir, split + '.jsonl'))
    true_rationales = list(chain.from_iterable(metrics.Rationale.from_annotation(ann) for ann in annotations))


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

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # all_rationale_targets = []
    all_rationale_results = []
    class_interner = dict((y, x) for (x, y) in enumerate(model_params['classes']))
    class_labels = [k for k, v in sorted(class_interner.items(), key=lambda x: x[1])]
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
            outputs = model(**inputs)
            logits = outputs[0]
            hard_rationale_pred = to_list(torch.sigmoid(logits))
            hard_rationale_pred = [np.where(np.asarray(l) > 0.5)[0] for l in hard_rationale_pred]
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            # based on indices selected; get corresponding start and end ; add those tokens to the Rationale list
            key = (eval_feature.annotation_id, eval_feature.doc_id)
            last_sentence = sum(eval_feature.sentence_mask)
            for sentence in hard_rationale_pred[i]:
                if sentence >= last_sentence:
                    continue
                start_token = eval_feature.sentence_starts[sentence]
                end_token = eval_feature.sentence_ends[sentence]

                orig_start_token = eval_feature.token_to_orig_map[start_token]
                if eval_feature.tokens[end_token] in ["[SEP]", "</s>"]:
                    # last sentence, then end_token points to SEP, we want to get it to point to last word
                    end_token -= 1
                orig_end_token = eval_feature.token_to_orig_map[end_token] + 1  # +1 since it needs to be exclusive

                all_rationale_results.append(metrics.Rationale(ann_id=key[0],
                                                               docid=key[1],
                                                               start_token=orig_start_token,
                                                               end_token=orig_end_token))

    output_ratioanale_prediction_file = os.path.join(args.output_dir, "rationale_predictions.json")
    file_pointer = open(output_ratioanale_prediction_file, "w+")
    ann_to_rat = metrics._keyed_rationale_from_list(all_rationale_results)
    ann_to_gold = metrics._keyed_rationale_from_list(true_rationales)
    example_dict = dict()
    for ex in examples:
        example_dict[(ex.id, ex.doc_id)] = ex

    evidence_iou = metrics.partial_match_score(true_rationales, all_rationale_results, [0.1, 0.5, 0.9])

    for e, feature in enumerate(features):
        unique_id = (feature.annotation_id, feature.doc_id)
        tokens = feature.tokens
        token_level_rationales = ann_to_rat[unique_id]
        doc_tokens = example_dict[unique_id].doc_toks
        class_label = class_labels[feature.class_label]
        # human_rationale = [tokens[i] for i in np.where(np.asarray(feature.evidence_label)== 1)[0]] # This is stale too since there is extra padding into what both models consider
        # query = " ".join([tok for tok in feature.tokens[:64] if tok not in ["[CLS]", "[PAD]", "[SEP]"]])
        rationales = []
        for rationale in token_level_rationales:
            start_token = rationale.start_token
            end_token = rationale.end_token
            rationale_text = doc_tokens[start_token:end_token]
            rationales.append({"start_token": start_token, "end_token": end_token})
        output_dict = {"annotation_id": feature.annotation_id,
                       "rationales": [{"docid": feature.doc_id, "hard_rationale_predictions": rationales}],
                       "gold_classification": class_label,
                       # "gold_rationales" : ann_to_gold[unique_id],
                       # "predicted_rationales" : ann_to_rat[unique_id],
                       "query": example_dict[unique_id].query,
                       "doc_tokens": doc_tokens}
        file_pointer.write(json.dumps(output_dict) + "\n")
        # Print the rationale chunks predicted to file
    result = evidence_iou
    file_pointer.close()
    return result

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

    # Input Parameters due to IB
    parser.add_argument("--max_num_sentences", default=50, type=int)
    parser.add_argument("--max_sentence_len", default=30, type=int) # Avoid this excessive padding
    parser.add_argument('--gold_evidence', action="store_true", default=False)
    parser.add_argument('--query_mask', action="store_true", default=False, help="Whether to add query or not")
    parser.add_argument('--low_resource', type=float, default=1.0)

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
