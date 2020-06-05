from __future__ import absolute_import, division, print_function

import os
import json
import re
import tqdm
import logging
import math
import collections
from io import open
import pdb
from spacy.lang.en import English
import numpy as np
from torch.utils.data import Dataset
import torch
from torch._six import container_abcs
import pickle
import bisect
from pytorch_transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize
from typing import Any, Callable, Dict, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class Example(object):
    def __init__(self, id, query, query_type, doc_toks, evidence_spans, evidence_toks, label):
        self.id = id
        self.query = query
        self.query_type = query_type
        self.doc_toks = doc_toks
        self.evidence_spans = evidence_spans
        self.evidence_toks = evidence_toks
        self.label = label


class ClassificationFeature(object):
    def __init__(self,
                 unique_id,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 class_label,
                 evidence_label=None,
                 doc_id=None):
        self.unique_id = unique_id
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.class_label = class_label
        self.evidence_label = evidence_label
        self.doc_id = doc_id

def read_json(args):
    filename = os.path.join(args.data_dir, "yelp_reviews.jsonl")
    dict_examples = [json.loads(ex) for ex in open(filename).readlines()]
    examples = []
    for ex in dict_examples:
        examples.append(Example(ex['id'], ex['query'], None, ex['doc_toks'], ex['evidence_spans'], ex['evidence_toks'], ex['label']))
    return examples

def read_examples(args, model_params, annotated_dataset, documents, split):

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False


    examples = []
    no_annot = 0
    for example_no, item in tqdm.tqdm(enumerate(annotated_dataset)):
        evidences = item.evidences
        label = item.classification
        evidence_list = []
        start_tokens = []
        end_tokens = []
        document_list = []
        for g in evidences:
            # List made up of one or more Evidence objects
            # Sometimes entire sentences in the document are turned on sometimes its just words in combined tokens
            for e in g:
                evidence_list.append(e)
                start_tokens.append(e.start_token)
                # All except boolq are span based
                if os.path.basename(args.data_dir) == "boolq":
                    end_tokens.append(e.end_token + 1)
                else:
                    end_tokens.append(e.end_token)
                document_list.append(e.docid)
        document_list = list(set(document_list))
        # Handle multple source documents
        assert len(document_list) < 2
        if len(document_list) == 0:
            no_annot += 1
            continue
        doc = documents[document_list[0]]
        tokens = [item for sublist in doc for item in sublist]
        # Unique evidence pairs
        evidence_spans = list(set(zip(start_tokens, end_tokens)))
        evidence_tokens = [] # List of lists
        for span in evidence_spans:
            evidence_tokens.append(tokens[span[0]:span[1]])
        # Label
        label = item.classification
        query = item.query
        annotation_id = item.annotation_id
        query_type = item.query_type
        examples.append(Example(
                        id = annotation_id,
                        query=query,
                        query_type = query_type,
                        doc_toks=tokens,
                        evidence_spans=evidence_spans,
                        evidence_toks=evidence_tokens,
                        label=label
                        ))
    if args.debug:
        return examples[:10]
    if args.low_resource and split == "train":
        return examples[:int(len(examples)/2)]
    return examples

requires_question_mask = {
    'movies' : True,
    'fever' : True,
    'boolq_truncated' : True,
    'evidence_truncated' : True,
    'boolq' : True,
    'multirc' : True,
}

def convert_examples_to_features(args, model_params, examples, tokenizer,
                                 max_seq_length, max_query_length, is_training,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):

    features = []

    if "roberta" in args.model_type:
        # Change the start and end token
        cls_token = "<s>"
        sep_token = "</s>"
        pad_token = tokenizer._convert_token_to_id("<pad>")

    evidence_classes = dict((y, x) for (x, y) in enumerate(model_params['classes']))
    
    predicted_evidence_labels = None
    if is_training and args.predicted_train_evidence_file and os.path.exists(args.predicted_train_evidence_file):
        predicted_evidence_labels = pickle.load(open(args.predicted_train_evidence_file, "rb"))
    if not is_training and args.predicted_eval_evidence_file and os.path.exists(args.predicted_eval_evidence_file):
        predicted_evidence_labels = pickle.load(open(args.predicted_eval_evidence_file, "rb"))
    unique_id = 1000000000
    for (example_index, example) in tqdm.tqdm(enumerate(examples)):
        query_tokens = []
        if example.query is not None:
            query_tokens = tokenizer.tokenize(example.query)

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        example.doc_toks = [tok.strip() for tok in example.doc_toks]
        for (i, token) in enumerate(example.doc_toks):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)  #, add_prefix_space=True)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        # all_doc_tokens = tokenizer.tokenize(" ".join(example.doc_toks))

        evidence_span_positions = []
        for span in example.evidence_spans:
            tok_start_position = orig_to_tok_index[span[0]]
            if span[1] < len(example.doc_toks) - 1:
                tok_end_position = orig_to_tok_index[span[1] + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            evidence_span_positions.append((tok_start_position, tok_end_position))

        # Not breaking up doc_tokens for now since its just a classification,
        # TODO: Introduce global normalization
        doc_span_index = 0

        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        tokens = []
        segment_ids = []
        p_mask = []
        token_to_orig_map = {}

        # CLS (Required for classification)
        tokens.append(cls_token)
        segment_ids.append(cls_token_segment_id)
        p_mask.append(0)
        cls_index = 0

        # Query (if not Empty)
        if requires_question_mask[os.path.basename(args.data_dir)]:
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

            # Pad or truncate tokens upto max_query_length
            while len(tokens) < max_query_length:
                tokens.append("[PAD]")
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            # Truncate
            if len(tokens) > max_query_length:
                tokens = tokens[:max_query_length - 1]
                segment_ids = segment_ids[:max_query_length - 1]
                p_mask = p_mask[:max_query_length - 1]
                # Add SEP token after this
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

        # Full Document
        # replace tokens with evidence only (in the truncated output?)
        if args.gold_evidence:
            evidence_length = 0
            for span in evidence_span_positions:
                for i in range(span[0],span[1]):
                    tokens.append(all_doc_tokens[i])
                    segment_ids.append(sequence_b_segment_id)
                    p_mask.append(0)
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
                    evidence_length += 1
            paragraph_len = evidence_length
            # Add SEP token after every evidence snippet
        else:
            for i,tok in enumerate(all_doc_tokens):
                tokens.append(all_doc_tokens[i])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)
                token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
            paragraph_len = len(all_doc_tokens)

        # SEP token
        tokens.append(sep_token)
        segment_ids.append(sequence_b_segment_id)
        p_mask.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if input_ids[i] != 0 else 0 for i in range(len(input_ids))]

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(pad_token)
            input_mask.append(0 if mask_padding_with_zero else 1)
            segment_ids.append(pad_token_segment_id)
            p_mask.append(1)

        # Truncate
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length-1]
            input_mask = input_mask[:max_seq_length-1]
            segment_ids = segment_ids[:max_seq_length-1]
            p_mask = segment_ids[:max_seq_length-1]
            # Add SEP token after this
            input_ids.append(tokenizer._convert_token_to_id(sep_token))
            input_mask.append(1)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # Label (Intern class label)
        label = evidence_classes[example.label]

        # Evidence label if required.
        evidence_label = np.zeros(max_seq_length)

        doc_offset = max_query_length
        for span in evidence_span_positions:
            for i in range(span[0], span[1]):
                i_alt = doc_offset + i
                if i_alt < max_seq_length:
                    evidence_label[i_alt] = 1
        # if query was added then that is required evidence
        evidence_label[:max_query_length] = input_mask[:max_query_length]

        # if evidence was already predicted for the rest of the doc, initialize with that instead of gold
        # Logic that with probability p do this replacement or dont (at train time only)
        if predicted_evidence_labels and (args.pal_attention or args.focus_attention):
            if example.id in predicted_evidence_labels:
                if is_training:
                    if np.random.rand() > (1.0 - args.gamma):
                        evidence_label[max_query_length:] = predicted_evidence_labels[example.id]
                else:
                    evidence_label[max_query_length:] = predicted_evidence_labels[example.id]


        if args.random_evidence:
            evidence_label = np.zeros(max_seq_length)
            for span in evidence_span_positions:
                span_length = span[1] - span[0]
                # choose a random part of the input from max_query_length: onwards and assign 1
                # for now same span or intersecting spans may be possible
                random_start = np.random.randint(max_query_length, 512)
                for i in range(random_start, random_start + span_length):
                    if i < max_seq_length:
                        evidence_label[i] = 1

        if example_index <  5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (unique_id))
            logger.info("example_index: %s" % (example_index))
            logger.info("doc_span_index: %s" % (doc_span_index))
            logger.info("tokens: %s" % " ".join(tokens))
            logger.info("token_to_orig_map: %s" % " ".join([
                "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(
                "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("answer label: %s" % (label))
            logger.info("evidence labels : {}". format(evidence_label))

        features.append(
            ClassificationFeature(
                unique_id=unique_id,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                cls_index=cls_index,
                p_mask=p_mask,
                paragraph_len=paragraph_len,
                class_label=label,
                evidence_label=evidence_label,
                doc_id = example.id))
        unique_id += 1


    return features

# start_token is inclusive, end_token is exclusive
class Rationale:
    ann_id: str
    docid: str
    start_token: int
    end_token: int

def convert_binary_mask_rationales(binary_mask):
    rationale_list = []


def _f1(_p, _r):
    if _p == 0 or _r == 0:
        return 0
    return 2 * _p * _r / (_p + _r)

def _keyed_rationale_from_list(rats: List[Rationale]) -> Dict[Tuple[str, str], Rationale]:
    ret = defaultdict(set)
    for r in rats:
        ret[(r.ann_id, r.docid)].add(r)
    return ret

def score_hard_rationale_predictions(truth: List[Rationale], pred: List[Rationale]) -> Dict[str, Dict[str, float]]:
    """Computes instance (annotation)-level micro/macro averaged F1s"""
    scores = dict()
    truth = set(truth)
    pred = set(pred)
    micro_prec = len(truth & pred) / len(pred)
    micro_rec = len(truth & pred) / len(truth)
    micro_f1 = _f1(micro_prec, micro_rec)

    scores['instance_micro'] = {
                                'p': micro_prec,
                                'r': micro_rec,
                                'f1': micro_f1,
                               }

    ann_to_rat = _keyed_rationale_from_list(truth)
    pred_to_rat = _keyed_rationale_from_list(pred)
    instances_to_scores = dict()
    for k in set(ann_to_rat.keys()) | (pred_to_rat.keys()):
        if len(pred_to_rat.get(k, set())) > 0:
            instance_prec = len(ann_to_rat.get(k, set()) & pred_to_rat.get(k, set())) / len(pred_to_rat[k])
        else:
            instance_prec = 0
        if len(ann_to_rat.get(k, set())) > 0:
            instance_rec = len(ann_to_rat.get(k, set()) & pred_to_rat.get(k, set())) / len(ann_to_rat[k])
        else:
            instance_rec = 0
        instance_f1 = _f1(instance_prec, instance_rec)
        instances_to_scores[k] = {
                                    'p': instance_prec,
                                    'r': instance_rec,
                                    'f1': instance_f1,
                                 }
    # these are calculated as sklearn would
    macro_prec = sum(instance['p'] for instance in instances_to_scores.values()) / len(instances_to_scores)
    macro_rec = sum(instance['r'] for instance in instances_to_scores.values()) / len(instances_to_scores)
    macro_f1 = sum(instance['f1'] for instance in instances_to_scores.values()) / len(instances_to_scores)
    scores['instance_macro'] = {
                                'p': macro_prec,
                                'r': macro_rec,
                                'f1': macro_f1,
                               }
    return scores
