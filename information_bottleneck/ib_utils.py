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
import argparse
from pytorch_transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize
from typing import Any, Callable, Dict, List, Tuple
from collections import defaultdict
import bisect
from copy import deepcopy as copy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from itertools import chain
import metrics
from eraser.rationale_benchmark.utils import annotations_from_jsonl, load_jsonl, write_jsonl

logger = logging.getLogger(__name__)

class Example(object):
    def __init__(self, id, doc_id, query, query_type, doc_toks, sentences, evidence_spans, evidence_toks, label):
        self.id = id
        self.doc_id = doc_id
        self.query = query
        self.query_type = query_type
        self.doc_toks = doc_toks
        self.sentences = sentences
        self.evidence_spans = evidence_spans
        self.evidence_toks = evidence_toks
        self.label = label

class ClassificationFeature(object):
    def __init__(self,
                 unique_id,
                 annotation_id,
                 doc_id,
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
                 evidence_label=None):
        self.unique_id = unique_id
        self.annotation_id = annotation_id
        self.doc_id = doc_id
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

class CognitiveFeature(object):
    def __init__(self,
                 unique_id,
                 annotation_id,
                 doc_id,
                 doc_span_index,
                 tokens,
                 sentences,
                 gold_sentences,
                 token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 sentence_starts,
                 sentence_ends,
                 sentence_mask,
                 paragraph_len,
                 class_label,
                 evidence_label=None):
        self.unique_id = unique_id
        self.annotation_id = annotation_id
        self.doc_id = doc_id
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.sentences = sentences
        self.gold_sentences =gold_sentences
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.sentence_starts = sentence_starts
        self.sentence_ends = sentence_ends
        self.sentence_mask = sentence_mask
        self.paragraph_len = paragraph_len
        self.class_label = class_label
        self.evidence_label = evidence_label

def read_examples(args, model_params, annotated_dataset, documents, split):

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False


    examples = []
    no_annot = 0
    no_evidence_in_window = 0

    truncated_annotation_dataset = {}
    
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
                # seems like boolq end token is not inclusive
                if os.path.basename(args.data_dir) == "boolq":
                    end_tokens.append(e.end_token + 1)
                else:
                    end_tokens.append(e.end_token)
                document_list.append(e.docid)
        document_list = list(set(document_list))
        # Handle multple source documents
        if "beer" in args.data_dir or 'imdb' in args.data_dir:
            document_list = [item.annotation_id]
        assert len(document_list) < 2
        if len(document_list) == 0:
            no_annot += 1
            document_list = item.docids
        if not document_list:
            document_list = [item.annotation_id]
        doc = documents[document_list[0]]
        tokens = [item for sublist in doc for item in sublist]
        sentences = doc
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

        # if evidence inference or boolq: do tfidf over of query over 20 sentence windows with stride 5
        if args.truncate:
            tfidf_vectorizer = TfidfVectorizer(use_idf=True)
            all_docs = [query]
            doc_spans = []
            sent_spans = []
            sentence_starts = [0] + np.cumsum([len(s) for s in sentences])[:-1].tolist()
            for span_start in range(0, len(sentences) - args.max_num_sentences + 1, 5):
                sentences_in_span = sentences[span_start:span_start + args.max_num_sentences]
                paragraph = [tok for sentence in sentences_in_span for tok in sentence]
                doc_spans.append((sentence_starts[span_start], len(paragraph)))
                sent_spans.append((span_start, span_start + args.max_num_sentences))
                all_docs.append(" ".join(paragraph))
                # tfidf vectorize
            # put the last args.max_num_sentences in another doc
            last_paragraph = [tok for sentence in sentences[-args.max_num_sentences:] for tok in sentence]
            if len(sentences) - args.max_num_sentences > 0:
                all_docs.append(" ".join(last_paragraph))
                sent_spans.append((len(sentences) - args.max_num_sentences, len(sentences)))
                doc_spans.append((sentence_starts[len(sentences) - args.max_num_sentences] ,len(last_paragraph)))
            else:
                # doc has fewer than max_num_sentences
                all_docs.append(" ".join(last_paragraph))
                sent_spans.append((0, len(sentences)))
                doc_spans.append((sentence_starts[0], len(last_paragraph)))
            tfidf_vecs = tfidf_vectorizer.fit_transform(all_docs)
            cosine_similarities = linear_kernel(tfidf_vecs[0:1], tfidf_vecs).flatten()

            # How often does the best window containing evidence (see what % of time some evidence s
            best_window = np.argsort(cosine_similarities[1:])[::-1][0]
            tokens = all_docs[best_window+1].split() # possible there is an error here since enough
            sentences = sentences[sent_spans[best_window][0]:sent_spans[best_window][1]]
            best_span = (doc_spans[best_window][0], doc_spans[best_window][0] + doc_spans[best_window][1])
            evidence_in_window = False
            new_evidence_spans = []
            for ev_no, evidence_span in enumerate(evidence_spans):
                if evidence_span[0] >= best_span[0] and evidence_span[1] <= best_span[1]:
                    evidence_in_window = True
                    new_evidence_spans.append((evidence_span[0] - best_span[0], evidence_span[1] - best_span[0]))
            ## new evidence spans
            evidence_spans = new_evidence_spans
            truncated_annotation_dataset[(annotation_id, document_list[0])] = (best_span[0], best_span[1],
                                                                               sent_spans[best_window][0],
                                                                               sent_spans[best_window][1])

            if evidence_in_window:
                no_evidence_in_window += 1


        examples.append(Example(
                        id = annotation_id,
                        doc_id = document_list[0],
                        query=query,
                        query_type = query_type,
                        doc_toks=tokens,
                        sentences = sentences,
                        evidence_spans=evidence_spans,
                        evidence_toks=evidence_tokens,
                        label=label
                        ))


    # The idea is to rewrite the jsonl so that when its loaded again used again,
    # the annotations of rationales are for the selected BERT window
    # if split is not "train":
    if args.truncate:
        json_data = load_jsonl(os.path.join(args.data_dir, split + '.jsonl'))
        for ex_no, pt in enumerate(json_data):
            key = pt['annotation_id'], pt['docids'][0]
            new_window = truncated_annotation_dataset[key]
            new_evidences = []
            for ev_grp in pt['evidences']:
                new_evidence_grp = []
                for ev in ev_grp:
                    if ev['start_token'] >= new_window[0] and ev['end_token'] <= new_window[1]:
                        new_start_token = ev['start_token'] - new_window[0]
                        new_end_token = new_start_token + (ev['end_token'] - ev['start_token'])
                        new_evidence_grp.append({
                            "docid" : pt['docids'][0],
                            "start_token" : new_start_token,
                            "start_sentence" : ev['start_sentence']  - new_window[2],
                            "end_token": new_end_token,
                            "end_sentence" : ev['end_sentence'] - new_window[2],
                            "text" :  ev['text']
                        })
                if len(new_evidence_grp) > 0:
                    new_evidences.append(new_evidence_grp)
            # replace current data
            json_data[ex_no]['evidences'] = new_evidences

        # write the new json data
        write_jsonl(json_data, os.path.join(args.data_dir + "_truncated", split + '.jsonl'))

    if args.debug:
        return examples[:10]
    if args.low_resource and split == "train":
        np.random.shuffle(examples)
        return examples[:int(len(examples) * args.low_resource)]
    return examples

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

    evidence_classes = dict((y, x) for (x, y) in enumerate(model_params['classes']))

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
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

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
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

        # SEP token
        tokens.append(sep_token)
        segment_ids.append(sequence_a_segment_id)
        p_mask.append(1)

        # Full Document
        # replace tokens with evidence only (in the truncated output?)
        if args.gold_evidence:
            evidence_length = 0
            for span in evidence_span_positions:
                for i in range(span[0],span[1]):
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
                    tokens.append(all_doc_tokens[i])
                    segment_ids.append(sequence_b_segment_id)
                    p_mask.append(0)
                    evidence_length += 1
            paragraph_len = evidence_length
            # Add SEP token after every evidence snippet
        else:
            for i,tok in enumerate(all_doc_tokens):
                token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
                tokens.append(all_doc_tokens[i])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)
            paragraph_len = len(all_doc_tokens)

        # SEP token
        tokens.append(sep_token)
        segment_ids.append(sequence_b_segment_id)
        p_mask.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

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
        evidence_label = None
        if args.multitask:
            evidence_label = np.zeros(max_seq_length)
            doc_offset = len(query_tokens) + 2
            for span in evidence_span_positions:
                for i in range(span[0], span[1]):
                    i_alt = doc_offset + i
                    if i_alt < max_seq_length:
                        evidence_label[i_alt] = 1


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
                evidence_label=evidence_label))
        unique_id += 1


    return features


requires_question_mask = {
    'movies' : True,
    'fever' : True,
    'boolq' : True,
    'multirc' : True,
}


def convert_examples_to_document_features(args, model_params, examples, tokenizer,
                                 max_seq_length, max_query_length, is_training,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    # Enumerating sentences  (for multirc, fever and boolq that is already being done)
    # F start and ends to sample from
    # Issue is how to deconvolute and spread the results back
    features = []
    num_sentences = []
    if "roberta" in args.model_type:
        # Change the start and end token
        cls_token = "<s>"
        sep_token = "</s>"

    evidence_classes = dict((y, x) for (x, y) in enumerate(model_params['classes']))

    unique_id = 1000000000
    for (example_index, example) in tqdm.tqdm(enumerate(examples)):
        query_tokens = []
        if example.query is not None:
            query_tokens = tokenizer.tokenize(example.query)
            query_tokens = query_tokens[:max_query_length-1]
            if len(query_tokens) < max_query_length-1:
                while len(query_tokens) < max_query_length-1:
                    query_tokens.append(pad_token)

        all_sentences = []
        sentence_starts = []
        sentence_ends = []
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        token_cnt = 0
        for sent_no, sent in enumerate(example.sentences):
            sent_tokens = [tok.strip() for tok in sent]
            sent_tokenized = []
            sentence_starts.append(len(all_doc_tokens))
            for token in sent_tokens:
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    all_doc_tokens.append(sub_token)
                    tok_to_orig_index.append(token_cnt)
                token_cnt += 1
                sent_tokenized += sub_tokens
            sentence_ends.append(len(all_doc_tokens))
            # If ther are sentences with fewer than 5 tokens ignore them
            # if len(sent_tokenized) < 5:
            #     continue
            all_sentences.append(sent_tokenized)

        all_sentences_untruncated = copy(all_sentences)
        sentence_starts_untruncated = copy(sentence_starts)
        sentence_ends_untruncated = copy(sentence_ends)

        # Document spans
        # sentence_starts, sentence_ends, all_sentences, sentence_mask, all_doc_tokens
        document_sentence_starts = []
        document_sentence_ends = []
        document_all_sentences = []
        document_sentence_masks = []
        document_span_tokens = []

        # truncate if all_doc_tokens + query + 2  > max_seq_len
        if len(query_tokens) + len(all_doc_tokens) + 3 > max_seq_length:
            sentence_break = bisect.bisect_left(sentence_starts, max_seq_length - len(query_tokens)- 2) - 1
            surplus = (max_seq_length - len(query_tokens) - 3)  - sentence_starts[sentence_break]  #>=0
            last_sentence = all_sentences[sentence_break][:surplus]
            all_sentences = all_sentences[:sentence_break] + [last_sentence]
            sentence_starts = sentence_starts[:sentence_break+1]
            sentence_ends = sentence_ends[:sentence_break] + [sentence_starts[-1] + len(last_sentence)]
            all_doc_tokens = all_doc_tokens[:sentence_ends[-1]]
            # update tok_to_orig and vice versa??

        sentence_mask = [1] * args.max_num_sentences
        num_valid_sentences = args.max_num_sentences
        if len(all_sentences) > args.max_num_sentences:
            # Shave off requisite from sentence_ends, sentence_starts, all_sentences and all_doc_tokens
            sentence_starts = sentence_starts[:args.max_num_sentences]
            sentence_ends = sentence_ends[:args.max_num_sentences]
            all_sentences = all_sentences[:args.max_num_sentences]
            all_doc_tokens = all_doc_tokens[:sentence_ends[-1]]
            num_valid_sentences = args.max_num_sentences
        elif len(all_sentences) < args.max_num_sentences:
            # sentence_starts, sentence_ends need to be expanded
            # tokens can be added anymore
            num_valid_sentences = len(all_sentences)
            num_sentences_to_add = args.max_num_sentences - len(all_sentences)
            for sent_no in range(num_sentences_to_add):
                # add position of the ending SEP token
                sentence_starts.append(-1)
                sentence_ends.append(-1)
                all_sentences.append([])
                sentence_mask[args.max_num_sentences - sent_no - 1] = 0
            # all_doc_tokens cant take more tokens and will be separately padded if required

        # Construct the sequence of CLS Q SEP DOC SEP
        for sent_num, sent in enumerate(all_sentences):
            assert sent == all_doc_tokens[sentence_starts[sent_num]:sentence_ends[sent_num]]

        doc_span_index = 0

        # Fill up tokens from query and sentence tokens
        tokens = []
        segment_ids = []
        p_mask = []  # p_mask (inverted) is turned off for CLS and Paragraph tokens only
        token_to_orig_map = {}

        # CLS (Required for classification)
        tokens.append(cls_token)
        segment_ids.append(cls_token_segment_id)
        # p_mask.append(0)
        cls_index = 0

        # TODO: Query not required?
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(sequence_a_segment_id)
            # p_mask.append(1)

        # SEP token
        # The only difference between this setting and the setting for cognitive features
        tokens.append(sep_token)
        segment_ids.append(sequence_a_segment_id)
        # p_mask.append(1)

        for i, tok in enumerate(all_doc_tokens):
            token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
            tokens.append(all_doc_tokens[i])
            segment_ids.append(sequence_b_segment_id)
            # p_mask.append(0)
            # the only changes in all_doc_tokens is shortening it, so tok_to_orig_index is still valid upto len(all_doc_tokens)
        paragraph_len = len(all_doc_tokens)

        # SEP token
        tokens.append(sep_token)
        segment_ids.append(sequence_b_segment_id)
        # p_mask.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if input_ids[i] != 0 else 0 for i in range(len(input_ids))]

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(pad_token)
            input_mask.append(0 if mask_padding_with_zero else 1)
            segment_ids.append(pad_token_segment_id)
            # p_mask.append(1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # Label (Intern class label)
        if len(evidence_classes) == 1:
            label = example.label
        else:
            label = evidence_classes[example.label]

        # Evidence labels
        evidence_sentences = [0]* args.max_num_sentences
        gold_sentences  = [0] * len(all_sentences_untruncated)
        doc_offset = len(query_tokens) + 2
        for span in example.evidence_spans:
            tok_start_position = orig_to_tok_index[span[0]]
            if span[1] < len(example.doc_toks) - 1:
                tok_end_position = orig_to_tok_index[span[1] + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            sentence_membership = range(bisect.bisect_left(sentence_starts[:num_valid_sentences], tok_start_position),
                                        bisect.bisect_left(sentence_ends[:num_valid_sentences], tok_end_position)+1)
            # tok_seq_start_position, tok_seq_end_position = tok_start_position + doc_offset, tok_end_position + doc_offset
            for j in sentence_membership:
                if j > len(evidence_sentences) - 1:
                    # The sentence that was found is beyond what fits in this bert window
                    continue
                evidence_sentences[j] = 1

            sentence_membership_untruncated = range(bisect.bisect_left(sentence_starts_untruncated, tok_start_position),
                                        bisect.bisect_left(sentence_ends_untruncated, tok_end_position) + 1)
            for j in sentence_membership_untruncated:
                if j > len(gold_sentences) - 1:
                    # The sentence that was found is beyond what fits in this bert window
                    continue
                gold_sentences[j] = 1



        # map from sentence in 0, args.max_num_sentences to the 512 tokens (store it in p_mask)
        # p_mask only contains sentence membership of the paragrph and is of length 512 - max_quey_len
        p_mask = np.zeros(max_seq_length-max_query_length+1)
        last_sent_id = None
        for sent_no, sent_boundary in enumerate(zip(sentence_starts, sentence_ends)):
            if sent_boundary != (-1,-1):
                # 1 offset is to accomodate first SEP token
                p_mask[sent_boundary[0]+1:sent_boundary[1]+1] = sent_no
                last_sent_id = sent_no
        # SEP tokens part of first and last sentence
        # handle empty sentences that have no tokens
        p_mask[sentence_ends[last_sent_id]+1] = last_sent_id
        p_mask[0] = 0
        # the rest of the tokens will have 0, but will not be included because of the input_mask

        # After this operation all the useless sentences will point to the first SEP
        sentence_starts = np.asarray(sentence_starts) + max_query_length + 1
        sentence_ends = np.asarray(sentence_ends) + max_query_length + 1


        features.append(CognitiveFeature(
                unique_id=unique_id,
                annotation_id = example.id,
                doc_id = example.doc_id,
                doc_span_index=doc_span_index,
                tokens=tokens,
                sentences = all_sentences_untruncated,
                gold_sentences = gold_sentences,
                token_to_orig_map=token_to_orig_map,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                cls_index=cls_index,
                p_mask=p_mask,
                sentence_starts=sentence_starts,
                sentence_ends=sentence_ends,
                sentence_mask=sentence_mask,
                paragraph_len=paragraph_len,
                class_label=label,
                evidence_label=evidence_sentences))

        num_sentences.append(len(all_sentences_untruncated))

    logger.info("Printing some statistics:")


    return features

def convert_examples_to_sentence_features(args, model_params, examples, tokenizer,
                                 max_seq_length, max_query_length, is_training,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    # Enumerating sentences  (for multirc, fever and boolq that is already being done)
    # F start and ends to sample from
    # Issue is how to deconvolute and spread the results back
    features = []
    num_sentences = []
    if "roberta" in args.model_type:
        # Change the start and end token
        cls_token = "<s>"
        sep_token = "</s>"

    evidence_classes = dict((y, x) for (x, y) in enumerate(model_params['classes']))

    unique_id = 1000000000
    for (example_index, example) in tqdm.tqdm(enumerate(examples)):
        query_tokens = []
        if example.query is not None:
            query_tokens = tokenizer.tokenize(example.query)
            query_tokens = query_tokens[:max_query_length-1]
            if len(query_tokens) < max_query_length-1:
                while len(query_tokens) < max_query_length-1:
                    query_tokens.append(pad_token)

        all_sentences = []
        sentence_starts = []
        sentence_ends = []
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        token_cnt = 0
        for sent_no, sent in enumerate(example.sentences):
            sent_tokens = [tok.strip() for tok in sent]
            sent_tokenized = []
            sentence_starts.append(len(all_doc_tokens))
            for token in sent_tokens:
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    all_doc_tokens.append(sub_token)
                    tok_to_orig_index.append(token_cnt)
                token_cnt += 1
                sent_tokenized += sub_tokens
            sentence_ends.append(len(all_doc_tokens))
            # If ther are sentences with fewer than 5 tokens ignore them
            # if len(sent_tokenized) < 5:
            #     continue
            all_sentences.append(sent_tokenized)

        all_sentences_untruncated = copy(all_sentences)
        sentence_starts_untruncated = copy(sentence_starts)
        sentence_ends_untruncated = copy(sentence_ends)

        # truncate if all_doc_tokens + query + 2  > max_seq_len
        if len(query_tokens) + len(all_doc_tokens) + 3 > max_seq_length:
            sentence_break = bisect.bisect_left(sentence_starts, max_seq_length - len(query_tokens)- 2) - 1
            surplus = (max_seq_length - len(query_tokens) - 3)  - sentence_starts[sentence_break]  #>=0
            last_sentence = all_sentences[sentence_break][:surplus]
            all_sentences = all_sentences[:sentence_break] + [last_sentence]
            sentence_starts = sentence_starts[:sentence_break+1]
            sentence_ends = sentence_ends[:sentence_break] + [sentence_starts[-1] + len(last_sentence)]
            all_doc_tokens = all_doc_tokens[:sentence_ends[-1]]
            # update tok_to_orig and vice versa??

        sentence_mask = [1] * args.max_num_sentences
        num_valid_sentences = args.max_num_sentences
        if len(all_sentences) > args.max_num_sentences:
            # Shave off requisite from sentence_ends, sentence_starts, all_sentences and all_doc_tokens
            sentence_starts = sentence_starts[:args.max_num_sentences]
            sentence_ends = sentence_ends[:args.max_num_sentences]
            all_sentences = all_sentences[:args.max_num_sentences]
            all_doc_tokens = all_doc_tokens[:sentence_ends[-1]]
            num_valid_sentences = args.max_num_sentences
        elif len(all_sentences) < args.max_num_sentences:
            # sentence_starts, sentence_ends need to be expanded
            # tokens can be added anymore
            num_valid_sentences = len(all_sentences)
            num_sentences_to_add = args.max_num_sentences - len(all_sentences)
            for sent_no in range(num_sentences_to_add):
                # add position of the ending SEP token
                sentence_starts.append(-1)
                sentence_ends.append(-1)
                all_sentences.append([])
                sentence_mask[args.max_num_sentences - sent_no - 1] = 0
            # all_doc_tokens cant take more tokens and will be separately padded if required

        # Construct the sequence of CLS Q SEP DOC SEP
        for sent_num, sent in enumerate(all_sentences):
            assert sent == all_doc_tokens[sentence_starts[sent_num]:sentence_ends[sent_num]]

        doc_span_index = 0

        # Fill up tokens from query and sentence tokens
        tokens = []
        segment_ids = []
        p_mask = []  # p_mask (inverted) is turned off for CLS and Paragraph tokens only
        token_to_orig_map = {}

        # CLS (Required for classification)
        tokens.append(cls_token)
        segment_ids.append(cls_token_segment_id)
        # p_mask.append(0)
        cls_index = 0

        # TODO: Query not required?
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(sequence_a_segment_id)
            # p_mask.append(1)

        # SEP token
        # The only difference between this setting and the setting for cognitive features
        tokens.append(sep_token)
        segment_ids.append(sequence_a_segment_id)
        # p_mask.append(1)

        for i, tok in enumerate(all_doc_tokens):
            token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
            tokens.append(all_doc_tokens[i])
            segment_ids.append(sequence_b_segment_id)
            # p_mask.append(0)
            # the only changes in all_doc_tokens is shortening it, so tok_to_orig_index is still valid upto len(all_doc_tokens)
        paragraph_len = len(all_doc_tokens)

        # SEP token
        tokens.append(sep_token)
        segment_ids.append(sequence_b_segment_id)
        # p_mask.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if input_ids[i] != 0 else 0 for i in range(len(input_ids))]

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(pad_token)
            input_mask.append(0 if mask_padding_with_zero else 1)
            segment_ids.append(pad_token_segment_id)
            # p_mask.append(1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # Label (Intern class label)
        if len(evidence_classes) == 1:
            label = example.label
        else:
            label = evidence_classes[example.label]

        # Evidence labels
        evidence_sentences = [0]* args.max_num_sentences
        gold_sentences  = [0] * len(all_sentences_untruncated)
        doc_offset = len(query_tokens) + 2
        for span in example.evidence_spans:
            tok_start_position = orig_to_tok_index[span[0]]
            if span[1] < len(example.doc_toks) - 1:
                tok_end_position = orig_to_tok_index[span[1] + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            range_start = bisect.bisect_left(sentence_starts[:num_valid_sentences], tok_start_position)
            range_end = bisect.bisect_left(sentence_ends[:num_valid_sentences], tok_end_position)+1
            if range_start < len(sentence_starts) and \
                    not tok_start_position == sentence_starts[range_start]:
                range_start -= 1
            sentence_membership = range(range_start, range_end)
            # tok_seq_start_position, tok_seq_end_position = tok_start_position + doc_offset, tok_end_position + doc_offset
            for j in sentence_membership:
                if j > len(evidence_sentences) - 1:
                    # The sentence that was found is beyond what fits in this bert window
                    continue
                evidence_sentences[j] = 1

            range_start = bisect.bisect_left(sentence_starts_untruncated, tok_start_position)
            range_end = bisect.bisect_left(sentence_ends_untruncated, tok_end_position)+1
            if range_start < len(sentence_starts_untruncated) and \
                    not tok_start_position == sentence_starts_untruncated[range_start]:
                range_start -= 1
            sentence_membership_untruncated = range(range_start, range_end)
            for j in sentence_membership_untruncated:
                if j > len(gold_sentences) - 1:
                    # The sentence that was found is beyond what fits in this bert window
                    continue
                gold_sentences[j] = 1



        # map from sentence in 0, args.max_num_sentences to the 512 tokens (store it in p_mask)
        # p_mask only contains sentence membership of the paragrph and is of length 512 - max_quey_len
        p_mask = np.zeros(max_seq_length-max_query_length+1)
        last_sent_id = None
        for sent_no, sent_boundary in enumerate(zip(sentence_starts, sentence_ends)):
            if sent_boundary != (-1,-1):
                # 1 offset is to accomodate first SEP token
                p_mask[sent_boundary[0]+1:sent_boundary[1]+1] = sent_no
                last_sent_id = sent_no
        # SEP tokens part of first and last sentence
        # handle empty sentences that have no tokens
        p_mask[sentence_ends[last_sent_id]+1] = last_sent_id
        p_mask[0] = 0
        # the rest of the tokens will have 0, but will not be included because of the input_mask

        # After this operation all the useless sentences will point to the first SEP
        sentence_starts = np.asarray(sentence_starts) + max_query_length + 1
        sentence_ends = np.asarray(sentence_ends) + max_query_length + 1


        features.append(CognitiveFeature(
                unique_id=unique_id,
                annotation_id = example.id,
                doc_id = example.doc_id,
                doc_span_index=doc_span_index,
                tokens=tokens,
                sentences = all_sentences_untruncated,
                gold_sentences = gold_sentences,
                token_to_orig_map=token_to_orig_map,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                cls_index=cls_index,
                p_mask=p_mask,
                sentence_starts=sentence_starts,
                sentence_ends=sentence_ends,
                sentence_mask=sentence_mask,
                paragraph_len=paragraph_len,
                class_label=label,
                evidence_label=evidence_sentences))

        num_sentences.append(len(all_sentences_untruncated))

    logger.info("Printing some statistics:")


    return features


def convert_examples_to_cognitive_features(args, model_params, examples, tokenizer,
                                 max_seq_length, max_query_length, is_training,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    # Instead of a linear list over tokens create ( max_num_sentences * max_num_words )
    features = []

    if "roberta" in args.model_type:
        # Change the start and end token
        cls_token = "<s>"
        sep_token = "</s>"

    evidence_classes = dict((y, x) for (x, y) in enumerate(model_params['classes']))

    query_lengths = []
    for example in examples:
        if example.query is not None:
            query_tokens = tokenizer.tokenize(example.query)
            query_lengths.append(len(query_tokens))
    maximum_fixed_mask_length = max(query_lengths)

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
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        all_sentences = []
        for sent in example.sentences:
            sent_tokens = [tok.strip() for tok in sent]
            sent_tokenized = []
            for token in sent_tokens:
                sub_tokens = tokenizer.tokenize(token)
                sent_tokenized += sub_tokens
            if len(sent_tokenized) < model_params["chunk_size"]:
                continue
            all_sentences.append(sent_tokenized[:args.max_sentence_len])
        all_sentences = all_sentences[:args.max_num_sentences]

        evidence_span_positions = []
        for span in example.evidence_spans:
            tok_start_position = orig_to_tok_index[span[0]]
            if span[1] < len(example.doc_toks) - 1:
                tok_end_position = orig_to_tok_index[span[1] + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            evidence_span_positions.append((tok_start_position, tok_end_position))

        # Not breaking up doc_tokens for now since its just a classification,
        # TODO: Introduce global normalization to accomodate multiple spans
        doc_span_index = 0

        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        tokens = []
        segment_ids = []
        p_mask = [] # p_mask (inverted) is turned off for CLS and Paragraph tokens only
        token_to_orig_map = {}

        # CLS (Required for classification)
        tokens.append(cls_token)
        segment_ids.append(cls_token_segment_id)
        p_mask.append(0)
        cls_index = 0

        # Query (if not Empty)
        # Only add the query if it's a QA dataset
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
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
                    tokens.append(all_doc_tokens[i])
                    segment_ids.append(sequence_b_segment_id)
                    p_mask.append(0)
                    evidence_length += 1
            paragraph_len = evidence_length
            # Add SEP token after every evidence snippet
        else:
            for i,tok in enumerate(all_doc_tokens):
                token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
                tokens.append(all_doc_tokens[i])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)
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
            p_mask = p_mask[:max_seq_length-1]
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

        # Evidence label if required
        evidence_label = np.zeros(max_seq_length)
        # window now always has max query length before actual document starts
        doc_offset = max_query_length
        # doc_offset = len(query_tokens) + 2 if requires_question_mask[os.path.basename(args.data_dir)] else 1
        for span in evidence_span_positions:
            for i in range(span[0],span[1]):
                i_alt = doc_offset + i
                if i_alt < max_seq_length:
                    evidence_label[i_alt] = 1



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
                annotation_id=example.id,
                doc_id=example.doc_id,
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
                evidence_label=evidence_label))
        unique_id += 1


    return features


def idxtobool(idx, size, device):
    V = torch.zeros(size, dtype=torch.long, device=device)
    if len(size) > 2:

        for i in range(size[0]):
            for j in range(size[1]):
                subidx = idx[i, j, :]
                V[i, j, subidx] = float(1)

    elif len(size) is 2:

        for i in range(size[0]):
            subidx = idx[i, :]
            V[i, subidx] = float(1)

    else:

        raise argparse.ArgumentTypeError('len(size) should be larger than 1')

    return V
