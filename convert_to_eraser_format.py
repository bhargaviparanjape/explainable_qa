import os
import pdb
import argparse
import random
import gzip
import json
import numpy as np
import torch
from collections import namedtuple
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch import nn
import math
from spacy.lang.en import English
import bisect

from eraser.rationale_benchmark.utils import annotations_from_jsonl

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
INIT_TOKEN = "@@NULL@@"

parser = argparse.ArgumentParser()
parser.add_argument('--aspect', type=int, default=0)
parser.add_argument('--train_path', type=str,
                    default="beer/reviews.aspect0.train.txt.gz")
parser.add_argument('--dev_path', type=str,
                    default="beer/reviews.aspect0.heldout.txt.gz")
parser.add_argument('--test_path', type=str,
                    default="beer/annotations.json")
parser.add_argument('--data_path',type=str,
                    default="beer/")
parser.add_argument('--max_len', type=int, default=512,
                    help="maximum input length (cut off afterwards)")


def filereader(path):
    """read SST lines"""
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\", "")


BeerExample = namedtuple("Example", ["tokens", "scores"])
BeerTestExample = namedtuple("Example", ["tokens", "scores", "annotations"])


def beer_reader(path, aspect=-1, max_len=0):
    """
    Reads in Beer multi-aspect sentiment data
    :param path:
    :param aspect: which aspect to train/evaluate (-1 for all)
    :return:
    """
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            scores = list(map(float, parts[:5]))

            if aspect > -1:
                scores = [scores[aspect]]

            tokens = parts[5:]
            if max_len > 0:
                tokens = tokens[:max_len]
            yield BeerExample(tokens=tokens, scores=scores)


def beer_annotations_reader(path, aspect=-1):
    """
    Reads in Beer annotations from json
    :param path:
    :param aspect: which aspect to evaluate
    :return:
    """
    examples = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            tokens = data["x"]
            scores = data["y"]
            annotations = [data["0"], data["1"], data["2"],
                           data["3"], data["4"]]

            if aspect > -1:
                scores = [scores[aspect]]
                annotations = [annotations[aspect]]

            ex = BeerTestExample(
                tokens=tokens, scores=scores, annotations=annotations)
            examples.append(ex)
    return examples


def get_approximate_sentence_map(sentences, doc_tokens):
    sentence_boundries = np.concatenate(([0], np.cumsum(np.array([len(s) for s in sentences]))[:-1])) + np.array(range(len(sentences)))
    sentence_lengths = [len(s) for s in sentences]
    word_boundaries = np.concatenate(([0], np.cumsum(np.array([len(t) for t in doc_tokens]))[:-1])) + np.array(range(len(doc_tokens)))
    word_lengths = [len(w) for w in doc_tokens]
    orig_to_sentence_index = [bisect.bisect_right(sentence_boundries, word_boundaries[i]+ word_lengths[i]) for i in range(len(word_boundaries))]
    return orig_to_sentence_index



if __name__ == '__main__':
    # Let's load the data into memory.

    args = parser.parse_args()
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))


    print("Loading data")

    train_data = list(beer_reader(
        args.train_path, aspect=args.aspect, max_len=args.max_len ))
    dev_data = list(beer_reader(
        args.dev_path, aspect=args.aspect, max_len=args.max_len ))
    test_data = beer_annotations_reader(args.test_path, aspect=args.aspect)

    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))

    print("Average tokens", np.average([len(ex.tokens) for ex in train_data]))
    print("Std tokens", np.std([len(ex.tokens) for ex in train_data]))

    # Average num of rationale tokens per example
    rat_tok_arr = []
    for ex in test_data:
        annots = ex.annotations[0]
        rat_toks = 0
        for a in annots:
            rat_toks += a[1]-a[0]
        rat_tok_arr.append(rat_toks)
    print("Average tokens", np.average(rat_tok_arr))
    print("Std tokens", np.std(rat_tok_arr))

    # Average tokens : 22.592555331991953
    # Std tokens 12.456281087897018
    # 1-2 sentences seems to be the ideal length for rationales


    # structure of ERASER corpus
    # docs/ Each is a list of sentences and each sentence is tokenized
    # dict_keys(['annotation_id', 'classification', 'docids', 'evidences', 'query', 'query_type'])
    # query :  'What is the sentiment of this review?'
    # query_type : None
    # evidences
    # [[{'docid': 'negR_900.txt', 'end_sentence': -1, 'end_token': 43, 'start_sentence': -1, 'start_token': 40,
    #    'text': 'i even giggled'},
    #   {'docid': 'negR_900.txt', 'end_sentence': -1, 'end_token': 76, 'start_sentence': -1, 'start_token': 58,
    #    'text': 'something about these films causes me to lower my inhibitions and return to the saturday afternoons of my'},
    #   {'docid': 'negR_900.txt', 'end_sentence': -1, 'end_token': 115, 'start_sentence': -1, 'start_token': 106,
    #    'text': "does n't quite pass the test . sure enough"},
    #   {'docid': 'negR_900.txt', 'end_sentence': -1, 'end_token': 191, 'start_sentence': -1, 'start_token': 182,
    #    'text': 'too - cheesy - to - be - accidental'},
    #   {'docid': 'negR_900.txt', 'end_sentence': -1, 'end_token': 249, 'start_sentence': -1, 'start_token': 241,
    #    'text': 'noteworthy primarily for the mechanical manner in which'},]]

    query =  'What is the sentiment of this review?'
    train_eraser_dataset = []
    dev_eraser_dataset = []
    test_eraser_dataset = []

    doc_directory = args.data_path
    if not os.path.exists(os.path.join(doc_directory, "docs")):
        os.mkdir(os.path.join(doc_directory, "docs"))
    
    lens = []
    for doc_id, ex in enumerate(train_data,1):
        tokens = ex.tokens
        doc = nlp(" ".join(tokens))
        sentences = [sent.string.strip() for sent in doc.sents]
        lens.append(len(sentences))
        orig_to_sentence_index = get_approximate_sentence_map(sentences, tokens)
        annotation_id = "train_doc_%d.txt" % doc_id
        doc_file = open(os.path.join(args.data_path, "docs", annotation_id), "w+")
        for sent in sentences:
            doc_file.write(sent + "\n")
        doc_file.close()
        classification = ex.scores[0]
        docids = None
        evidences = [[]]
        eraser_ex = {
            "annotation_id" : annotation_id,
            'classification' : classification,
            "docids" : docids,
            "query" :  query,
            "query_type" : None,
            'evidences' : evidences,
        }
        train_eraser_dataset.append(eraser_ex)
    print("Average #sents %f" % np.average(lens))


    for doc_id, ex in enumerate(dev_data,1):
        tokens = ex.tokens
        doc = nlp(" ".join(tokens))
        sentences = [sent.string.strip() for sent in doc.sents]
        orig_to_sentence_index = get_approximate_sentence_map(sentences, tokens)
        annotation_id = "dev_doc_%d.txt" % doc_id
        doc_file = open(os.path.join(args.data_path, "docs", annotation_id), "w+")
        for sent in sentences:
            doc_file.write(sent + "\n")
        doc_file.close()
        classification = ex.scores[0]
        docids = None
        evidences = [[]]
        eraser_ex = {
            "annotation_id" : annotation_id,
            'classification' : classification,
            "docids" : docids,
            "query" :  query,
            "query_type" : None,
            'evidences' : evidences,
        }
        dev_eraser_dataset.append(eraser_ex)

    evs = []
    for doc_id, ex in enumerate(test_data,1):
        tokens = ex.tokens
        doc = nlp(" ".join(tokens))
        sentences = [sent.string.strip() for sent in doc.sents]
        orig_to_sentence_index = get_approximate_sentence_map(sentences, tokens)
        annotation_id = "test_doc_%d.txt" % doc_id
        doc_file = open(os.path.join(args.data_path, "docs", annotation_id), "w+")
        for sent in sentences:
            doc_file.write(sent + "\n")
        doc_file.close()
        classification = ex.scores[0]
        docids = None
        evidences = ex.annotations[0]
        converted_evidences = []
        for ev in evidences:
            text = " ".join(tokens[ev[0]:ev[1]])
            converted_evidences.append({
                'docid': annotation_id,
                'end_sentence': -1,
                'end_token': ev[1],
                'start_sentence': -1,
                'start_token': ev[0],
                'text' : text,
            })
        evs.append(len(converted_evidences))
        eraser_ex = {
            "annotation_id" : annotation_id,
            'classification' : classification,
            "docids" : docids,
            "query" :  query,
            "query_type" : None,
            'evidences' : [converted_evidences],
        }
        test_eraser_dataset.append(eraser_ex)
    print("Avg evidence sentences %f " % np.average(evs))
    train_eraser_file = open(os.path.join(args.data_path, "train.jsonl"), "w+")
    val_eraser_file = open(os.path.join(args.data_path, "val.jsonl"),"w+")
    test_eraser_file = open(os.path.join(args.data_path, "test.jsonl"),"w+")

    for ex in train_eraser_dataset:
        train_eraser_file.write(json.dumps(ex) + "\n")
    for ex in dev_eraser_dataset:
        val_eraser_file.write(json.dumps(ex) + "\n")
    for ex in test_eraser_dataset:
        test_eraser_file.write(json.dumps(ex) + "\n")
    train_eraser_file.close()
    val_eraser_file.close()
    test_eraser_file.close()


    # Verify whether BEER can be opened using
    test_dataset = annotations_from_jsonl(os.path.join(args.data_path, "test.jsonl"))
    train_dataset = annotations_from_jsonl(os.path.join(args.data_path, "train.jsonl"))
    val_dataset = annotations_from_jsonl(os.path.join(args.data_path, "val.jsonl"))
