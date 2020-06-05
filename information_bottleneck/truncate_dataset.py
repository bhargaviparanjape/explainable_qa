import sys; sys.path.insert(0, "..")
from ib_utils import read_examples
import argparse
import json
from itertools import chain
from eraser.rationale_benchmark.utils import load_documents,annotations_from_jsonl
import logging,os

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', dest='data_dir', required=True)
parser.add_argument('--model_params', dest='model_params', required=True)
parser.add_argument("--split", type=str, default="val")
parser.add_argument("--truncate", default=False, action="store_true")
parser.add_argument("--max_seq_length", default=512, type=int)
parser.add_argument("--max_query_length", default=24, type=int)
parser.add_argument("--max_num_sentences", default=20, type=int)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument('--low_resource', action="store_true", default=False)
args = parser.parse_args()

# Parse model args json
with open(args.model_params, 'r') as fp:
    logging.debug(f'Loading model parameters from {args.model_params}')
    model_params = json.load(fp)

dataset = annotations_from_jsonl(os.path.join(args.data_dir, args.split + ".jsonl"))
docids = set(e.docid for e in
            chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(dataset)))))
documents = load_documents(args.data_dir, docids)
examples = read_examples(args, model_params, dataset, documents, args.split)
