# Conciseness in Rationale Extraction with an Information Bottleneck

This repository contains code for the paper [An Information Bottleneck Approach for Controlling Conciseness in Rationale Extraction](https://arxiv.org/abs/2005.00652) by Bhargavi Paranjape, Mandar Joshi, John Thickstun, Hannaneh Hajishirzi, Luke Zettlemoyer.
To run the code follow these instructions:

### Installing Dependencies 
To run this code, you'll need to install the libraries in `requirements.txt` : `pip install -r requirements.txt`

### Downloading Datasets 
To download `ERASER` : `./download_data.sh`
To download Beer review dataset : `./download_beer.sh`
To convert Beer dataset into ERASER format : `{0}` is replaced with absolute path to the main code directory.
```
python convert_to_eraser_format.py \
    --aspect 0
    --train_path ${0}/data/beer/reviews.aspect0.train.txt.gz \
    --dev_path ${0}/data/beer/reviews.aspect0.heldout.txt.gz \
    --test_path ${0}/research@uw/explainable_qa/data/beer/annotations.json \
```
To compress BoolQ documents, run for splits train,test and val:
```
cd information_bottleneck
python truncate_dataset.py --data_dir ../data/boolq --split X --model_params ../params/gated_truefalse.json --truncate --max_num_sentences 25
```

To compress Evidence Inference documents, run for splits train,test and val:
```
cd information_bottleneck
python truncate_dataset.py --data_dir ../data/evidence_inference --split test --model_params ../params/gated_evidence.json --truncate --max_num_sentences 20
```

### Running Code
To run the Sparse prior approach on dataset X, run the appropriate `run_sent_X.sh` file. For example, to run our code Fever dataset, set {0} as the absolute path to the main code directory 
```
cd information_bottleneck
./run_sent_fever.sh {0}
```

### Model details
Some useful flags and options to run our model and baselines are explained below. 
In shell scripts the following arguments can be toggled:
- `--per_gpu_train_batch_size`: batch size for training on one GPU
- `--per_gpu_eval_batch_size`: batch size for evaluating on one GPU
- `--data_dir`: Path to directory created when downloading data
- `--output_dir`: Path to output directory to checkpoint and save trained model 
- `--model_params`: Path to `.json` file which has hyperparameter configuration
- `--model_type` : Type of model in [distilbert_gated_sent, bert_gated_sent]
- `--max_num_sentences` : Avg number of sentences in each example (NS)
- `--max_query_length` : Fixed query length 

Hyperparameters in `paramss/*.json` files can be toggled:
- `beta`: Information bottleneck trade-off parameter
- `beta_norm` : Lgrangian multiplier on norm-regularization losses
- `threshold` : Sparsity parameter $\pi$
- `distribution` : Choose between [binary, kumaraswamy]


### Contact

For questions, contact [Bhargavi Paranjape](https://bhargaviparanjape.github.io/) or raise issues.
