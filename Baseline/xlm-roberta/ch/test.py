from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from seqeval.metrics import classification_report
from model.xlmr_for_token_classification import XLMRForTokenClassification
from utils.test_utils import add_xlmr_args, evaluate_model
from utils.data_utils import NerProcessor, create_dataset, convert_examples_to_features

from tqdm import tqdm_notebook as tqdm
from tqdm import trange

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
    parser = argparse.ArgumentParser()
    parser = add_xlmr_args(parser)

    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    processor = NerProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1  # add one for IGNORE label


    
    # preparing model configs
    hidden_size = 768 if 'base' in args.pretrained_path else 1024 # TODO: move this inside model.__init__

    device = 'cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'
    print(device)
    # creating model
    model = XLMRForTokenClassification(pretrained_path=args.pretrained_path,
                                       n_labels=num_labels, hidden_size=hidden_size,
                                       dropout_p=args.dropout, device=device)
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        model.load_state_dict(torch.load('model_dir//model.pt'))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    model.to(device)
    no_decay = ['bias', 'final_layer_norm.weight']

    
    params = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in params if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    
    # freeze model if necessary
    if args.freeze_model:
        logger.info("Freezing XLM-R model...")
        for n, p in model.named_parameters():
            if 'xlmr' in n and p.requires_grad:
                p.requires_grad = False


    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)
    

    model.to(device)

    if args.do_eval:
        if args.eval_on == "dev":
            eval_examples = processor.get_dev_examples(args.data_dir)
        elif args.eval_on == "test":
            eval_examples = processor.get_test_examples(args.data_dir)
        else:
            raise ValueError("eval on dev or test set only")
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, model.encode_word)
        
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        eval_data = create_dataset(eval_features)
        f1_score, report = evaluate_model(model, eval_data, label_list, args.eval_batch_size, device)

       
        logger.info("\n%s", report)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Writing results to file *****")
            writer.write(report)
            logger.info("Done.")


if __name__ == "__main__":
    main()
