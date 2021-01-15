import os, sys, json, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

import bert_tools as bt

from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model

pretrain_path = '../tfhub/Chinese-GPT2_ML-1.5B-v1'
gpt2 = build_transformer_model(pretrain_path, model='gpt2_ml_decoder', return_keras_model=False) 
print('load ok!')

tic = time.time()
pre = '按概率采样函数，'
token_ids, _ = bt.tokenizer.encode(pre) 
rr = gpt2.sample_decode(token_ids, topk=5, maxlen=128, minlen=50)
rr = pre + bt.tokenizer.decode(rr)
print(rr)
print('%.3fs' % (time.time() - tic))
