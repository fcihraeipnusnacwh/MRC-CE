import os, sys, time, ljqpy, math
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_name = 'gpt2'
from transformers import TFGPT2Model, TFGPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

#model.summary()

input_ids = tokenizer.encode('My pussy loves your cock', return_tensors='tf')

beam_output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

#sample_output = model.generate(input_ids, do_sample=True, top_k=50, top_p=0.95, min_length=20, max_length=100, early_stopping=True)
#print(tokenizer.decode(sample_output[0], skip_special_tokens=True))


