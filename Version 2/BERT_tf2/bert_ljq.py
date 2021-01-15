import os, sys, time, json, ljqpy
import unicodedata, re
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def is_string(s):
	return isinstance(s, str)

def sequence_padding(inputs, length=None, padding=0):
	if length is None:
		length = max([len(x) for x in inputs])

	pad_width = [(0, 0) for _ in np.shape(inputs[0])]
	outputs = []
	for x in inputs:
		x = x[:length]
		pad_width[0] = (0, length - len(x))
		x = np.pad(x, pad_width, 'constant', constant_values=padding)
		outputs.append(x)
	return np.array(outputs)

from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
model = TFBertModel.from_pretrained('hfl/chinese-roberta-wwm-ext', from_pt=True)

inputs = tokenizer.tokenize("夏天的风")
inputs = tokenizer.convert_tokens_to_ids(['[CLS]']+inputs+['[SEP]'])
print(inputs)
inputs = np.array([inputs])
segs = np.zeros(inputs.shape)
outputs = model([inputs, segs])[0][0]
print(outputs.shape)
outs = outputs.numpy().sum(-1)
print(outs)

from bert4keras.layers import *
from bert4keras.models import build_transformer_model

#mm = build_transformer_model('../tfhub/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json', 
#							 checkpoint_path='../tfhub/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt')

from tensorflow.keras.models import *
mm = load_model('model.h5')

#mm.summary()
oo = mm([inputs, segs])[0]
outs = oo.numpy().sum(-1)
print(outs)

if __name__ == '__main__':
	print('done')