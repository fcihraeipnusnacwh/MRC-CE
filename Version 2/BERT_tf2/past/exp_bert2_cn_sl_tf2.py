import os, sys, time, ljqpy, math
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
import numpy as np
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import bert_tools as bt

max_seq_len = 100

def LoadCoNLLFormat(fn, tag_column=-1, has_headline=False):
	datax, datay = [], []
	tempx, tempy = [], []
	with open(fn, encoding='utf-8') as fin:
		for lln in fin:
			lln = lln.strip()
			if has_headline or lln.startswith('-DOCSTART-'):
				has_headline = False; continue
			if lln == '':
				if len(tempx) >= 1:
					datax.append(tempx); datay.append(tempy)
				tempx, tempy = [], []
			else:
				items = lln.split()
				tempx.append(items[0])
				tempy.append(items[tag_column])
	if len(tempx) >= 1:
		datax.append(tempx); datay.append(tempy)
	return datax, datay

datadir = '../dataset/chsner_char-level'
xys = [LoadCoNLLFormat(os.path.join(datadir, '%s.txt') % tp) for tp in ['train', 'test']]

id2y = {}
for yy in xys[0][1]:
	for y in yy: id2y[y] = id2y.get(y, 0) + 1
id2y = [x[0] for x in ljqpy.FreqDict2List(id2y)]
y2id = {v:k for k,v in enumerate(id2y)}

def convert_data(df):
	text = [' '.join(t[:max_seq_len]) for t in df[0]]
	label = [[0]+[y2id.get(x, 0) for x in t[:max_seq_len-1]] for t in df[1]]
	return text, label
(train_text, train_label), (test_text, test_label) = map(convert_data, xys)


model_name = 'hfl/chinese-roberta-wwm-ext'
from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained(model_name)

def convert_sentences(sents, max_seq_len=256):
	shape = (len(sents), max_seq_len)
	input_ids = np.zeros(shape, dtype='int32')
	for ii, sent in tqdm(enumerate(sents), desc="Converting sentences"):
		tokens = sent.split()[:max_seq_len-2]
		tokens = ['[CLS]'] + tokens + ['[SEP]']
		idlist = tokenizer.convert_tokens_to_ids(tokens)
		input_ids[ii,:len(idlist)] = idlist
	return input_ids

# must post padding!
pad_func = lambda x:np.expand_dims(tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_seq_len, padding='post', truncating='post'), -1)
train_label, test_label = map(pad_func, [train_label, test_label])

train_inputs, test_inputs = map(lambda x:convert_sentences(x, max_seq_len), [train_text, test_text])

blayer = TFBertModel.from_pretrained(model_name, from_pt=True)

#blayer.trainable = False
blayer.bert.embeddings.trainable = False
for x in blayer.bert.encoder._layers[0][:6]: x.trainable = False

bert_inputs = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)
seq_output = blayer(bert_inputs)[0]
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(seq_output)
crf = tfa.layers.CRF(len(y2id), name='crf_layer')
pred = crf(x)
model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)

epochs = 4
batch_size = 16
total_steps = epochs*train_inputs.shape[0]//batch_size
optimizer = bt.get_suggested_optimizer(init_lr=1e-4, total_steps=total_steps)

model.compile(optimizer, loss={'crf_layer': crf.get_loss}, metrics=[crf.get_accuracy])
model.summary()

print(train_inputs.shape, train_label.shape)

from seqeval.metrics import f1_score, accuracy_score, classification_report
class TestCallback(Callback):
	def __init__(self, XY, model, tags):
		self.X, self.Y = XY
		self.Y = np.squeeze(self.Y, -1)
		self.smodel = model
		self.tags = tags
		self.best_f1 = 0
	def on_epoch_end(self, epoch, logs = None):
		# self.model is auto set by keras
		yt, yp = [], []
		pred = self.smodel.predict(self.X, batch_size=16)
		lengths = [x.sum() for x in self.X[1]]
		for pseq, yseq, llen in zip(pred, self.Y, lengths):
			yt.append([self.tags[z] for z in pseq[1:llen-1]])
			yp.append([self.tags[z] for z in yseq[1:llen-1]])
		f1 = f1_score(yt, yp)
		self.best_f1 = max(self.best_f1, f1)
		accu = accuracy_score(yt, yp)
		print('\naccu: %.4f  F1: %.4f  BestF1: %.4f\n' % (accu, f1, self.best_f1))
		print(classification_report(yt, yp))

test_cb = TestCallback((test_inputs, test_label), model, id2y)
model.fit(train_inputs, train_label, epochs=epochs, batch_size=batch_size,
		  validation_data=(test_inputs, test_label), callbacks=[test_cb])

Y = model.predict_on_batch(test_inputs[:8])
for ii in range(8):
	tlist = [id2y[x] for x in Y[ii][1:]]
	print(' '.join(['%s/%s'%x for x in zip(test_text[ii].split(), tlist)]))


#Epoch 1/4
#50624/50658 [============================>.] - ETA: 0s - loss: 6.9069 - get_accuracy: 0.9801       
#accu: 0.9922  F1: 0.9322  BestF1: 0.9322
#
#           precision    recall  f1-score   support
#
#      LOC       0.98      0.96      0.97       117
#      PER       0.98      0.95      0.97        60
#      ORG       0.83      0.85      0.84        68
#
#micro avg       0.94      0.93      0.93       245
#macro avg       0.94      0.93      0.93       245
#50658/50658 [==============================] - 359s 7ms/sample - loss: 6.9027 - get_accuracy: 0.9801 - val_loss: 1.1134 - val_get_accuracy: 0.9963 