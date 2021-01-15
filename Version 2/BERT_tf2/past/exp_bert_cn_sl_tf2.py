import os, sys, time, ljqpy, math
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
import numpy as np
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

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


bert_tl = bt.BERTLayer(lang='cn')

def convert_sentences(sents, max_seq_len=256):
	shape = (len(sents), max_seq_len)
	input_ids = np.zeros(shape, dtype='int32')
	for ii, sent in tqdm(enumerate(sents), desc="Converting sentences"):
		tokens = bert_tl.convert_single_sentence(sent, max_seq_len)
		idlist = bert_tl.tokenizer.convert_tokens_to_ids(tokens)
		input_ids[ii,:len(idlist)] = idlist
	return input_ids

# must post padding!
pad_func = lambda x:np.expand_dims(tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_seq_len, padding='post', truncating='post'), -1)
train_label, test_label = map(pad_func, [train_label, test_label])

train_inputs, test_inputs = map(lambda x:convert_sentences(x, max_seq_len), [train_text, test_text])


bert_inputs = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)
seq_output = bert_tl.get_bert_layer()(bert_inputs)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(seq_output)
crf = tfa.layers.CRF(len(y2id), name='crf_layer')
pred = crf(x)

model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)

model.build(input_shape=(None, max_seq_len))

bert_tl.post_build_model()
bert_tl.set_trainable(12, False)

epochs = 4
batch_size = 64
total_steps = epochs*train_inputs.shape[0]//batch_size
lr_scheduler, optimizer = bt.get_suggested_scheduler_and_optimizer(model, init_lr=1e-4, total_steps=total_steps)

model.compile('adam', loss={'crf_layer': crf.get_loss}, metrics=[crf.get_accuracy])
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
		pred = self.smodel.predict(self.X, batch_size=32)
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
		  validation_data=(test_inputs, test_label), callbacks=[test_cb, lr_scheduler])

Y = model.predict_on_batch(test_inputs[:8])
for ii in range(8):
	tlist = [id2y[x] for x in Y[ii][1:]]
	print(' '.join(['%s/%s'%x for x in zip(test_text[ii].split(), tlist)]))


#Epoch 1/4
#accu: 0.9944  F1: 0.9342  BestF1: 0.9342
#
#           precision    recall  f1-score   support
#
#      LOC       0.95      0.97      0.96       111
#      PER       1.00      1.00      1.00        58
#      ORG       0.87      0.81      0.84        75
#
#micro avg       0.94      0.93      0.93       244
#macro avg       0.94      0.93      0.93       244
#50658/50658 [==============================] - 283s 6ms/sample - loss: 3.2254 - get_accuracy: 0.9890 - val_loss: 1.1732 - val_get_accuracy: 0.9956                                    
#Epoch 2/4
#accu: 0.9966  F1: 0.9607  BestF1: 0.9607
#
#           precision    recall  f1-score   support
#
#      LOC       0.96      0.97      0.97       113
#      PER       0.98      0.98      0.98        58
#      ORG       0.93      0.93      0.93        70
#
#micro avg       0.96      0.96      0.96       241
#macro avg       0.96      0.96      0.96       241
#50658/50658 [==============================] - 285s 6ms/sample - loss: 1.2789 - get_accuracy: 0.9946 - val_loss: 0.9916 - val_get_accuracy: 0.9957     