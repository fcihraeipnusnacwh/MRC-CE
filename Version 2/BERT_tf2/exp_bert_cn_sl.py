import os, sys, time, ljqpy, math
import tensorflow as tf
from tqdm import tqdm
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import bert_tools as bt

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


maxlen = 100

datadir = '../dataset/chsner_char-level'
xys = [LoadCoNLLFormat(os.path.join(datadir, '%s.txt') % tp) for tp in ['train', 'test']]

id2y = {}
for yy in xys[0][1]:
	for y in yy: id2y[y] = id2y.get(y, 0) + 1
id2y = [x[0] for x in ljqpy.FreqDict2List(id2y)]
y2id = {v:k for k,v in enumerate(id2y)}

def convert_data(df):
	text = [' '.join(t[:maxlen]) for t in df[0]]
	label = [[0]+[y2id.get(x, 0) for x in t[:maxlen-1]] for t in df[1]]
	return text, label
(train_text, train_label), (test_text, test_label) = map(convert_data, xys)

# must post padding!
pad_func = lambda x:np.expand_dims(tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=maxlen, padding='post', truncating='post'), -1)
train_label, test_label = map(pad_func, [train_label, test_label])

train_inputs, test_inputs = map(lambda x:bt.convert_sentences(x, maxlen), [train_text, test_text])

print(train_inputs[0].shape, train_label.shape)
print(train_inputs[0][0])
print(train_inputs[1][0])


bert_path = '../tfhub/chinese_roberta_wwm_ext_L-12_H-768_A-12'

from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import Callback
from bert4keras.backend import keras, K
from bert4keras.layers import ConditionalRandomField
from bert4keras.models import build_transformer_model
from bert4keras.snippets import ViterbiDecoder, to_array
bert = build_transformer_model(bert_path, return_keras_model=False) 

output = Dense(len(y2id))(bert.model.output)
CRF = ConditionalRandomField(lr_multiplier=1000)
output = CRF(output)

model = tf.keras.models.Model(inputs=bert.model.input, outputs=output)

bt.lock_transformer_layers(bert, 8)

epochs = 2
batch_size = 32
total_steps = epochs*train_inputs[0].shape[0]//batch_size
optimizer = bt.get_suggested_optimizer(1e-4, total_steps=total_steps)
model.compile(optimizer, loss=CRF.sparse_loss, metrics=[CRF.sparse_accuracy])

#model.summary()

from seqeval.metrics import f1_score, accuracy_score, classification_report

class NamedEntityRecognizer(ViterbiDecoder):
	def recognize(self, text):
		labels = self.decode(nodes)
		return [labels]


NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])


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
		trans = K.eval(CRF.trans)
		NER.trans = trans
		pred = self.smodel.predict(self.X, batch_size=16)

		for i, yseq in enumerate(self.Y):
			labels = NER.decode(pred[i])
			yt.append([self.tags[z] for z in labels])
			yp.append([self.tags[z] for z in yseq])

		f1 = f1_score(yt, yp)
		self.best_f1 = max(self.best_f1, f1)
		accu = accuracy_score(yt, yp)
		print('\naccu: %.4f  F1: %.4f  BestF1: %.4f\n' % (accu, f1, self.best_f1))
		print(classification_report(yt, yp))

test_cb = TestCallback((test_inputs, test_label), model, id2y)
model.fit(train_inputs, train_label, epochs=epochs, batch_size=batch_size,
		  validation_data=(test_inputs, test_label), callbacks=[test_cb])

trans = K.eval(CRF.trans)
NER.trans = trans
Y = model([x[:8] for x in test_inputs]).numpy()
for ii in range(8):
	tlist = [id2y[x] for x in NER.decode(Y[ii])][1:]
	print(' '.join(['%s/%s'%x for x in zip(test_text[ii].split(), tlist)]))


#Epoch 1/2
#1584/1584 [==============================] - ETA: 0s - loss: 1.6699 - sparse_accuracy: 0.9542    
#accu: 0.9970  F1: 0.9427  BestF1: 0.9427
#
#           precision    recall  f1-score   support
#
#      ORG       0.91      0.90      0.91      1333
#      PER       0.95      0.96      0.95      1957
#      LOC       0.94      0.97      0.95      2798
#
#micro avg       0.93      0.95      0.94      6088
#macro avg       0.93      0.95      0.94      6088
#1584/1584 [==============================] - 421s 266ms/step - loss: 1.6699 - sparse_accuracy: 0.9542 - val_loss: 0.3938 - val_sparse_accuracy: 0.9581 
#Epoch 2/2
#1584/1584 [==============================] - ETA: 0s - loss: 0.2321 - sparse_accuracy: 0.9569  
#accu: 0.9973  F1: 0.9458  BestF1: 0.9458
#
#           precision    recall  f1-score   support
#
#      ORG       0.91      0.92      0.91      1318
#      PER       0.94      0.96      0.95      1953
#      LOC       0.96      0.96      0.96      2873
#
#micro avg       0.94      0.95      0.95      6144
#macro avg       0.94      0.95      0.95      6144
#1584/1584 [==============================] - 453s 286ms/step - loss: 0.2321 - sparse_accuracy: 0.9569 - val_loss: 0.3623 - val_sparse_accuracy: 0.9621 
