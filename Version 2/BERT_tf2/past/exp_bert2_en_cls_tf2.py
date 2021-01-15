import os, sys, time, ljqpy, math
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import bert_tools as bt

from imdb import *
max_seq_len = 150

model_name = 'bert-base-uncased'
from transformers import AutoTokenizer, TFBertModel

tokenizer = AutoTokenizer.from_pretrained(model_name)

def convert_sentences(sents, max_seq_len=256):
	shape = (len(sents), max_seq_len)
	input_ids = np.zeros(shape, dtype='int32')
	for ii, sent in tqdm(enumerate(sents), desc="Converting sentences"):
		tokens = tokenizer.tokenize(sent)[:max_seq_len-2]
		tokens = ['[CLS]'] + tokens + ['[SEP]']
		idlist = tokenizer.convert_tokens_to_ids(tokens)
		input_ids[ii,:len(idlist)] = idlist
	return input_ids

(train_text, train_label), (test_text, test_label) = GetImdbData(max_seq_len, 3000)
train_inputs, test_inputs = map(lambda x:convert_sentences(x, max_seq_len), [train_text, test_text])

blayer = TFBertModel.from_pretrained(model_name)

#blayer.trainable = False
#blayer.bert.embeddings.trainable = False
#for x in blayer.bert.encoder._layers[0][:10]: x.trainable = False

input_word_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)
seq_output = blayer(input_word_ids)[0]

class MyMasking(tf.keras.layers.Layer):
	def call(self, x): return x[0]
	def compute_mask(self, input, input_mask=None): return input[1]
	def compute_output_shape(self, input_shape): return input_shape[0]

mask = tf.keras.layers.Lambda(lambda x:K.greater(x, 0))(input_word_ids)
seq_output = MyMasking()([seq_output, mask])
pooled_output = tf.keras.layers.GlobalAveragePooling1D('channels_last')(seq_output)
x = tf.keras.layers.Dense(256, activation='relu')(pooled_output)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs=input_word_ids, outputs=x)

#print(train_inputs[0])

epochs = 4
batch_size = 32
total_steps = epochs*train_inputs.shape[0]//batch_size
optimizer = bt.get_suggested_optimizer(init_lr=1e-4, total_steps=total_steps)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#model.summary()

model.fit(train_inputs, train_label, epochs=epochs, batch_size=batch_size, shuffle=True, 
		  validation_data=(test_inputs, test_label))


#Epoch 1/10
#3000/3000 [==============================] - 39s 13ms/sample - loss: 0.5614 - accuracy: 0.7017 - val_loss: 0.3556 - val_accuracy: 0.8417
#Epoch 2/10
#3000/3000 [==============================] - 32s 11ms/sample - loss: 0.4161 - accuracy: 0.8197 - val_loss: 0.3697 - val_accuracy: 0.8423
#Epoch 3/10
#3000/3000 [==============================] - 32s 11ms/sample - loss: 0.3335 - accuracy: 0.8593 - val_loss: 0.3520 - val_accuracy: 0.8470