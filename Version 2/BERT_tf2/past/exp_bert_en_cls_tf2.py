import os, sys, time, ljqpy, math
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
import numpy as np

import bert_tools as bt

from tensorflow.keras.models import Model

from imdb import *
max_seq_len = 150

bert_tl = bt.BERTLayer()

input_word_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)
seq_output = bert_tl.get_bert_layer()(input_word_ids)
pooled_output = bert_tl.get_pooled_output(seq_output)
x = tf.keras.layers.Dense(256, activation='relu')(pooled_output)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs=input_word_ids, outputs=x)

model.build(input_shape=(None, max_seq_len))

bert_tl.post_build_model()
bert_tl.set_trainable(11, False)

def convert_sentences(sents, max_seq_len=256):
	shape = (len(sents), max_seq_len)
	input_ids = np.zeros(shape, dtype='int32')
	for ii, sent in tqdm(enumerate(sents), desc="Converting sentences"):
		tokens = bert_tl.convert_single_sentence(sent, max_seq_len)
		idlist = bert_tl.tokenizer.convert_tokens_to_ids(tokens)
		input_ids[ii,:len(idlist)] = idlist
	return input_ids

(train_text, train_label), (test_text, test_label) = GetImdbData(max_seq_len, 30000)
train_inputs, test_inputs = map(lambda x:convert_sentences(x, max_seq_len), [train_text, test_text])

print(train_inputs[0])

epochs = 10
batch_size = 32
total_steps = epochs*train_inputs.shape[0]//batch_size
lr_scheduler, optimizer = bt.get_suggested_scheduler_and_optimizer(model, init_lr=1e-3, total_steps=total_steps)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(train_inputs, train_label, epochs=epochs, batch_size=batch_size, shuffle=True, 
		  validation_data=(test_inputs, test_label), callbacks=[lr_scheduler])


#Epoch 1/10
#3000/3000 [==============================] - 42s 14ms/sample - loss: 0.5612 - accuracy: 0.6973 - val_loss: 0.4435 - val_accuracy: 0.8200
#Epoch 2/10
#3000/3000 [==============================] - 34s 11ms/sample - loss: 0.4325 - accuracy: 0.8280 - val_loss: 0.3773 - val_accuracy: 0.8493
#Epoch 3/10
#3000/3000 [==============================] - 35s 12ms/sample - loss: 0.4067 - accuracy: 0.8333 - val_loss: 0.4374 - val_accuracy: 0.8187
#Epoch 4/10
#3000/3000 [==============================] - 35s 12ms/sample - loss: 0.4064 - accuracy: 0.8417 - val_loss: 0.4196 - val_accuracy: 0.8317