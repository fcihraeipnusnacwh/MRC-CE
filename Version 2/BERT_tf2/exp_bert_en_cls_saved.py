import os, sys, time, ljqpy, math
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import bert_tools as bt
bt.switch_to_en()

from bert4keras.models import build_transformer_model

if 'save' in sys.argv:
	bert_path = '../tfhub/uncased_L-12_H-768_A-12'
	bert = build_transformer_model(bert_path, return_keras_model=False) 
	bert.model.save('../tfhub/bert_uncased.h5')
	sys.exit()

bert = load_model('../tfhub/bert_uncased.h5')
output = Lambda(lambda x: x[:,0], name='CLS-token')(bert.output)
output = Dense(1, activation='sigmoid')(output)
model = Model(bert.input, output)
bt.lock_transformer_layers(bert, 10)

maxlen = 150

from imdb import *
(train_text, train_label), (test_text, test_label) = GetImdbData(maxlen, 3000)
train_inputs, test_inputs = map(lambda x:bt.convert_sentences(x, maxlen), [train_text, test_text])

epochs = 3
batch_size = 32
total_steps = epochs*train_inputs[0].shape[0]//batch_size
optimizer = bt.get_suggested_optimizer(1e-4, total_steps=total_steps)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(train_inputs, train_label, epochs=epochs, batch_size=batch_size, shuffle=True, 
		  validation_data=(test_inputs, test_label))

#Epoch 1/3
#94/94 [==============================] - 32s 337ms/step - loss: 0.4417 - accuracy: 0.7877 - val_loss: 0.3099 - val_accuracy: 0.8663
#Epoch 2/3
#94/94 [==============================] - 31s 329ms/step - loss: 0.2525 - accuracy: 0.8873 - val_loss: 0.3133 - val_accuracy: 0.8740
#Epoch 3/3
#94/94 [==============================] - 33s 348ms/step - loss: 0.1613 - accuracy: 0.9293 - val_loss: 0.4119 - val_accuracy: 0.8750
