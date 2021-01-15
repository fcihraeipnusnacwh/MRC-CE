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

maxlen = 150

from imdb import *
(train_text, train_label), (test_text, test_label) = GetImdbData(maxlen, 3000)
train_inputs, test_inputs = map(lambda x:bt.convert_sentences(x, maxlen), [train_text, test_text])

bert_path = '../tfhub/uncased_L-12_H-768_A-12'

from bert4keras.models import build_transformer_model
bert = build_transformer_model(bert_path, return_keras_model=False) 

output = Lambda(lambda x: x[:,0], name='CLS-token')(bert.model.output)
output = Dense(1, activation='sigmoid', kernel_initializer=bert.initializer)(output)

model = Model(bert.model.input, output)

bt.lock_transformer_layers(bert, 10)

#model.summary()

epochs = 3
batch_size = 32
total_steps = epochs*train_inputs[0].shape[0]//batch_size
optimizer = bt.get_suggested_optimizer(1e-4, total_steps=total_steps)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(train_inputs, train_label, epochs=epochs, batch_size=batch_size, shuffle=True, 
		  validation_data=(test_inputs, test_label))

#Epoch 1/4
#94/94 [==============================] - 32s 337ms/step - loss: 0.4304 - accuracy: 0.7897 - val_loss: 0.3249 - val_accuracy: 0.8550
#Epoch 2/4
#94/94 [==============================] - 34s 363ms/step - loss: 0.2796 - accuracy: 0.8840 - val_loss: 0.3118 - val_accuracy: 0.8677
#Epoch 3/4
#94/94 [==============================] - 37s 396ms/step - loss: 0.1934 - accuracy: 0.9240 - val_loss: 0.3359 - val_accuracy: 0.8700