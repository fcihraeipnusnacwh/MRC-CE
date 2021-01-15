import os, sys, time, ljqpy, math, json
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append('../BERT_tf2')
import bert_tools as bt


maxlen = 100

datadir = '../dataset/tnews_public'
label_table = {}
trains, tests = [], []
for x in ljqpy.LoadJsons(os.path.join(datadir, 'train.json')):
	trains.append( (x['label_desc'], x['sentence']+'：'+x['keywords']) )
for x in ljqpy.LoadJsons(os.path.join(datadir, 'dev.json')):
	tests.append( (x['label_desc'], x['sentence']+'：'+x['keywords']) )

for x in trains:
	if x[0] not in label_table:
		n = len(label_table) // 2
		label_table[x[0]] = n
		label_table[n] = x[0]

print(trains[:5])

train_text, test_text = map(lambda x:[z[1] for z in x], [trains, tests])
train_inputs, test_inputs = map(lambda x:bt.convert_sentences(x, maxlen), [train_text, test_text])
train_ys, test_ys = map(lambda x:np.array([label_table[z] for z in x]), map(lambda x:[z[0] for z in x], [trains, tests]))

n = len(label_table) // 2

model = bt.build_classifier(classes=n)
#bt.lock_transformer_layers(model.bert_encoder, 6)

model.summary()

epochs = 4
batch_size = 32
total_steps = epochs*train_inputs[0].shape[0]//batch_size
optimizer = bt.get_suggested_optimizer(1e-4, total_steps=total_steps)
model.compile(optimizer, 'sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_inputs, train_ys, epochs=epochs, batch_size=batch_size, 
		  shuffle=True, validation_data=(test_inputs, test_ys))

#Epoch 1/4
#1668/1668 [==============================] - 380s 228ms/step - loss: 1.1284 - accuracy: 0.6217 - val_loss: 0.9690 - val_accuracy: 0.6474
#Epoch 2/4
#1668/1668 [==============================] - 435s 261ms/step - loss: 0.8341 - accuracy: 0.6973 - val_loss: 0.9418 - val_accuracy: 0.6710
