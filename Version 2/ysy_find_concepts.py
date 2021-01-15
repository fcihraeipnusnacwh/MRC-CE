import os, sys, time, re, ljqpy

datadir = 'dataset/'
sys.path.append('BERT_tf2')
import bert_tools as bt

import tensorflow as tf
from tensorflow.keras.models import load_model
from bert4keras.models import build_transformer_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import Callback
from bert4keras.backend import keras, K
from collections import defaultdict
import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
from Rule_based_Pruning import rule_check

appear = []
with open('head_distribution_new.txt', 'r', encoding="utf-8") as f:
    for data in f.readlines():
        con,numbert = data.replace('\n','').split('\t')
        appear.append(con)
f.close()

with open('forest_model_v3.pickle','rb') as f:
    model_get = pickle.load(f)

def cover_check(con,con_ls):
    flag = 0
    for ele in con_ls:
        if con.endswith(ele):
            flag = 1
            
    return flag

bert_path = "chinese_L-12_H-768_A-12"
maxlen = 192
bert = build_transformer_model(bert_path, return_keras_model=False)
#bert = load_model('chinese_L-12_H-768_A-12')
pos = Dense(2, activation='sigmoid')(bert.output)
model = keras.models.Model(inputs=bert.input, outputs=pos)
bt.lock_transformer_layers(bert, 6)

def LoadFile(fn):
    items = {}
    for x in ljqpy.LoadCSVg(fn):
        st, ed = map(lambda z:[int(y) for y in z.split(',')], x[1:])
        if x[0] in items:
            st = [max(u,v) for u, v in zip(items[x[0]][0], st)]
            ed = [max(u,v) for u, v in zip(items[x[0]][1], ed)]
        items[x[0]] = (st, ed)
    items = list(items.items())
    return items

def Convert(items):
    N = len(items)
    X, Xseg = np.zeros((N, maxlen), dtype='int32'), np.zeros((N, maxlen), dtype='int32')
    Y = np.zeros((N, maxlen, 2))
    infos = []
    for i, item in enumerate(items):
        x1 = bt.tokenizer.tokenize('下面有哪些概念？')

        #x2 = bt.tokenizer.tokenize(item[0])
        x2 = [x for x in item[0]]
        otokens = bt.restore_token_list(item[0], x2)

        xx = (x1 + x2[1:])[:maxlen]
        seg = ([0]*len(x1) + [1]*len(x2[1:]))[:maxlen]
        offset = len(x1)
        X[i,:len(xx)] = bt.tokenizer.tokens_to_ids(xx)
        Xseg[i,:len(seg)] = seg
        st = ([0]*offset + item[1][0])[:maxlen]
        ed = ([0]*offset + item[1][1])[:maxlen]
        Y[i,:len(st),0] = st
        Y[i,:len(ed),1] = ed
        infos.append({'offset': offset, 'otokens': otokens})
    return X, Xseg, Y, infos



def GetTopSpans(tokens, rr, K=20):
    cands = defaultdict(float)
    start_indexes = sorted(enumerate(rr[:,0]), key=lambda x:-x[1])[:K]
    end_indexes = sorted(enumerate(rr[:,1]), key=lambda x:-x[1])[:K]


    for start_index, start_score in start_indexes:
        #if start_score < 0.1: continue
        if start_index >= len(tokens): continue
        for end_index, end_score in end_indexes:
            #if end_score < 0.1: continue
            if end_index >= len(tokens): continue
            if end_index < start_index: continue
            #length = end_index - start_index + 1
            #if length > 40 or length == 0: continue
            ans = ''.join(tokens[start_index:end_index+1]).strip()
            #aas = ans.split('、')
            #for aa in aas: 
                #cands[aa.strip()] += start_score * end_score / len(aas)

            #if end_index+1 < len(tokens) and tokens[start_index-1] == '《' and tokens[end_index+1] == '》': end_score *= 1.1
            #if re.sub('[^a-zA-Z0-9\u4e00-\u9fa5]', '', ans).strip() == '': end_score *= 0.6
            cands[ans] = [start_score + end_score, start_score, end_score]
    #cands = {x:y for x,y in cands.items() if len(x) > 0}
    return cands


def neg_log_mean_loss(negw=40):
    def ff(y_true, y_pred):
        eps = 1e-6
        y_true = K.cast(y_true, dtype=tf.float32)
        pos = - K.sum(y_true * K.log(y_pred+eps), 1) 
        neg = K.sum((1-y_true) * y_pred, 1) / K.maximum(eps, K.sum(1-y_true, 1))
        neg = - K.log(1 - neg + eps)
        negw = K.mean(K.sum(1-y_true, 1))
        return K.mean(pos + neg * negw)
    return ff


tests = LoadFile(os.path.join(datadir, 'test.txt'))

if 'train' in sys.argv:
    trains = LoadFile(os.path.join(datadir, 'train.txt'))
    devs = LoadFile(os.path.join(datadir, 'dev.txt'))
    train, dev, test = map(Convert, [trains, devs, tests])
   
    epochs = 2
    batch_size = 32
    total_steps = epochs*len(trains)//batch_size
    optimizer = bt.get_suggested_optimizer(1e-4, total_steps=total_steps)
    model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit([train[0], train[1]], train[2], epochs=epochs, batch_size=batch_size)
    model.save('1.h5')

if 'test' in sys.argv:
    model.load_weights('1.h5')
    model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
    test = list(map(Convert, [tests]))[0]
    pred = model([test[0][:5], test[1][:5]]).numpy()

    for pp, info, item in zip(pred, test[-1], tests):
        offset = info['offset']
        pp = pp[offset:]
        spans = GetTopSpans(info['otokens'], pp)

        result_test = []
        i = 0
        con_ls = []
        for con in spans:
            con_ls.append(con)
        for con in spans:
            key = i
            concept = con
            probability = spans[con][0]
            start_logit = spans[con][1]
            end_logit = spans[con][2]
            if concept in appear:
                appear_or_not = 1
            else:
                appear_or_not = 0
            cover_or_not = cover_check(concept,con_ls)
            label = 1
            transation = [key,concept,probability,start_logit,end_logit,appear_or_not,cover_or_not,label]
            result_test.append(transation)
            
        column=['key','concept','probability','start_logit','end_logit','appear_or_not','cover_or_not','label'] # 列表对应每列的列名
        test=pd.DataFrame(columns=column,data=result_test)
        x_test,y_test = test.iloc[:,2:-1].values,test.iloc[:,-1].values
        x_test = x_test.astype(np.float32)
        if len(x_test) == 0:
            continue

        val_predictions = model_get.predict(x_test)
        test.loc[:, 'pre_forest']  = val_predictions
    
        infer_test = {}
        infer_test[item[0]] = []
        
        for i,trans in test.iterrows():
            if trans['pre_forest'] == 0:
                continue
            else:
                infer_test[item[0]].append(trans['concept'])
    
    
        for key in infer_test.keys():
            infer_test[key], flag = rule_check(infer_test[key])
        
        print(item[0])
        print(infer_test[key])