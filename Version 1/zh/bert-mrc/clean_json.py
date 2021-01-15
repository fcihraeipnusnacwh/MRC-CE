# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 15:58:14 2020

@author: yuansiyu
"""
from collections import Counter
import re
def compare(s, t):
    s_l = [e.lower() for e in s]
    t_l = [e.lower() for e in t]
    return Counter(s_l) == Counter(t_l)

import json
path = 'nbest_predictions.json'
file = open(path, "r")
fileJson = json.load(file)

pre = {}
for key in fileJson.keys():
    pre[key] = []
    ls = fileJson[key]
    for ele in ls:
        if ele['probability'] >= 7:
            ele['text'] = ele['text'].replace('《','')
            ele['text'] = ele['text'].replace('》','')
            cons = re.split(r'[。！；（？，、）]',ele['text'].strip())

            for con in cons:
                if len(con) <= 40 and len(con) > 2:
                    pre[key].append(con)
    if len(pre[key]) == 0:
        pre[key].append(fileJson[key][0]['text'])
    pre[key] = list(set(pre[key]))
    

dataset = []
with open('test_for_check_probase.txt', 'r', encoding="utf-8") as f:
    for data in f.readlines():
        info, con = data.replace('\n','').split('\t')
        dataset.append([info,con])
f.close()

r = []
i = 0
for key in pre.keys():
    r.append([dataset[i][0], dataset[i][1], pre[key]])
    i = i+1
    
dic = {}
for ele in r:
    dic[ele[0]] = [[],[]]

for ele in r:
    dic[ele[0]][0].append(ele[1])
    dic[ele[0]][1] = ele[2]

result1 = []
result2 = []
c = 0
result = []

for key in dic.keys():
    c = c+1
        #result.append([key, dic[key][0], dic[key][1]])
    result.append([key, dic[key][0], dic[key][1]])
    
with open('Bert_mrc_original_result_probase_easysee.txt', 'w', encoding="utf-8") as f1:
    for ele in result:
        f1.write(ele[0])
        f1.write('-----')
        f1.write(','.join(ele[1]))
        f1.write('-----')
        f1.write(','.join(ele[2]))
        f1.write('\n')
f1.close()
'''
with open('ent_con_probase.txt', 'w', encoding="utf-8") as f1:
    for ele in result:
        f1.write(','.join(ele[1]))
        f1.write('--------')
        f1.write(','.join(ele[2]))
        f1.write('\n')
f1.close()

c = 0
result = []
for key in dic.keys():
    if compare(dic[key][0], dic[key][1]):
        c =  c+1
        #result.append([key, dic[key][0], dic[key][1]])
        result.append([key, dic[key][0], dic[key][1]])

with open('correct_result.txt', 'w', encoding="utf-8") as f1:
    for ele in result:
        f1.write(ele[0])
        f1.write('\t')
        f1.write(','.join(ele[1]))
        f1.write('\t')
        f1.write(','.join(ele[2]))
        f1.write('\n')
f1.close()          

c = 0
result3 = []
result1 = []
result2 = []
for key in dic.keys():
    if not compare(dic[key][0], dic[key][1]):
        #result.append([key, dic[key][0], dic[key][1]])
        if set(dic[key][1]) > set(dic[key][0]):
            result2.append([key, dic[key][0], dic[key][1]])
            c = c  + 1
        else:
            result1.append([key, dic[key][0], dic[key][1]])
i = 1
result3 = result1 + result2


with open('mrc_result.txt', 'w', encoding="utf-8") as f1:
    for ele in result3:
        f1.write(ele[0])
        f1.write('\t')
        f1.write(','.join(ele[1]))
        f1.write('\t')
        f1.write(','.join(ele[2]))
        f1.write('\n')
f1.close()


with open('mrc_result1.txt', 'w', encoding="utf-8") as f1:
    for ele in result1:
        f1.write(ele[0])
        f1.write('\t')
        f1.write(','.join(ele[1]))
        f1.write('\t')
        f1.write(','.join(ele[2]))
        f1.write('\n')
f1.close()


with open('mrc_result2.txt', 'w', encoding="utf-8") as f1:
    for ele in result2:
        f1.write(ele[0])
        f1.write('\t')
        f1.write(','.join(ele[1]))
        f1.write('\t')
        f1.write(','.join(ele[2]))
        f1.write('\n')
f1.close()

'''
