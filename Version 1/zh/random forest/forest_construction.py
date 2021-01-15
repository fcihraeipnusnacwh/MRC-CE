import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def cover_check(con,con_ls):
    flag = 0
    for ele in con_ls:
        if con.endswith(ele):
            flag = 1
            
    return flag




if __name__ == '__main__':
    
    path = 'D:\\PuTTY\\mechain_learning\\nbest_predictions.json'
    file = open(path, "r")
    fileJson = json.load(file)
    
    dataset = []
    with open('test_for_check.txt', 'r', encoding="utf-8") as f:
        for data in f.readlines():
            info, con = data.replace('\n','').split('\t')
            dataset.append([info,con])
    f.close()
    
    i = 0
    
    for key in fileJson.keys():
        fileJson[key] = [dataset[i][0],fileJson[key]]
        i = i+1
    
    infer_train = {}
    with open('train_data1.txt', 'r', encoding="utf-8") as f:
        for data in f.readlines():
            info, raw_con, con = data.replace('\n','').split('\t')
            con = con.split(',')
            infer_train[info] = [raw_con,con]
    f.close()
    
    
    infer_test = {}
    with open('test_data1.txt', 'r', encoding="utf-8") as f:
        for data in f.readlines():
            info, raw_con, con = data.replace('\n','').split('\t')
            con = con.split(',')
            infer_test[info] = [raw_con,con]
    f.close()
    
    
    
    appear = []
    with open('head_distribution_new.txt', 'r', encoding="utf-8") as f:
        for data in f.readlines():
            con,numbert = data.replace('\n','').split('\t')
            appear.append(con)
    f.close()
    
 
    result=[]
    
    for key in fileJson.keys():
        info = fileJson[key][0]
        ls = fileJson[key][1]
        con_ls = []
        for ele in ls:
            con_ls.append(ele['text'])
        if info not in infer_train:
            continue
        real_label_ls = infer_train[info][1]
        for ele in ls:
            
            concepts = ele['text']
            if '】' in concepts or '【' in concepts or  '。' in concepts or '；' in concepts:
                continue
            cons = re.split(r'[，、]',concepts.strip())
            for concept in cons:
                probability = round(ele['probability'],2)
                start_logit = round(ele['start_logit'],2)
                end_logit = round(ele['end_logit'],2)
                if concept in appear:
                    appear_or_not = 1
                else:
                    appear_or_not = 0
                cover_or_not = cover_check(concept,con_ls)
                if concept in real_label_ls:
                    label = 1
                else:
                    label = 0
                
                transation = [key,concept,probability,start_logit,end_logit,appear_or_not,cover_or_not,label]
                result.append(transation)
            
 
    column=['key','concept','probability','start_logit','end_logit','appear_or_not','cover_or_not','label'] # 列表对应每列的列名
 
    train=pd.DataFrame(columns=column,data=result)
 
    #train.to_csv('train_big.csv') # 如果生成excel，可以用to_excel
    
    
    result_test=[]
    
    for key in fileJson.keys():
        info = fileJson[key][0]
        ls = fileJson[key][1]
        con_ls = []
        for ele in ls:
            con_ls.append(ele['text'])
        if info not in infer_test:
            continue
        real_label_ls = infer_test[info][1]
        for ele in ls:
            
            concepts = ele['text']
            if '】' in concepts or '【' in concepts or  '。' in concepts or '；' in concepts:
                continue
            cons = re.split(r'[，、]',concepts.strip())
            for concept in cons:
                probability = round(ele['probability'],2)
                start_logit = round(ele['start_logit'],2)
                end_logit = round(ele['end_logit'],2)
                if concept in appear:
                    appear_or_not = 1
                else:
                    appear_or_not = 0
                cover_or_not = cover_check(concept,con_ls)
                if concept in real_label_ls:
                    label = 1
                else:
                    label = 0
                
                transation = [key,concept,probability,start_logit,end_logit,appear_or_not,cover_or_not,label]
                result_test.append(transation)
 
    column=['key','concept','probability','start_logit','end_logit','appear_or_not','cover_or_not','label'] # 列表对应每列的列名
 
    test=pd.DataFrame(columns=column,data=result_test)
 
    #test.to_csv('test_big.csv') # 如果生成excel，可以用to_excel
    
    x_train,y_train = train.iloc[:,2:-1].values,train.iloc[:,-1].values
    x_test,y_test = test.iloc[:,2:-1].values,test.iloc[:,-1].values
    
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32) 
    
    
    
    forest = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)
    forest.fit(x_train, y_train)
    importances = forest.feature_importances_
    print("重要性：",importances)
    val_predictions = forest.predict(x_test)
    
    test.loc[:, 'pre_forest']  = val_predictions
    
    for key in infer_test.keys():
        infer_test[key].append([])
        
    for i,trans in test.iterrows():
        if trans['pre_forest'] == 0:
            continue
        else:
            key = trans['key']
            info = fileJson[key][0]
            
            infer_test[info][2].append(trans['concept'])
            
    with open('forest_model.pickle','wb') as f:
        pickle.dump(forest,f)
    
    with open('pre_test_forest.txt', 'w', encoding="utf-8") as f1:
        for info in infer_test.keys():
    
            f1.write(info)
            f1.write('\t')
            f1.write(infer_test[info][0])
            f1.write('\t')
            f1.write(','.join(list(set(infer_test[info][2]))))
            f1.write('\n')
            
    f1.close()
    
    '''
    log_model = LogisticRegression()
    log_model.fit(x_train, y_train)
    y_pred = log_model.predict(x_test)
    
    test.loc[:, 'pre_logistic']  = y_pred
    
    for key in infer_test.keys():
        infer_test[key].append([])
        
    for i,trans in test.iterrows():
        if trans['pre_logistic'] == 0:
            continue
        else:
            key = trans['key']
            info = fileJson[key][0]
            
            infer_test[info][3].append(trans['concept'])
    
    with open('pre_test_logistic.txt', 'w', encoding="utf-8") as f1:
        for info in infer_test.keys():
    
            f1.write(info)
            f1.write('\t')
            f1.write(infer_test[info][0])
            f1.write('\t')
            f1.write(','.join(list(set(infer_test[info][3]))))
            f1.write('\n')
            
    f1.close()
    '''
    
