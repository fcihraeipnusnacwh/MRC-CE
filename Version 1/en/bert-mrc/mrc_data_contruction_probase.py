import re
from string import punctuation

def dataset_clean(dataset):
    dic = {}
    k = 0
    for ele in dataset:
        entity = ele[0]
        text = ele[1]
        '''
        dicts={i:'' for i in punctuation}
        punc_table=str.maketrans(dicts)
        text=text.translate(punc_table)
        '''

        l_1 = len(text)
        cons = list(set(ele[2].split(',')))
        r = []
        for con in cons:
            con_temp = con
            #pad = ['0' for i in range(l_1)]
            l_2 = len(con_temp)
            if l_2 == 0:
                continue
            if l_2 == 1:
                if con_temp not in text:
                    continue

                
            #ls = [substr.start() for substr in re.finditer(con, info)]
            p = -1
            a = [m.start() for m in re.finditer(con_temp.lower(),text.lower())]
            if len(a) > 0:            
                p = a[0]


            
            if p != -1:
                start = p
                r.append([start, l_2, con])
            

            
        question = 'what is the ' + entity + ' concept?'
        if len(r) == 0:
            continue
        else:
            dic[text] = [r, question, ele[1]]
    
    return dic


        
if __name__ == '__main__':
    train_set = []
    dev_set = []
    test_set = []
    dataset = []
    with open('clean_result_large.txt', 'r', encoding="utf-8") as f:
        for data in f.readlines():
            data = data.replace('\n','').replace('(','').replace(')','').split('\t')
            dataset.append(data)
    f.close()
   
    train_set = dataset[:10000]
    dev_set = dataset[10000:13500]
    test_set = dataset[13500:]
    
    train_dic = dataset_clean(train_set)
    dev_dic = dataset_clean(dev_set)
    test_dic = dataset_clean(test_set)
    i = 0
    
    

    with open('train_mrc_probase.txt', 'w', encoding="utf-8") as f1:
        for info in train_dic.keys():
            r,question,text = train_dic[info]
            for ele in r:
                f1.write(info)
                f1.write('\t')
                f1.write(str(ele[0]))
                f1.write('\t')
                f1.write(str(ele[1]))
                f1.write('\t')
                f1.write(ele[2])
                f1.write('\t')
                f1.write(question)
                f1.write('\n')
    f1.close()
    
    with open('dev_mrc_probase.txt', 'w', encoding="utf-8") as f1:
        for info in dev_dic.keys():
            r,question,text = dev_dic[info]
            for ele in r:
                f1.write(info)
                f1.write('\t')
                f1.write(str(ele[0]))
                f1.write('\t')
                f1.write(str(ele[1]))
                f1.write('\t')
                f1.write(ele[2])
                f1.write('\t')
                f1.write(question)
                f1.write('\n')
    f1.close()
    k1 = 0
    with open('test_mrc_probase.txt', 'w', encoding="utf-8") as f1:
        for info in test_dic.keys():
            r,question,text = test_dic[info]
            for ele in r:
                f1.write(info)
                f1.write('\t')
                f1.write(str(ele[0]))
                f1.write('\t')
                f1.write(str(ele[1]))
                f1.write('\t')
                f1.write(ele[2])
                f1.write('\t')
                f1.write(question)
                f1.write('\n')
    f1.close()
    
    k2 = 0
    with open('test_for_check_probase.txt', 'w', encoding="utf-8") as f1:
        for info in test_dic.keys():
            pads,question,text = test_dic[info]
            for pad in pads:
                f1.write(text)
                f1.write('\t')
                
                f1.write(pad[2])
                f1.write('\n')
                k2 = k2+1
            
    f1.close()
