import re

def dataset_clean(dataset):
    dic = {}
    for ele in dataset:
        entity = ele[0]
        info = ele[2]
        l_1 = len(info)
        cons = ele[1].split(',')
        r = []
        for con in cons:
            
            pad = ['0' for i in range(l_1)]
            l_2 = len(con)
            #ls = [substr.start() for substr in re.finditer(con, info)]
            p=[match.start() for match in re.finditer(con, info)]
            try:
                start = p[0]
                end = p[0] + l_2 - 1
                start_pad = pad[:start] + ['1'] + pad[start+1:]
                end_pad = pad[:end] + ['1'] + pad[end+1:]
            
                r.append([start_pad,end_pad,con])
            except:
                continue
        question = entity+'的概念是什么?'
        dic[info] = [r, question]
    
    return dic


        
if __name__ == '__main__':
    dataset = []
    with open('sample_total.txt', 'r', encoding="utf-8") as f:
        for data in f.readlines():
            data = data.replace('\n','').split('\t')
            dataset.append(data)
            
    train_set = dataset[:80000]
    dev_set = dataset[80000:90000]
    test_set = dataset[90000:100000]
    
    train_dic = dataset_clean(train_set)
    dev_dic = dataset_clean(dev_set)
    test_dic = dataset_clean(test_set)
    
    with open('train.txt', 'w', encoding="utf-8") as f1:
        for info in train_dic.keys():
            pads,question = train_dic[info]
            for pad in pads:
                f1.write(info)
                f1.write('\t')
                f1.write(','.join(pad[0]))
                f1.write('\t')
                f1.write(','.join(pad[1]))
                f1.write('\t')
                f1.write(question)
                f1.write('\n')
    f1.close()
    
    with open('dev.txt', 'w', encoding="utf-8") as f1:
        for info in dev_dic.keys():
            pads,question = dev_dic[info]
            for pad in pads:
                f1.write(info)
                f1.write('\t')
                f1.write(','.join(pad[0]))
                f1.write('\t')
                f1.write(','.join(pad[1]))
                f1.write('\t')
                f1.write(question)
                f1.write('\n')
    f1.close()
    
    with open('test.txt', 'w', encoding="utf-8") as f1:
        for info in test_dic.keys():
            pads,question = test_dic[info]
            for pad in pads:
                f1.write(info)
                f1.write('\t')
                f1.write(','.join(pad[0]))
                f1.write('\t')
                f1.write(','.join(pad[1]))
                f1.write('\t')
                f1.write(question)
                f1.write('\n')
    f1.close()
    
    with open('test_for_check.txt', 'w', encoding="utf-8") as f1:
        for info in test_dic.keys():
            pads,question = test_dic[info]
            for pad in pads:
                f1.write(info)
                f1.write('\t')
                
                f1.write(pad[2])
                f1.write('\n')
            
    f1.close()
    
