
def get_con(sentence, padding):
    l = len(padding)
    i = 0
    con = []
    flag = 0
    s = ''
    s1 = sentence
    while i < l:
        
        if padding[i] != 'O':
            
            if flag == 1:
                s = s + s1[i]
                
                
            if flag == 0:
                s = s1[i]
                flag = 1
                
            
        if padding[i] == 'O':
                if s != '':
                    con.append(s)
                    s = ''
                flag = 0
        i = i+1
        
    return con

if __name__ == '__main__':
    
    
    dataset = []
    s = ''
    with open('test.txt', 'r', encoding="utf-8") as f:
        for data in f.readlines():
            if data == '\n':
                dataset.append(s)
                s = ''
                continue
                
            s = s + data[0]
    f.close()
    
    for i in range(len(dataset)):
        dataset[i] = dataset[i].replace('\n','')
        dataset[i] = [dataset[i]]
        
    i = 0
    with open('xlm_roberta_result.txt', 'r', encoding="utf-8") as f:
        for data in f.readlines():
            
            data = data.replace('\n','').split('\t')
            true = data[0]
            pred = data[1]
            
            dataset[i].append(true)
            dataset[i].append(pred)
            
            i = i+1
    f.close()
    
    dic = {}
    for ele in dataset:
        dic[ele[0]] = [ele[1],ele[2]]
    result = {}
    for key in dic.keys():
        true_con = get_con(key,dic[key][0])
        pred_con = get_con(key,dic[key][1])
        result[key] = [true_con, pred_con]
    
    with open('xlm_roberta_ner_result.txt', 'w', encoding="utf-8") as f:
        for key in result.keys():
            f.write(key)
            f.write('\t')
            f.write(','.join(list(set(result[key][0]))))
            f.write('\t')
            f.write(','.join(list(set(result[key][1]))))
            f.write('\n')
    f.close()
    
            
            