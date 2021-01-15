from collections import Counter
from string import punctuation

dicts={i:'' for i in punctuation}
punc_table=str.maketrans(dicts)

        
def compare(s, t):
    s_l = [e.lower() for e in s]
    t_l = [e.lower() for e in t]
    return Counter(s_l) == Counter(t_l)

def rule_check(info, con_ls):
    con_ls = [con for con in con_ls if len(con) > 2]
    r = con_ls[:]
    l = len(con_ls)
    
    for i in range(l):
        con = con_ls[i]
        con_word = con.split()
        if  'also' in con_word or 'called' in con_word or 'are' in con_word or 'is' in con_word:
            if con_ls[i] in r:
                ind = r.index(con_ls[i])
                r.pop(ind)
    
    for i in range(l):
        con = con_ls[i]
        con_word = con.split()
        
        if len(con_ls[i]) <= 2:
            if con_ls[i] in r:
                ind = r.index(con_ls[i])
                r.pop(ind)
        
        if len(con_word) > 30:
            if con_ls[i] in r:
                ind = r.index(con_ls[i])
                r.pop(ind)
    
    r = list(set(r))
     
    if compare(r, con_ls):
        return r,0
    else:
        return r,1

if __name__ == '__main__':
    dataset = {}
    with open('v2_Bert_mrc_forest_result_probase.txt', 'r', encoding="utf-8") as f:
        for data in f.readlines():
            info, raw_con, con = data.replace('\n','').split('\t')
            con = con.split(',')
            dataset[info] = [raw_con,con]
    f.close()
    
    with open('v2_Bert_mrc_forest_result_probase_easysee.txt', 'w', encoding="utf-8") as f1:
        for info in dataset.keys():

            f1.write(info)
            f1.write('-----')
            f1.write(dataset[info][0])
            f1.write('-----')
            f1.write(','.join(dataset[info][1]))
            f1.write('\n')
    f1.close()

    k = 0
    for info in dataset.keys():
        con_ls = dataset[info][1]
        r,flag = rule_check(info, con_ls)
        if flag != 0:
            k = k + 1
            print(dataset[info][0], con_ls, r)
            dataset[info][1] = r
            
    with open('Bert_mrc_forest_rule_result_probase_easysee.txt', 'w', encoding="utf-8") as f1:
        for info in dataset.keys():

            f1.write(info)
            f1.write('-----')
            f1.write(dataset[info][0])
            f1.write('-----')
            f1.write(','.join(dataset[info][1]))
            f1.write('\n')
    f1.close()
    
    '''
    with open('ent_con_forest_rule_probase.txt', 'w', encoding="utf-8") as f1:
        for info in dataset.keys():
            f1.write(dataset[info][0])
            f1.write('-----')
            f1.write(','.join(dataset[info][1]))
            f1.write('\n')
    f1.close()
   
    i = 0
    with open('correct_result1.txt', 'w', encoding="utf-8") as f1:
        for info in dataset.keys():

            if dataset[info][0] == ','.join(dataset[info][1]):
                i = i + 1

                f1.write(info)
                f1.write('\t')
                f1.write(dataset[info][0])
                f1.write('\t')
                f1.write(','.join(dataset[info][1]))
                f1.write('\n')
    f1.close()
    
    k = 0
    with open('diff_result1.txt', 'w', encoding="utf-8") as f1:
        for info in dataset.keys():

            if dataset[info][0] != ','.join(dataset[info][1]):
                k = k + 1

                f1.write(info)
                f1.write('\t')
                f1.write(dataset[info][0])
                f1.write('\t')
                f1.write(','.join(dataset[info][1]))
                f1.write('\n')
    f1.close()
    '''
    '''            
    for i in range(l):
        con = con_ls[i]
        con_word = con.split()
        if 'a' in con_word:
            ind = con_word.index('a')
            
            new_con_word_start = con_word[ind+1]
            try:
                temp = info.index('a' + ' ' +  new_con_word_start)
            except:
                continue
            new_info = info[temp+2:]
            new_con_word = ''
            for ele in new_info:
                if ele != ',' and ele != '.':
                    new_con_word = new_con_word + ele
                else:
                    break
                
            if con_ls[i] in r:
                ind2 = r.index(con_ls[i])
                new_con_word = new_con_word.translate(punc_table)
                r[ind2] = new_con_word
    
    
    for i in range(l):
        con = con_ls[i]
        con_word = con.split()
        if 'an' in con_word:
            ind = con_word.index('an')
            
            new_con_word_start = con_word[ind+1]
            try:
                temp = info.index('an' + ' ' +  new_con_word_start)
            except:
                info_ls = info.split()
                ind_temp = info_ls.index('an')
                temp = len(' '.join(info_ls[:ind_temp]))+1
            new_info = info[temp+3:]
            new_con_word = ''
            for ele in new_info:
                if ele != ',' and ele != '.':
                    new_con_word = new_con_word + ele
                else:
                    break
                
            if con_ls[i] in r:
                ind2 = r.index(con_ls[i])
                new_con_word = new_con_word.translate(punc_table)
                r[ind2] = new_con_word
    
    
    for i in range(l):
        con = con_ls[i]
        con_word = con.split()
        if con_word[0] == 'a' or con_word[0] == 'the':
            ind = 0
            new_con_word = ' '.join(con_word[ind+1:])
            if con_ls[i] in r:
                ind2 = r.index(con_ls[i])
                r[ind2] = new_con_word
                
    for i in range(l):
        con = con_ls[i]
        con_word = con.split()
        if con_word[0] == 'of' or con_word[0] == 'the':
            ind = 0
            new_con_word = ' '.join(con_word[ind+1:])
            if con_ls[i] in r:
                ind2 = r.index(con_ls[i])
                r[ind2] = new_con_word
     '''           