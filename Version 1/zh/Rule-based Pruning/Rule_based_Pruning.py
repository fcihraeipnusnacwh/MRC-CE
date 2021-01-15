from collections import Counter
def compare(s, t):
    return Counter(s) == Counter(t)

def rule_check(con_ls):
    con_ls = list(set(con_ls))
    
    r = con_ls[:]
    l = len(con_ls)
    
    for i in range(l):
        if  len(con_ls[i]) <= 1  or con_ls[i][0] == '名' or con_ls[i] == '会员' or con_ls[i] == '成员' or '由' in con_ls[i] or '现任' in con_ls[i] or '之一' in con_ls[i] or '兼' in con_ls[i] or '及' in con_ls[i] or '和' in con_ls[i] or '为' in con_ls[i] or '称' in con_ls[i] or '等' in con_ls[i] or '是' in con_ls[i] or '于' in con_ls[i] or '在' in con_ls[i] or '以' in con_ls[i] or '与' in con_ls[i]:
            if con_ls[i] in r:
                ind = r.index(con_ls[i])
                r.pop(ind)
    
    for i in range(l):
        for j in range(l):
            if i == j:
                continue
            if con_ls[i].endswith(con_ls[j]):
                if con_ls[i][0] == '副' or con_ls[i][0] == '非' or con_ls[i][0] == '前':
                    if con_ls[j] in r:
                        ind = r.index(con_ls[j])
                        r.pop(ind)
    for i in range(l):
        for j in range(l):
            if i == j:
                continue
            if con_ls[i].startswith(con_ls[j]):
                if con_ls[i][-1] == '家' and con_ls[j][-1] != '家':
                    if con_ls[j] in r:
                        ind = r.index(con_ls[j])
                        r.pop(ind)
                        
    for i in range(l):
        for j in range(l):
            if i == j:
                continue
            if con_ls[i].startswith(con_ls[j]):
                if con_ls[i][-1] == '人' and con_ls[j][-1] != '人':
                    if con_ls[j] in r:
                        ind = r.index(con_ls[j])
                        r.pop(ind)
    
    for i in range(l):
        for j in range(l):
            if i == j:
                continue
            if con_ls[i].startswith(con_ls[j]):
                if con_ls[i][-1] == '剧' and con_ls[j][-1] != '剧':
                    if con_ls[j] in r:
                        ind = r.index(con_ls[j])
                        r.pop(ind)
                        
    digs = ['0','1','2','3','4','5','6','7','8','9']
    for i in range(l):
        for dig in digs:
            if dig in con_ls[i]:
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
    with open('Bert_mrc_original_result.txt', 'r', encoding="utf-8") as f:
        for data in f.readlines():
            info, raw_con, con = data.replace('\n','').split('\t')
            con = con.split(',')
            dataset[info] = [raw_con,con]
    f.close()
    k = 0
    for info in dataset.keys():
        con_ls = dataset[info][1]
        r,flag = rule_check(con_ls)
        if flag != 0:
            k = k + 1
            print(con_ls,r)
            dataset[info][1] = r
            
    with open('Bert_mrc_rule_result.txt', 'w', encoding="utf-8") as f1:
        for info in dataset.keys():

            f1.write(info)
            f1.write('\t')
            f1.write(dataset[info][0])
            f1.write('\t')
            f1.write(','.join(dataset[info][1]))
            f1.write('\n')
    f1.close()