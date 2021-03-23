from collections import Counter
def compare(s, t):
    return Counter(s) == Counter(t)

def rule_check(con_ls):
    con_ls = list(set(con_ls))
    
    r1 = con_ls[:]
    l = len(con_ls)
    
    for i in range(l):
        if '、' in con_ls[i] or '。' in con_ls[i] or '，' in con_ls[i] or '《' in con_ls[i] or '》' in con_ls[i]:
            if con_ls[i] in r1:
                ind = r1.index(con_ls[i])
                r1.pop(ind)
    
    r2 = r1[:]
    l = len(r1)
                
    for i in range(l):
        if  len(r1[i]) <= 1  or r1[i][0] == '名' or r1[i] == '会员' or r1[i] == '成员' or '由' in r1[i] or '现任' in r1[i] or '之一' in r1[i] or '兼' in r1[i] or '及' in r1[i] or '和' in r1[i] or '为' in r1[i] or '称' in r1[i] or '等' in r1[i] or '是' in r1[i] or '于' in r1[i] or '在' in r1[i] or '以' in r1[i] or '与' in r1[i]:
            if r1[i] in r2:
                ind = r2.index(r1[i])
                r2.pop(ind)
                
    r3 = r2[:]
    l = len(r2)
    for i in range(l):
        for j in range(l):
            if i == j:
                continue
            if r2[i].endswith(r2[j]):
                if r2[i][0] == '副' or r2[i][0] == '非' or r2[i][0] == '前':
                    if r2[j] in r3:
                        ind = r3.index(r2[j])
                        r3.pop(ind)
    r4 = r3[:]
    l = len(r4)
    for i in range(l):
        for j in range(l):
            if i == j:
                continue
            if r3[i].startswith(r3[j]):
                if r3[i][-1] == '家' and r3[j][-1] != '家':
                    if r3[j] in r4:
                        ind = r4.index(r3[j])
                        r4.pop(ind)
    
    r5 = r4[:]
    l = len(r5)
    for i in range(l):
        for j in range(l):
            if i == j:
                continue
            if r4[i].startswith(r4[j]):
                if r4[i][-1] == '人' and r4[j][-1] != '人':
                    if r4[j] in r5:
                        ind = r5.index(r4[j])
                        r5.pop(ind)
    
    r6 = r5[:]
    l = len(r6)
    for i in range(l):
        for j in range(l):
            if i == j:
                continue
            if r5[i].startswith(r5[j]):
                if r5[i][-1] == '剧' and r5[j][-1] != '剧':
                    if r5[j] in r6:
                        ind = r6.index(r5[j])
                        r6.pop(ind)
    
    r7 = r6[:]
    l = len(r7)
    digs = ['0','1','2','3','4','5','6','7','8','9']
    for i in range(l):
        for dig in digs:
            if dig in r6[i]:
                if r6[i] in r7:
                    ind = r7.index(r6[i])
                    r7.pop(ind)


    r = list(set(r7))
     
    if compare(r, con_ls):
        return r,0
    else:
        return r,1

if __name__ == '__main__':
    dataset = {}
    with open('mrc-ce_result_ch.txt', 'r', encoding="utf-8") as f:
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
            
    with open('mrc-ce_result_ch2.txt', 'w', encoding="utf-8") as f1:
        for info in dataset.keys():

            f1.write(info)
            f1.write('\t')
            f1.write(dataset[info][0])
            f1.write('\t')
            f1.write(','.join(dataset[info][1]))
            f1.write('\n')
    f1.close()