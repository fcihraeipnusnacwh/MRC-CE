

dataset = []
line = []
con_ls = []
con = []
flag = 0

with open('test_en_result.txt', 'r', encoding="utf-8") as f1:
    for data in f1.readlines():
        data = data.split()
        if len(data) == 0:
            st = ' '.join(line)
            st = st.replace(' ,',',')
            st = st.replace(' .','.')
            st = st.replace('( ','(')
            st = st.replace(' )',')')
            dataset.append([st,','.join(con_ls)])
            line = []
            con_ls = []
        if len(data) == 3:
            line.append(data[0])
            if flag == 0:
                if len(con) != 0:
                    con_ls.append(' '.join(con))
                    con = []
                if data[2] == 'B':
                    flag = 1
                    con.append(data[0])
            if flag == 1:
                if data[2] == 'I':
                    flag = 1
                    con.append(data[0])
                if data[2] == 'O':
                    flag = 0
f1.close()
k = 0

raw = []
with open('test_probase.txt', 'r', encoding="utf-8") as f1:
    for data in f1.readlines():
        data = data.replace('\n','').split('\t')
        raw.append(data[-1])
f1.close()

raw = raw[:1001]

wrong = []
i = 0
with open('flair_en_result.txt', 'w', encoding="utf-8") as f1:
    for ele in dataset:
        
        f1.write(ele[0])
        f1.write('\t')
        if len(ele[1]) == 0:
            wrong.append(i)
            k = k+1
        f1.write(raw[i])
        f1.write('\t')
        f1.write(ele[1])
        f1.write('\n')
        i = i+1
f1.close()

i = 0
with open('flair_en_result_easysee.txt', 'w', encoding="utf-8") as f1:
    for ele in dataset:
        f1.write(ele[0])
        f1.write('-----')
        f1.write(raw[i])
        f1.write('-----')
        f1.write(ele[1])
        f1.write('\n')
        i = i+1
f1.close()