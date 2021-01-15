# -*- coding: utf-8 -*-

dataset = []
with open('test_result.txt', 'r', encoding="utf-8") as f:
    for data in f.readlines():
        info, raw_con, con_temp = data.replace('\n','').split('\t')
        con_temp = eval(con_temp)
        con_ls = []
        for ele in con_temp:
            if len(ele["entity"]) > 1:
                con_ls.append(ele["entity"])
        dataset.append([info, raw_con, list(set(con_ls))])
f.close()

with open('xlnet_ner_result_ch.txt', 'w', encoding="utf-8") as f1:
    for ele in dataset:
        f1.write(ele[0])
        f1.write('\t')
        f1.write(ele[1])
        f1.write('\t')
        f1.write(','.join(ele[2]))
        f1.write('\n')
f1.close()

with open('xlnet_ner_result_easysee_ch.txt', 'w', encoding="utf-8") as f1:
    for ele in dataset:
        f1.write(ele[0])
        f1.write('-----')
        f1.write(ele[1])
        f1.write('-----')
        f1.write(','.join(ele[2]))
        f1.write('\n')
f1.close()
