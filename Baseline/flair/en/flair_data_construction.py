
dataset = []
line = []
with open('train_probase_mark.txt', 'r', encoding="utf-8") as f1:
    for data in f1.readlines():
        data = data.split()
        if len(data) == 1:
            continue
        if len(data) == 0:
            dataset.append(line)
            line = []
        if len(data) == 2:
            line.append(data)
f1.close()

dataset_small = dataset[:10000]
with open('flair_train_small_en.txt', 'w', encoding="utf-8") as f1:
    for line in dataset_small:
        for ele in line:
            f1.write(ele[0])
            f1.write(' ')
            f1.write('N')
            f1.write(' ')
            f1.write(ele[1])
            f1.write('\n')
        f1.write('\n')
f1.close()


dataset = []
line = []
with open('test_probase_mark.txt', 'r', encoding="utf-8") as f1:
    for data in f1.readlines():
        data = data.split()
        if len(data) == 1:
            continue
        if len(data) == 0:
            dataset.append(line)
            line = []
        if len(data) == 2:
            line.append(data)
f1.close()

dataset_small = dataset[:1000]
with open('flair_test_small_en.txt', 'w', encoding="utf-8") as f1:
    for line in dataset_small:
        for ele in line:
            f1.write(ele[0])
            f1.write(' ')
            f1.write('N')
            f1.write(' ')
            f1.write(ele[1])
            f1.write('\n')
        f1.write('\n')
f1.close()

dataset = []
line = []
with open('dev_probase_mark.txt', 'r', encoding="utf-8") as f1:
    for data in f1.readlines():
        data = data.split()
        if len(data) == 1:
            continue
        if len(data) == 0:
            dataset.append(line)
            line = []
        if len(data) == 2:
            line.append(data)
f1.close()
dataset_small = dataset[:1000]

with open('flair_dev_small_en.txt', 'w', encoding="utf-8") as f1:
    for line in dataset_small:
        for ele in line:
            f1.write(ele[0])
            f1.write(' ')
            f1.write('N')
            f1.write(' ')
            f1.write(ele[1])
            f1.write('\n')
        f1.write('\n')
f1.close()