import re

patterns1 = [
    '[^。；总名称作者隶]+(是|属于)[^。；]*?一\\w[^。；，]+的([\\w、《》]+)[，。；]',  # xxx是一家xxx的（电商导购平台）
    '[^。；总名称作者隶]+(是|属于)[^。；]*?一\\w[^。；，]+?的([\\w、《》]+)[，。；]',  # xxx是一家xxx的（国内领先的电商导购平台）
    '[^。；总名称作者隶]+(是|属于)[^。；]*?[^。；，]+的一\\w([\\w、《》]+)[，。；]',

    '\\w*作为[^。，；]*?一\\w[\\w、]+的([\\w、]+)\\b',
    '\\w*作为[^。，；]*?一\\w[\\w、]+?的([\\w、]+)\\b',

    '\\w*作为[^。，；]*的([\\w、]+)\\b',
    '\\w*作为[^。，；]*?的([\\w、]+)\\b',

    '[^。；总名称作者隶]+(是|作为|属于|为)([\\w、]+)之一\\b',
    '[^。；总名称作者隶]+(是|作为|属于|为)[^。]+的([\\w、]+)之一\\b',
    '[^。；总名称作者隶]+(是|作为|属于|为)[^。]+?的([\\w、]+)之一\\b',  # xxx是xxx的（企业）之一

    '\\w*(现任|现为)(\\w+)\\b',
    '\\w*是家([\\w、]+)\\b',
    '\\w*是([\\w、]+)的一\\w\\b'  # 欧姆贝属是腕足动物门有铰纲长身贝目的一属
]

patterns2 = [
    '[^。；总名称作者隶]+(是|属于)[^。；]+的([\\w、《》]+)[，。；]',
    '[^。；总名称作者隶]+(是|属于)[^。；]+?的([\\w、《》]+)[，。；]',
]

patterns3 = [
    '[^。；总名称作者隶]+(是|属于)[^。；]*?一\\w([^。；，《》坐落]+)[，。；]',
    '(\\w{2})于.*成立',
    '(\\w{2})(\(.*\))*(位于|坐落于|创立于|创办于|始创于|创建于|正式成立于|成立于|连载于|目前连载于)'
]

patterns4 = [
    '^\\w{2,4}(（.*）)*，(\\w{1,8})\\b',
    '^\\w{2,4}(（.*）)*，(\\w{1,8})\\b',
    '^\\w{2,4}(（.*）)*，(\\w{1,8})，(\\w{1,8})\\b',
    '^\\w{2,4}(（.*）)*，(\\w{1,8})、(\\w{1,8})\\b',
    '^\\w{2,4}(（.*）)*，(\\w{1,8})，(\\w{1,8})，(\\w{1,8})\\b',
    '^\\w{2,4}(（.*）)*，(\\w{1,8})、(\\w{1,8})、(\\w{1,8})\\b',
    '^\\w{2,4}(（.*）)*，(\\w{1,8})等\\b',
    '^\\w{2,4}(（.*）)*，(\\w{1,8})等\\b',
    '^\\w{2,4}(（.*）)*，(\\w{1,8})，(\\w{1,8})等\\b',
    '^\\w{2,4}(（.*）)*，(\\w{1,8})、(\\w{1,8})等\\b',
    '^\\w{2,4}(（.*）)*，(\\w{1,8})，(\\w{1,8})，(\\w{1,8})等\\b',
    '^\\w{2,4}(（.*）)*，(\\w{1,8})、(\\w{1,8})、(\\w{1,8})等\\b',
]

patterns6 = [
    '\\w*[^称]为[\\w、，\-]*[\\w、，\-]+的([\\w、]+)\\b',
    '\\w*[^称]为[\\w、，\-]*[\\w、，\-]+?的([\\w、]+)\\b',
]

f = open('mrc-ce_result_Roberta.txt', 'r')
lines = f.readlines()
f.close()
count = 0

check_pattern = '(的|之一|一\\w|，|（|）|现为|现任|位于|坐落于|创立于|创办于|始创于|创建于|正式成立于|成立于|连载于|' \
                '目前连载于|致力于|出生|生于|生人|生\\b|年\\b|现年|\\b是\\b|\\b作为\\b|\\b属\\b|\\b属于\\b|\\b为\\b|' \
                '等\\b|人\\b|祖籍|\\b字|是|于|[a-z]|\\b男\\b|\\b女\\b|族\\b|\\b\\w\\b|原名|籍贯|和|上\\b|[0-9])'
all_patterns = [patterns1, patterns2, patterns3, patterns4, patterns6]

for i in range(len(lines)):
    line = lines[i].strip('\n')
    line_list = line.split('\t')
    text = line_list[0]
    txt = re.sub(',', '，', text)
    txt = txt.split('。')[0] + '。'
    real = line_list[1]
    pre = list()
    
    for j in range(len(all_patterns)):
        for sub_pattern in all_patterns[j]:
            pattern = re.compile(sub_pattern)
            result = pattern.findall(txt)
            if result:
                for concept in result:
                    if isinstance(concept, tuple):
                        for sub_concept in concept:
                            if sub_concept == '' or re.findall(check_pattern, sub_concept):
                                continue
                            else:
                                pre.append(sub_concept)
                    else:
                        if concept == '' or re.findall(check_pattern, concept):
                            continue
                        else:
                            pre.append(concept)
        if pre:
            pre = list(set(pre))
            break
    
    pre_str = ''
    for concept in pre:
        pre_str += concept
        pre_str += ','
    if pre_str != '':
        pre_str = pre_str[:-1]
        count += 1
    print(text)
    print(pre_str)
    print()
    f = open('baseline_Chinese.txt', 'a')
    f.write(text + '\t' + real + '\t' + pre_str + '\n')
print(count)