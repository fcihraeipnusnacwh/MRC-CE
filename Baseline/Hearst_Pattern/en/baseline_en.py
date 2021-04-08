import re

def proper(results):
    if len(results) == 1:
        return results[0]
    min_num = 10000
    for i in range(len(results)):
        max_num = 0
        for word in results[i]:
            if len(word) > max_num:
                max_num = len(word)
        if max_num < min_num:
            min_num = max_num
            key = i
    return results[key]


patterns1 = [
'.*(is|are|was|were|has been|have been|refer to|refers to|concerns) (one of )*(a |an |the )([\\w ,\'-]*?(\\b[\\w\'-]+\\b)) (written|found|made|spoken|held|present|native|known|left)'
]

patterns2 = [
'.*(is|are|was|were|has been|have been|refer to|refers to) (a |an |the )(form|class|kind|branch|member|part|species|example|type|synonym|case|measure|genus|name|item) of ([\\w \'-]*?\\b[\\w\'-]+\\b)( that is| that\'s| which is| who is)* (\\b\\w+ly\\b )*\\b[\\w-]+ed\\b',
'.*(is|are|was|were|has been|have been|refer to|refers to) (a |an |the )(form|class|kind|branch|member|part|species|example|type|synonym|case|measure|genus|name|item) of ([\\w \'-]*?\\b[\\w\'-]+\\b)( for | between | of | with | without | within | in | on | from | at | by | whereby | that | which | who | where | whose |\.|,)'
]
patterns3 = [
'.*(is|are|was|were|has been|have been|refer to|refers to) the most \\b[\\w\'-]+\\b ([\\w \'-]*?(\\b[\\w\'-]+\\b))( that is| that\'s| which is| who is)* (\\b\\w+ly\\b )*\\b[\\w-]+ed\\b',
'.*(is|are|was|were|has been|have been|refer to|refers to) the most \\b[\\w\'-]+\\b ([\\w \'-]*?(\\b[\\w\'-]+\\b))( for | of | with | without | within | between | in | on | from | at | by | whereby | that | which | who | where | whose |\.|,)'
]
patterns4 = [
'.*(is|are|was|were|has been|have been|refer to|refers to) the \\b[\\w\'-]+est\\b ([\\w \'-]*?(\\b[\\w\'-]+\\b))( that is| that\'s| which is| who is)* (\\b\\w+ly\\b )*\\b[\\w-]+ed\\b',
'.*(is|are|was|were|has been|have been|refer to|refers to) the \\b[\\w\'-]+est\\b ([\\w \'-]*?(\\b[\\w\'-]+\\b))( for | of | with | without | within | between | in | on | from | at | by | whereby | that | which | who | where | whose |\.|,)'
]
patterns5 = [
'.*(is|are|was|were|has been|have been|refer to|refers to|concerns) (one of )*(a |an |the )([\\w ,\'-]*?(\\b[\\w\'-]+\\b))( that is| that\'s| which is| who is)* (\\b\\w+ly\\b )*\\b[\\w-]+ed\\b',
'.*(is|are|was|were|has been|have been|refer to|refers to|concerns) (one of )*(a |an |the )([\\w ,\'-]*?(\\b[\\w\'-]+\\b))( for | of | with | without | within | between | in | on | from | at | by | whereby | that | which | who | where | whose |\.|,)'
]
patterns6 = [
'.*(is|are|was|were|has been|have been|refer to|refers to) (one of )*([\\w ,\'-]*?(\\b[\\w\'-]+\\b))( that is| that\'s| which is| who is)* (\\b\\w+ly\\b )*\\b[\\w-]+ed\\b',
'.*(is|are|was|were|has been|have been|refer to|refers to) (one of )*([\\w ,\'-]*?(\\b[\\w\'-]+\\b))( for | of | with | without | within | between | in | on | from | at | by | whereby | that | which | who | where | whose |\.|,)'
]
patterns7 = [
'.*, (one of )*(a |an |the )([\\w ,\'-]*?(\\b[\\w\'-]+\\b))( that is| that\'s| which is| who is)* (\\b\\w+ly\\b )*\\b[\\w-]+ed\\b',
'.*, (one of )*(a |an |the )([\\w ,\'-]*?(\\b[\\w\'-]+\\b))( for | of | with | without | within | between | in | on | from | at | by | whereby | that | which | who | where | whose |\.|,)'
]
patterns8 = [
'as (a |an |the )([\\w ,\'-]*?(\\b[\\w\'-]+\\b))( that is| that\'s| which is| who is)* (\\b\\w+ly\\b )*\\b[\\w-]+ed\\b',
'as (a |an |the )([\\w ,\'-]*?(\\b[\\w\'-]+\\b))( for | of | with | without | within | between | in | on | from | at | by | whereby | that | which | who | where | whose |\.|,)'
]
patterns = [patterns1, patterns2, patterns3, patterns4, patterns5, patterns6, patterns7]


f = open('mrc-ce_result_probase_easysee.txt', 'r')
lines = f.readlines()
f.close()

stop_words = ['a ', 'an ', 'the ', ',', '.',
              ' of ', ' for ', ' with ', ' in ', ' on ', ' from ', 
              ' within ', ' without ', ' by ', ' at ', ' whereby ', 
              ' which ', ' where ', ' that ', ' whose ', ' who ', ' and ', ' or ', 
              ' that is', ' that\'s', ' which is', ' who is', ' which', ' who', ' that',
              'is', 'are', 'was', 'were', 'has been', 'have been', 
              'refer to', 'refers to', 'concerns', 'now', 'different', 'similar', 'common', 'capatle'
              'and', 'one of ', 'written', 'found', 'made', 'spoken', 'held', 'present', 'native', 'known', 'left']
check_pattern = '(\\bnot\\b|\\bonce\\b|\\bfirst\\b|\\bmost\\b|\\bfor\\b|\\bmass\\b|\\bpart\\b|\\bset\\b|\\bform\\b|' \
                'known\\b|\\bseries\\b|\\btype\\b|\\bof\\b|\\bwith\\b|\\bin\\b|\\bon\\b|\\bwithout\\b|\\bwithin\\b|' \
                '\\bat\\b|\\bfrom\\b|\\bby\\b|\\ban\\b|\\ba\\b|\\bthe\\b|\\bwhereby\\b|\\bwhich\\b|\\bwhere\\b|' \
                '\\bthat\\b|\\bwhose\\b|\\bwho\\b|\\band\\b|\\bor\\b|ed$|ly $|[0-9]|ing$|est\\b|\\bto\\b|ly$|' \
                'written\\b|found\\b|made\\b|spoken\\b|held\\b|present\\b|native\\b)'

count = 0
for i in range(len(lines)):
    line = lines[i].strip('\n')
    line_list = line.split('-----')
    text = line_list[0]
    real = line_list[1]
    pre = list()
    for txt in text.split('. '):
        sub_pre = list()
        txt = txt + '.'
        for sub_pattern in patterns:
            results = list()
            for j in sub_pattern:
                pattern = re.compile(j)
                result = pattern.findall(txt)
                if result:
                    results.append(result[0])
            if results:
                result = proper(results)
                for i in range(len(result)):
                    if ',' in result[i] or re.findall(check_pattern, result[i]):
                        continue
                    else:
                        sub_pre.append(result[i])
            if sub_pre:
                pre += sub_pre
        if pre:
            break
    pre = list(set(pre))
    
    pre_str = ''
    for i in pre:
        if i not in stop_words and i != '':
            pre_str += i
            pre_str += ','
    if pre_str != '':
        pre_str = pre_str[:-1]
        count += 1
    print(text)
    print(pre_str)
    print()
    f = open('baseline_English.txt', 'a')
    f.write(text + '-----' + real + '-----' + pre_str + '\n')
print(count)    
