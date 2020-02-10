import heapq
import math

data_path = '/Users/xiadanqing/Desktop/码上人生/推荐系统/视频点击率预估/CTR_Prediction/data/'


key2index = {'sign_by_xdq': 0}
def varFeatEncoder(key_ans):
    if key_ans == 'sign_by_xdq':
        return [0]
    for key in key_ans:
        if key not in key2index:
            key2index[key] = len(key2index)
    return list(map(lambda x: key2index[x], key_ans))

def topKeys(dic, k):
    if len(dic) >= k:
        topKeys = heapq.nlargest(k,dic.keys(),key=dic.get)
    else:
        l1 = list(dic.keys())
        topKeys = l1
    return topKeys

def catfunc(x):
    l = x.split(',')
    if l[0] == '-':
        return l[1]
    else:
        return l[0]

