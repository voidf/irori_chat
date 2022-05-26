"""目标：生成Voc词汇表，以及一个n*2*length的list表示一问一答语料"""

import json
import pickle
import unicodedata
import re
import os
from Voc import *
import jieba

from config import *

def P(*args):
    return os.path.join(os.getcwd(), *args)

Pmultispace_emitter = re.compile(r"\s+")
Pinvalid_character = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
def fil(s: str) -> list:
    # s = Pinvalid_character.sub(' ', s)
    s = Pmultispace_emitter.sub(r" ", s).strip()
    return jieba.lcut(s)

def get_raw():

    with open(HUANGJI_SOURCE, 'r', encoding='utf-8') as f:
        content = f.read().split('\nE\n')[1:]

        qa_pairs = []
        for p, i in enumerate(content):
            if i:
                a, b = i.split('\nM ')
                a = fil(a[2:].strip())
                b = fil(b.strip())
                qa_pairs.append([a, b])
            else:
                print(p)
            if p % 1000 == 0:
                print(f'{p}/{len(content)}')
    with open('xiaohuangji.cache', 'wb') as f:
        pickle.dump(qa_pairs, f)

# get_raw()

def get_db_export():
    print('filtering from db exported...')
    with open('../conversationS4_312949.json', 'r', encoding='utf-8') as f:
        j = json.load(f)
    return [[fil(x['q']), fil(x['a'])] for x in j]

def read_cache():
    with open('xiaohuangji.cache', 'rb') as f:
        return pickle.load(f)
    

def save_qa_to_db(qa_pairs: list):
    from mongoengine import Document
    from mongoengine.fields import StringField
    from mongoengine import connect
    from asyncable import Asyncable
    import asyncio
    # connect('mongodb://localhost:27017')

    class Conversation(Document, Asyncable):
        q = StringField()
        a = StringField()

    async def save_main():
        for p, (q, a) in enumerate(qa_pairs):
            asyncio.create_task(Conversation(q=q, a=a).asave())
            if p % 1000 == 0:
                print(f'{p}/{len(qa_pairs)}')
    asyncio.run(save_main())

# save_qa_to_db(qa_pairs)


# 把Unicode字符串变成ASCII
# 参考https://stackoverflow.com/a/518232/2809427



def normalizeString(s):
    # 变成小写、去掉前后空格，然后unicode变成ascii
    # s = unicodeToAscii(s.lower().strip())
    # 在标点前增加空格，这样把标点当成一个词
    s = re.sub(r"([.!?])", r" \1", s)
    # 字母和标点之外的字符都变成空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    # 因为把不用的字符都变成空格，所以可能存在多个连续空格
    # 下面的正则替换把多个空格变成一个空格，最后去掉前后空格
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# 读取问答句对并且返回Voc词典对象


def readVocs(corpus_name):
    print("Reading lines...")
    # 每行用tab切分成问答两个句子，然后调用normalizeString函数进行处理。

    pairs = [[normalizeString(s) for s in l] for l in qa_pairs]
    voc = Voc(corpus_name)
    return voc, pairs


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# 过滤太长的句对


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# 使用上面的函数进行处理，返回Voc对象和句对的list


def loadPrepareData(corpus_name, pairs):
    print("Start preparing training data ...")
    voc = Voc(corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    # pairs = filterPairs(pairs)
    # print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        for qw in pair[0]:
            voc.addWord(qw)
        for aw in pair[1]:
            voc.addWord(aw)
    print("Counted words:", voc.num_words)
    return voc




def trimRareWords(voc, pairs, MIN_COUNT):
    # 去掉voc中频次小于3的词
    voc.trim(MIN_COUNT)
    # 保留的句对
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # 检查问题
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # 检查答案
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # 如果问题和答案都只包含高频词，我们才保留这个句对
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs),
                                                                len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

if __name__ == '__main__':
    # pairs = read_cache()
    pairs = get_db_export()
    print('prefilter done.')

    voc = loadPrepareData('huangji', pairs)

    # pairs = trimRareWords(voc, pairs, MIN_COUNT)

    with open('processed_voc_pairs', 'wb') as f:
        pickle.dump((voc, pairs), f)
