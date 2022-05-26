"""目标：生成Voc词汇表，以及一个n*2的list表示一问一答语料"""

import os
from Voc import *

corpus_raw = 'movie_lines.txt'

def P(*args):
    return os.path.join(os.getcwd(), *args)



qa_pairs = extractSentencePairs(conversations)
import pickle

import re
import pickle
import unicodedata

MAX_LENGTH = 10  # 句子最大长度是10个词(包括EOS等特殊词)

# 把Unicode字符串变成ASCII
# 参考https://stackoverflow.com/a/518232/2809427


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    # 变成小写、去掉前后空格，然后unicode变成ascii
    s = unicodeToAscii(s.lower().strip())
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
def loadPrepareData(corpus_name):
    print("Start preparing training data ...")
    voc, pairs = readVocs(corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

voc, pairs = loadPrepareData('NANANA')
print("\npairs:")
for pair in pairs[:10]:
    print(pair)

MIN_COUNT = 3    # 阈值为3

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

pairs = trimRareWords(voc, pairs, MIN_COUNT)

with open('processed_voc_pairs', 'wb') as f:
    pickle.dump((voc, pairs), f)
    