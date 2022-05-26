from depend.luong_attn_decoder_rnn import LuongAttnDecoderRNN
from depend.encoder_rnn import EncoderRNN
import itertools
import os
import re
import pickle
import unicodedata
import torch
import random
from depend.Voc import *
from depend.greedy_search_decoder import GreedySearchDecoder
# from liandan_functions import *
from prework import fil
import torch.nn.functional as F

from config import (
    small_batch_size,
    MAX_LENGTH,
    )



from depend.device import dev as device

# 把句子的词变成ID
from depend.funcs import (
    indexesFromSentence,
    zeroPadding,
    binaryMatrix,
    inputVar,
    outputVar,
    batch2TrainData,
    maskNLLLoss,
    train,
    trainIters,
    evaluate,
    initialize
    
)


# l是多个长度不同句子(list)，使用zip_longest padding成定长，长度为最长句子的长度。




def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # 得到用户终端的输入
            input_sentence = input('> ')
            # 是否退出
            if input_sentence == 'q' or input_sentence == 'quit':
                break
            # 句子归一化
            input_sentence = fil(input_sentence)
            # 生成响应Evaluate sentence
            output_words = evaluate(
                encoder, decoder, searcher, voc, input_sentence)
            # 去掉EOS后面的内容
            words = []
            for word in output_words:
                if word == EOS_TOKEN:
                    break
                elif word != PAD_TOKEN:
                    words.append(word)
            print('Bot:', ' '.join(words))

        except KeyError:
            print("Error: Encountered unknown word.")


if __name__ == '__main__':
    with open('processed_voc_pairs', 'rb') as f:
        voc, pairs = pickle.load(f)

    encoder, decoder, searcher = initialize(voc, pairs)
    evaluateInput(encoder, decoder, searcher, voc)
