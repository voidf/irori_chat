

import os

def P(*args):
    return os.path.join(os.getcwd(), *args)


def loadLines(fileName, fields):
    """
    # 把每一行都parse成一个dict，key是lineID、characterID、movieID、character和text
    # 分别代表这一行的ID、人物ID、电影ID，人物名称和文本。
    # 最终输出一个dict，key是lineID，value是一个dict。
    # value这个dict的key是lineID、characterID、movieID、character和text
    """
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # 抽取fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i].strip()
            lines[lineObj['lineID']] = lineObj
    return lines

lines = loadLines(
    P('movie_lines.txt'), 
    ('lineID', 'characterID', 'movieID', 'character', 'text'))

def loadConversations(fileName, lines, fields):
    """
    # 根据movie_conversations.txt文件和上输出的lines，把utterance组成对话。
    # 最终输出一个list，这个list的每一个元素都是一个dict，
    # key分别是character1ID、character2ID、movieID和utteranceIDs。
    # 分别表示这对话的第一个人物的ID，第二个的ID，电影的ID以及它包含的utteranceIDs
    # 最后根据lines，还给每一行的dict增加一个key为lines，其value是个list，
    # 包含所有utterance(上面得到的lines的value)
    """
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # 抽取fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i].strip()
            # convObj["utteranceIDs"]是一个字符串，形如['L198', 'L199']
            # 我们用eval把这个字符串变成一个字符串的list。
            lineIds = eval(convObj["utteranceIDs"])
            # 根据lineIds构造一个数组，根据lineId去lines里检索出存储utterance对象。
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations

conversations = loadConversations(
    P('movie_conversations.txt'), 
    lines, 
    ('character1ID', 'character2ID', 'movieID', 'utteranceIDs'))

def extractSentencePairs(conversations):
    """
    # 从对话中抽取句对 
    # 假设一段对话包含s1,s2,s3,s4这4个utterance
    # 那么会返回3个句对：s1-s2,s2-s3和s3-s4。
    """
    qa_pairs = []
    for conversation in conversations:
        # 遍历对话中的每一个句子，忽略最后一个句子，因为没有答案。
        for i in range(len(conversation["lines"]) - 1): 
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # 如果有空的句子就去掉 
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs

qa_pairs = extractSentencePairs(conversations)
import pickle
with open('pairs.cache', 'wb') as f:
    pickle.dump(qa_pairs, f)
print(qa_pairs[:10])