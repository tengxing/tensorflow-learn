#coding=utf-8
import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import nltk
from nltk.tokenize import word_tokenize


"""
'I'm super man'
tokenize:
['I', ''m', 'super','man' ]
"""
from nltk.stem import WordNetLemmatizer

"""
词形还原(lemmatizer)，即把一个任何形式的英语单词还原到一般形式，与词根还原不同(stemmer)，后者是抽取一个单词的词根。
"""

pos_file = 'data/pos.txt'
neg_file = 'data/neg.txt'

# 创建词汇表
def create_lexicon(pos_file, neg_file):
    lex = []

    # 读取文件
    def process_file(f):
        with open(pos_file, 'r') as f:
            lex = []
            lines = f.readlines()
            # print(lines)
            for line in lines:
                words = word_tokenize(line.lower())
                lex += words
            return lex

    lex += process_file(pos_file)
    lex += process_file(neg_file)
    # print(len(lex))
    lemmatizer = WordNetLemmatizer()
    lex = [lemmatizer.lemmatize(word) for word in lex]  # 词形还原 (cats->cat)

    word_count = Counter(lex)
    # print(word_count)
    # {'.': 13944, ',': 10536, 'the': 10120, 'a': 9444, 'and': 7108, 'of': 6624, 'it': 4748, 'to': 3940......}
    # 去掉一些常用词,像the,a and等等，和一些不常用词; 这些词对判断一个评论是正面还是负面没有做任何贡献
    lex = []
    for word in word_count:
        if word_count[word] < 2000 and word_count[word] > 20:  # 这写死了，好像能用百分比
            lex.append(word)  # 齐普夫定律-使用Python验证文本的Zipf分布 http://blog.topspeedsnail.com/archives/9546
    return lex


lex = create_lexicon(pos_file, neg_file)
with open('lex.pickle', 'wb') as f:
	pickle.dump(lex, f)

# lex里保存了文本中出现过的单词。

# 把每条评论转换为向量, 转换原理：
# 假设lex为['woman', 'great', 'feel', 'actually', 'looking', 'latest', 'seen', 'is'] 当然实际上要大的多
# 评论'i think this movie is great' 转换为 [0,1,0,0,0,0,0,1], 把评论中出现的字在lex中标记，出现过的标记为1，其余标记为0
def normalize_dataset(lex):
    dataset = []

    # lex:词汇表；review:评论；clf:评论对应的分类，[0,1]代表负面评论 [1,0]代表正面评论
    def string_to_vector(lex, review, clf):
        words = word_tokenize(line.lower())
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大
        return [features, clf]

    with open(pos_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [1, 0])  # [array([ 0.,  1.,  0., ...,  0.,  0.,  0.]), [1,0]]
            dataset.append(one_sample)
    with open(neg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [0, 1])  # [array([ 0.,  0.,  0., ...,  0.,  0.,  0.]), [0,1]]]
            dataset.append(one_sample)

    print(len(dataset))
    return dataset


dataset = normalize_dataset(lex)
random.shuffle(dataset)

#把整理好的数据保存到文件，方便使用。到此完成了数据的整理工作
with open('save.pickle', 'wb') as f:
	pickle.dump(dataset, f)
