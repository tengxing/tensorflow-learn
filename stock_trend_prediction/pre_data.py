# -*- coding:utf-8 -*-
import re
import random
import ast
import itertools
import pickle
import numpy as np

train_data_file = './CBTest/data/cbtest_NE_train.txt'
valid_data_file = './CBTest/data/cbtest_NE_valid_2000ex.txt'


def preprocess_data(data_file, out_file):
    # stories[x][0]  tories[x][1]  tories[x][2]
    stories = []
    with open(data_file) as f:
        story = []
        for line in f:
            line = line.strip()
            if not line:
                story = []
            else:
                _, line = line.split(' ', 1)
                if line:
                    if '\t' in line:
                        q, a, _, answers = line.split('\t')
                        # tokenize
                        q = [s.strip() for s in re.split('(\W+)+', q) if s.strip()]
                        stories.append((story, q, a))
                    else:
                        line = [s.strip() for s in re.split('(\W+)+', line) if s.strip()]
                        story.append(line)

    samples = []
    for story in stories:
        story_tmp = []
        content = []
        for c in story[0]:
            content += c
        story_tmp.append(content)
        story_tmp.append(story[1])
        story_tmp.append(story[2])

        samples.append(story_tmp)

    random.shuffle(samples)
    print(len(samples))

    with open(out_file, "w") as f:
        for sample in samples:
            f.write(str(sample))
            f.write('\n')


preprocess_data(train_data_file, 'train.data')
preprocess_data(valid_data_file, 'valid.data')


# 创建词汇表
def read_data(data_file):
    stories = []
    with open(data_file) as f:
        for line in f:
            line = ast.literal_eval(line.strip())
            stories.append(line)
    return stories


stories = read_data('train.data') + read_data('valid.data')

content_length = max([len(s) for s, _, _ in stories])
question_length = max([len(q) for _, q, _ in stories])
print(content_length, question_length)

vocab = sorted(set(itertools.chain(*(story + q + [answer] for story, q, answer in stories))))
vocab_size = len(vocab) + 1
print(vocab_size)
word2idx = dict((w, i + 1) for i, w in enumerate(vocab))
pickle.dump((word2idx, content_length, question_length, vocab_size), open('vocab.data', "wb"))


# From keras 补齐
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='post', truncating='post', value=0.):
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


# 转为向量
def to_vector(data_file, output_file):
    word2idx, content_length, question_length, _ = pickle.load(open('vocab.data', "rb"))

    X = []
    Q = []
    A = []
    with open(data_file) as f_i:
        for line in f_i:
            line = ast.literal_eval(line.strip())
            x = [word2idx[w] for w in line[0]]
            q = [word2idx[w] for w in line[1]]
            a = [word2idx[line[2]]]

            X.append(x)
            Q.append(q)
            A.append(a)

    X = pad_sequences(X, content_length)
    Q = pad_sequences(Q, question_length)

    with open(output_file, "w") as f_o:
        for i in range(len(X)):
            f_o.write(str([X[i].tolist(), Q[i].tolist(), A[i]]))
            f_o.write('\n')


to_vector('train.data', 'train.vec')
to_vector('valid.data', 'valid.vec')

"""
# to_word
word2idx, content_length, question_length, _ = pickle.load(open('vocab.data', "rb"))

def get_value(dic,value):
    for name in dic:
        if dic[name] == value:
            return name

with open('train.vec') as f:
	for line in f:
		line = ast.literal_eval(line.strip())
		for word in line[0]:
			print(get_value(word2idx, word))
"""
