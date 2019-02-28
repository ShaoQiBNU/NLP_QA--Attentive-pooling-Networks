#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong

"""This Scripts gen train format WikiQA data for AP model."""
########################## load packages  ##########################
from collections import Counter

# TODO: text normalization
########################## 生成训练集  ##########################
def gen_train(infile, outfile):
    """
    extract question answer pairs which has right answer (label 1) and gen triple
    (question, positive answer, negative answer)
    """

    ############# 生成 q 的 a和label 字典  #############
    with open(infile) as fi:
        qa_pairs = {}
        for line in fi:
            q, a, label = line.split("\t")
            label = int(label)
            if q in qa_pairs:
                qa_pairs[q].append((a, label))
            else:
                qa_pairs[q] = [(a, label)]


    ############# 将label存在1的 q 生成(q, a+, a-)，输出 #############
    with open(outfile, "w") as fo:
        for q in qa_pairs:
            pos_ans = set()
            neg_ans = set()
            for a, label in qa_pairs[q]:
                if label == 1:
                    pos_ans.add(a)
                else:
                    neg_ans.add(a)

            if pos_ans:  # has answer
                for a1 in pos_ans:
                    for a2 in neg_ans:
                        triple = [q, a1, a2]
                        fo.write('\t'.join(triple) + '\n')


########################## 生成测试集  ##########################
def gen_pred(infile, outfile):
    """Use test.txt to generate predict file (question, answer)."""
    with open(outfile, "w") as fo:
        for line in open(infile):
            q, a, l = line.strip().split("\t")
            fo.write(q+"\t"+a+"\n")


########################## 构建词表  ##########################
def build_vocab(infile, outfile, max_vocab_size=None, min_word_count=1):
    """Build vocablury file."""
    tokens = []
    for line in open(infile):
        line = ' '.join(line.strip().split('\t')).split(' ')
        for word in line:
            tokens.append(word)

    counter = Counter(tokens)
    word_count = counter.most_common(max_vocab_size - 1)  # sort by word freq.
    vocab = ['UNK', '<PAD>']  # for oov words and padding
    vocab += [w[0] for w in word_count if w[1] >= min_word_count]
    print("Vocabulary size: {}".format(len(vocab)))
    with open(outfile, 'w') as fo:
        fo.write('\n'.join(vocab))


########################## main  ##########################
if __name__ == '__main__':
    gen_train("train.txt", "train")
    gen_pred("test.txt", "pred")
    build_vocab("train", "vocab", 100000, 3)