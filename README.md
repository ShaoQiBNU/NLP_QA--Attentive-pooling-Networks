Attentive pooling Networks解读
=============================

# 一. 背景

> 本文是一个做问答对匹配的模型，主要是在向量表征和余弦函数的距离之间添加了Attention层，这样问题向量表征上就能得到答案的信息，答案向量表征上就能得到问题的信息。**如单独对 “How old are you?” 和 “How are you?” 问题的向量表征可能非常相近，但是现实可能是完全不同的两个问题。但是如果对 “How old are you?“ “I am 18.” 和 “How are you?” “I’m fine”，将问题和答案的信息都考虑进来，这样就可以很明显的区分“How old are you?”和“How are you?”，这两句话实际是两个问题。**

# 二. 模型

## (一) QA模型

> 之前的QA模型主要是两种：QA-CNN 和 QA-biLSTM，模型结构如下：

![image](https://github.com/ShaoQiBNU/NLP_QA--Attentive-pooling-Networks/blob/master/image/1.png)

> 给定一个(q,a)对，q代表question，a代表answer，分别对q和a做embedding，输入CNN或biLSTM中得到Q和A，然后做max-pooling得到向量表示 <a href="https://www.codecogs.com/eqnedit.php?latex=r_{q}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?r_{q}" title="r_{q}" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=r_{a}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?r_{a}" title="r_{a}" /></a>，最后计算二者的余弦相似度。

## (二) AP模型

> AP模型结构如图所示

![image](https://github.com/ShaoQiBNU/NLP_QA--Attentive-pooling-Networks/blob/master/image/2.png)

> 借鉴了attention机制的思想，在QA模型基础上构造了Attentive Pooling层G，

![image](https://github.com/ShaoQiBNU/NLP_QA--Attentive-pooling-Networks/blob/master/image/3.png)

> 其中Q是问题的向量表征，A是答案的向量表征，U是参数矩阵，可以通过神经网络学习得到的。然后分别做基于行的max-pooling和基于列的max-pooling，得到Attention层中重要的信息。

![image](https://github.com/ShaoQiBNU/NLP_QA--Attentive-pooling-Networks/blob/master/image/4.png)

> 这样问题向量中就会存在答案的信息，答案向量中就会存在问题的信息。然后再分别做softmax归一化之后再乘上开始问题和答案的向量表征，表示提取出看过问题和答案之后，问题和答案中的重要词语分别做的向量表征。

> 模型采用问答三元组的形式进行建模（q，a+，a-），q代表问题，a+代表正向答案，a-代表负向答案，loss函数为hinge loss：

![image](https://github.com/ShaoQiBNU/NLP_QA--Attentive-pooling-Networks/blob/master/image/5.png)

# 三. 代码

> 采用的数据为WikiQA数据集，代码来源https://github.com/Lapis-Hong/Attentive-Pooling-Networks. ，对其进行了修改和注释。

## (一) 数据集生成

> 在/data/WikiQA下，调用 data_process.py 生成训练集和测试集，并基于训练集构建词表，代码如下：

```python
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
```

## (二) 训练模型

> 在 config.py 里可以修改相关配置，然后 python train.py 即可开始训练模型。

## (三) 预测结果

> python pred.py 可以查看测试集上预测score与真实score的余弦相似度。
