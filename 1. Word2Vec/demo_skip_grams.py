"""
Word2Vec, skip-gram  # code by Tae Hwan Jung(Jeff Jung) @graykode, modify by WangYH-BUPT
Paper: Distributed Representations of Words and Phrases and their Compositionality
- Python 3.6+
- Torch 1.2.0+
"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optimizer
import torch.utils.data as Data
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.FloatTensor


# Corpus
sentences = ["jack like dog", "jack like cat", "jack like animal",
             "dog cat animal", "banana apple cat dog like", "dog fish milk like",
             "dog cat animal like", "jack like apple", "apple like", "jack like banana",
             "apple banana jack movie book music like", "cat dog hate", "cat dog like"]
sentences_list = " ".join(sentences).split()  # ['jack', 'like', 'dog', 'jack' ...], len(sentences_list) = 46
vocab = set(sentences_list)  # {'jack', 'like', 'dog', 'cat' ...}, len(vocab) = 13
word2idx = {word: idx for idx, word in enumerate(vocab)}  # {'jack': 0, 'like': 1, ...}
idx2word = {idx: word for idx, word in enumerate(vocab)}  # {0: 'jack', 1: 'like', ...}
vocab_size = len(vocab)


# model parameters
window_size = 2
batch_size = 8
word_embedding_dim = 2


skip_grams = []
for center_idx in range(window_size, len(sentences_list)-window_size):  # idx = 2, 3, ..., len-2
    center_word2idx = word2idx[sentences_list[center_idx]]  # center_word2idx is the unique index of center in word2idx
    context_idx = list(range(center_idx-window_size, center_idx)) + list(range(center_idx+1, center_idx+window_size+1))
    context_word2idx = [word2idx[sentences_list[i]] for i in context_idx]

    for w in context_word2idx:
        skip_grams.append([center_word2idx, w])
        # len(skip_gram): 168 = (len(sentences_list) - window_size*2) * window_size*2 = (46 - 2*2) * 2*2


def make_data(skip_grams):
    input_data = []  # input is one-hot code
    output_data = []  # output is a class

    for center_one_hot, context_class in skip_grams:
        input_data.append(np.eye(vocab_size)[center_one_hot])
        output_data.append(context_class)
    return input_data, output_data


input_data, output_data = make_data(skip_grams)  # instantiate
input_data, output_data = torch.Tensor(input_data), torch.LongTensor(output_data)
dataset = Data.TensorDataset(input_data, output_data)
loader = Data.DataLoader(dataset, batch_size, True)


class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        self.W = nn.Parameter(torch.randn(vocab_size, word_embedding_dim).type(dtype))
        self.V = nn.Parameter(torch.randn(word_embedding_dim, vocab_size).type(dtype))

    def forward(self, training_input):  # training_input: [batch_size, vocab_size], each line is one-hot code
        hidden = torch.mm(training_input, self.W)  # [batch_size, word_embedding_dim]
        output = torch.mm(hidden, self.V)  # [batch_size, vocab_size], class_num = vocab_size
        return output


model = Word2Vec().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optim = optimizer.Adam(model.parameters(), lr=1e-3)


for epoch in range(2000):
    for i, (batch_x, batch_y) in enumerate(loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        predict = model(batch_x)
        loss = loss_fn(predict, batch_y)

        if (epoch + 1) % 1000 == 0:
            print("epoch =", epoch + 1, i, "loss =", loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()


for i, label in enumerate(vocab):
    W, WT = model.parameters()
    x,y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()
