"""
NNLM(Neural Network Language Model)  # code by Tae Hwan Jung(Jeff Jung) @graykode, modify by WangYH-BUPT
Paper: A Neural Probabilistic Language Model(2003)
1. Python 3.6+
2. Torch 1.2.0+
"""

import torch
import torch.nn as nn
import torch.optim as optimizer
import torch.utils.data as Data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.FloatTensor


sentences = ['i like cat', 'i love coffee', 'i hate milk']
sentences_list = " ".join(sentences).split()  # ['i', 'like', 'cat', 'i', 'love'. 'coffee',...]
word2idx = {word: idx for idx, word in enumerate(list(set(sentences_list)))}
idx2word = {idx: word for idx, word in enumerate(list(set(sentences_list)))}
max_len = len(list(set(sentences_list)))  # 7


# dataset, dataloader
def make_data(sentences):
    input_data = []
    target_data = []

    for sen in sentences:
        sen = sen.split()  # ['i', 'like', 'cat']
        input_tmp = [word2idx[w] for w in sen[: -1]]
        target_tmp = word2idx[sen[-1]]

        input_data.append(input_tmp)
        target_data.append(target_tmp)
    return torch.LongTensor(input_data), torch.LongTensor(target_data)


input_data, target_data = make_data(sentences)
dataset = Data.TensorDataset(input_data, target_data)
loader = Data.DataLoader(dataset=dataset, batch_size=2, shuffle=True)


# NNLM parameters
word_embedding_dim = 2
n_step = 2
n_hidden = 10


# NNLM
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.word_embedding = nn.Embedding(max_len, word_embedding_dim)
        self.hidden_weight = nn.Parameter(torch.randn(n_step * word_embedding_dim, n_hidden).type(dtype))
        self.hidden_bias = nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.output_bias = nn.Parameter(torch.randn(max_len).type(dtype))
        self.input_output_weight = nn.Parameter(torch.randn(n_step * word_embedding_dim, max_len).type(dtype))
        self.output_weight = nn.Parameter(torch.randn(n_hidden, max_len).type(dtype))

    def forward(self, input_data):
        input_data = self.word_embedding(input_data)  # [batch_size, n_step] --> [batch_size, n_step, word_embedding_dim]
        input_data = input_data.view(-1, n_step * word_embedding_dim)  # [batch_szie, n_step * word_embedding_dim]
        hidden_out = torch.tanh(self.hidden_bias + torch.mm(input_data, self.hidden_weight))  # [batch_size, n_hidden]
        output = self.output_bias + torch.mm(input_data, self.input_output_weight) + torch.mm(hidden_out, self.output_weight)
        return output


model = NNLM()
optim = optimizer.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


# Training
for epoch in range(5000):
    for batch_x, batch_y in loader:
        predict_data = model(batch_x)
        loss = criterion(predict_data, batch_y)

        if (epoch + 1) % 1000 == 0:
            print(epoch + 1, loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()


# Predict
pred = model(input_data).max(1, keepdim=True)[1]
print([idx2word[idx.item()] for idx in pred.squeeze()])
