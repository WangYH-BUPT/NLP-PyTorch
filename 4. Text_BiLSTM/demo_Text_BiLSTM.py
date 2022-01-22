"""
Bi-LSTM  # code by Tae Hwan Jung(Jeff Jung) @graykode, modify by WangYH-BUPT
1. Python 3.6+
2. Torch 1.2.0+
"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.FloatTensor


# Corpus
sentence = (
    'GitHub Actions makes it easy to automate all your software workflows '
    'from continuous integration and delivery to issue triage and more'
)
# len(sentence) = 21
# 'Github x x x ... x' --> Actions  num = 21 - 1
# 'Github Actions x x ... x' --> makes  num = 21 - 2
words = sentence.split()
word2idx = {word: idx for idx, word in enumerate(set(words))}
idx2word = {idx: word for idx, word in enumerate(set(words))}


# Bi-LSTM Parameters
n_class = len(word2idx)  # classification problem, 不同词的个数
max_len = len(sentence.split())  # 21
batch_size = 16
n_hidden = 5


# dataset, dataloader
def make_data(sentence):
    input_batch = []
    target_batch = []
    for i in range(max_len - 1):
        input = [word2idx[n] for n in words[:(i + 1)]]
        input = input + [0] * (max_len - len(input))
        # i = 0: input = [idx(Github), 0, 0, ..., 0] (21-1个0)
        # i = 1: input = [idx(Github), idx(Actions), 0, 0, ..., 0] (21-2个0)
        # 多个0只是用来占位，也可以用其他的表示，例如: 'UNK'_idx = max_len + 1。这里可以理解成transformer中的mask
        target = word2idx[words[i + 1]]
        input_batch.append(np.eye(n_class)[input])  # 每一行是Github、Actions、...对应的one-hot编码
        target_batch.append(target)
    return torch.Tensor(input_batch), torch.LongTensor(target_batch)


input_batch, target_batch = make_data(sentence)
# input_batch: [max_len - 1, max_len, n_class]; target_batch = [1, max_len - 1]
dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


# BiLSTM
class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(batch_first=False, input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        self.fc = nn.Linear(n_hidden * 2, n_class)
    
    def forward(self, input_data):
        batch_size = input_data.shape[0]
        input_data = input_data.transpose(0, 1)  # [batch_size, max_len, n_class] --> [max_len, batch_size, n_class]
        hidden_state = torch.zeros(1*2, batch_size, n_hidden)  # [num_layer * num_directions, batch_size, n_hidden]
        cell_state = torch.zeros(1*2, batch_size, n_hidden)  # [num_layer * num_directions, batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input_data, (hidden_state, cell_state))  # [max_len, batch_size, n_hidden * 2]
        outputs = outputs[-1]  # [batch_size, n_hidden * 2]
        model = self.fc(outputs)  # [batch_size, n_class]
        return model


model = BiLSTM()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=1e-3)


# Training
for epoch in range(5000):
    for input_data, target_data in loader:
        predict_data = model(input_data)
        loss = loss_fn(predict_data, target_data)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'Loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Predict
predict = model(input_batch).data.max(1, keepdim=True)[1]
print(sentence)
print([idx2word[n.item()] for n in predict.squeeze()])
