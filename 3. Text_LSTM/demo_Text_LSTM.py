"""
TextLSTM
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
char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']  # ['a', 'b', 'c', ...]
""" 相当于:
sentences = ['abcdefghijklmnopqrstuvwxyz']
sentences_list = ' '.join(sentences).split()
"""
word2idx = {word: idx for idx, word in enumerate(char_arr)}
idx2word = {idx: word for idx, word in enumerate(char_arr)}
n_class = len(word2idx)  # number of class(= number of vocab = 26)
seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']


# TextLSTM Parameters
n_step = len(seq_data[0]) - 1  # seq_data[0]: make
n_hidden = 128


# dataset, dataloader
def make_data(seq_data):
    input_batch, target_batch = [], []
    for seq in seq_data:
        input = [word2idx[n] for n in seq[:-1]]  # 'm', 'a' , 'k' is input
        target = word2idx[seq[-1]]  # 'e' is target
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)
    # print(np.shape(input_batch))  # [10, 3, 26]: [len(seq_data), n_step, one_hot(char_arr)]
    # print(target_batch)  # [10, 1]: don't need ont-hot code
    return torch.Tensor(input_batch), torch.LongTensor(target_batch)


input_batch, target_batch = make_data(seq_data)
dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset=dataset, batch_size=3, shuffle=True)


# Construct LSTM
class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)  # 1 layer LSTM, so num_layer = num_directions = 1
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, input_data):
        batch_size = input_data.shape[0]
        input_lstm = input_data.transpose(0, 1)  # [batch_size, n_step, n_class] --> [n_step, batch_size, n_class]
        hidden_state = torch.zeros(1, batch_size, n_hidden)  # [num_layers * num_directions, batch_size, n_hidden]
        cell_state = torch.zeros(1, batch_size, n_hidden)  # [num_layers * num_directions, batch_size, n_hidden]
        outputs, (_, _) = self.lstm(input_lstm, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        TextLSTM_model = self.fc(outputs)
        return TextLSTM_model


model = TextLSTM()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Training
for epoch in range(1000):
    for input_data, target_data in loader:
        predict_data = model(input_data)
        loss = loss_fn(predict_data, target_data)
        if (epoch + 1) % 100 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Predict
inputs = [sen[:3] for sen in seq_data]
predict = model(input_batch).data.max(1, keepdim=True)[1]
print(inputs, '-->', [idx2word[n.item()] for n in predict.squeeze()])
