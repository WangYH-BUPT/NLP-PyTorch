"""
TextRNN
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
sentences = ["i like dog", "i love coffee", "i hate milk"]
sentences_list = " ".join(sentences).split()
vocab = set(sentences_list)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}
n_class = len(vocab)


# Parameters
batch_size = 2
n_step = 2  # number of input words (= number of cells)
n_hidden = 5  # number of hidden units in one cell (word_embedding_dim --> hidden_dim)


# Construct datasets and dataloader
def make_data(sentences):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()  # Take each sentence as a unit, extract words
        input = [word2idx[n] for n in word[: -1]]  # Traverse the first word to the penultimate word of each sentence
        target = word2idx[word[-1]]  # the last word of each sentence

        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)
        """
        print(np.eye(5)[[0, 2]])  # 单位矩阵的第0行、第2行取出来
        [[1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0.]]
        """
    return input_batch, target_batch


input_batch, target_batch = make_data(sentences)  # input_batch: [num(sentences), n_step, n_class] (here: [3, 2, 7])
input_batch, target_batch = torch.Tensor(input_batch), torch.LongTensor(target_batch)  # !! target_batch: LongTensor !!
dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


# Construct RNN
class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)  # 7-dim(n_class) --> 5-dim(n_hidden)
        # !input_size and hidden_size are essential parameters in nn.RNN
        # !input_size is n-dim vector encoding of each word. (Here is one-hot, so 7-dim vector is represented each word)
        # !hidden_size is the hidden dimension through RNN
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, hidden, X):
        # torch要求的X是: [seq_len, batch_size, word2vec]
        # 我们的X是: [batch_size, n_step, n_class]
        X = X.transpose(0, 1)  # X: [batch_size, n_step, n_class] --> [n_step, batch_size, n_class]
        out, hidden = self.rnn(X, hidden)
        # hidden: [num_of_layers(1层RNN) * num_directions(=1), batch_size, hidden_size]
        # out: [seq_len, batch_size, hidden_size]
        out = out[-1]
        model = self.fc(out)
        return model


model = TextRNN().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Training
for epoch in range(5000):
    for input_data, output_data in loader:  # input_data: [batch_size, n_step, n_class]; output_data: [batch_size]
        input_data = input_data.to(device)
        output_data = output_data.to(device)
        hidden = torch.zeros(1, input_data.shape[0], n_hidden).to(device)
        pred = model(hidden, input_data)  # predict: [batch_size, n_class]
        loss = loss_fn(pred, output_data)

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Predict
input = [sen.split()[: 2] for sen in sentences]
hidden = torch.zeros(1, len(input), n_hidden)
predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
print([sen.split()[: 2] for sen in sentences], '-->', [idx2word[n.item()] for n in predict.squeeze()])
