"""
Text_CNN  # code by Tae Hwan Jung(Jeff Jung) @graykode, modify by WangYH-BUPT
Paper: Convolutional Neural Networks for Sentence Classification (2014)
1. Python 3.6+
2. Torch 1.2.0+
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.FloatTensor


# Corpus
sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.
sentences_list = " ".join(sentences).split()
vocab = set(sentences_list)
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for idx, word in enumerate(vocab)}
vocab_size = len(vocab)


# TextCNN Parameters
word_embedding_dim = 2
sequence_length = len(sentences[0])
num_class = len(set(labels))  # 2(0 or 1)
batch_size = 3


# dataset, dataloader
def make_data(sentences, labels):
    inputs, targets = [], []
    for sen in sentences:
        inputs.append([word2idx[n] for n in sen.split()])

    for lab in labels:
        targets.append(lab)
    return torch.LongTensor(inputs), torch.LongTensor(targets)


input_batch, target_batch = make_data(sentences, labels)  # input_batch: [num_sequence, sequence_length]
dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


# TextCNN
class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_embedding_dim)
        out_channel = 3
        self.conv = nn.Sequential(
            # conv: [in_channel(=1), out_channel, (filter_height, filter_width), stride=1]
            nn.Conv2d(in_channels=1, out_channels=out_channel, kernel_size=(2, word_embedding_dim), stride=1, padding=0),
            # [batch_size, out_channel, 2, 1]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        self.fc = nn.Linear(in_features=out_channel, out_features=num_class)

    def forward(self, input_data):
        # input_data: [batch_size, sequence_length]
        batch_size = input_data.shape[0]  # 动态获取batch_size，若batch_size一直为3，在预测的时候可能会报错
        input_data_embedding = self.word_embedding(input_data)  # [batch_size, sequence_length, word_embedding_dim]
        input_data_embedding = input_data_embedding.unsqueeze(1)  # 在第1个位置add channel(= 1), CNN需要通道数
        # input_data_embedding: [batch_size, channel(= 1), sequence_length, word_embedding_dim]
        conved = self.conv(input_data_embedding)  # [batch_size, out_channel, 1, 1]
        flatten = conved.view(batch_size, -1)  # [batch_size, out_channel*1*1]
        output = self.fc(flatten)  # [batch_size, num_class]
        return output


model = TextCNN()
loss_fn = nn.CrossEntropyLoss()
optim = optim.Adam(model.parameters(), lr=1e-3)


# Training
for epoch in range(5000):
    for input_data, target_data in loader:
        input_data, target_data = input_data.to(device), target_data.to(device)
        predict_data = model(input_data)
        loss = loss_fn(predict_data, target_data)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optim.zero_grad()
        loss.backward()
        optim.step()


# Test & Predict
test_text = 'i hate me'
tests = [[word2idx[n] for n in test_text.split()]]
test_batch = torch.LongTensor(tests).to(device)
model = model.eval()
predict = model(test_batch).data.max(1, keepdim=True)[1]
if predict[0][0] == 0:
    print(test_text, "is Bad Mean.")
else:
    print(test_text, "is Good Mean.")
