# Bi-LSTM(Attention)

Bi-LSTM + Attention

##

**1.1 作用：** 提高精度

##

**1.2 网络结构：** 词嵌入层、LSTM 层、Attention 层、输出层。

***1.2.1 词嵌入层：*** 输入矩阵 `input_data` 通过 `nn.Embedding` 进行词嵌入：

	def __init__(self):
		super(BiLSTM_Attention, self).__init__()
		self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_embedding_dim)
		...

	def forward(self, input_data):
		input_data_embedding = self.word_embedding(input_data)  # [batch_size, seq_len, word_embedding_dim]
		...
 
***1.2.2 LSTM 层：*** 

	def __init__(self):
		...
		self.lstm = nn.LSTM(batch_first=True, input_size=word_embedding_dim, hidden_size=n_hidden, bidirectional=True)
		...

	def forward(self, input_data):
		...
		output, (hidden_state, cell_state) = self.lstm(input_data_embedding)  # output: [batch_size, seq_len, n_hidden]
        # hidden_state, cell_state: [batch_size, num_layers*num_directions, n_hidden]
		...

***1.2.3 Attention 层：*** 将 LSTM 层的输出当作 Attention 层的输入

    def attention_layer(self, lstm_output, final_state):
        batch_size = len(lstm_output)
        hidden = final_state.view(batch_size, -1, 1)  # [batch_size, n_hidden*num_directions, n_hidden]
        atten_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # [batch_size, n_step]
        soft_atten_weights = F.softmax(atten_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_atten_weights.unsqueeze(2)).squeeze(2)
        return context, soft_atten_weights

	def forward(self, input_data):
		...
		return self.fc(atten_output), attention

***1.2.4 输出层：*** 

	def __init__(self):
		...
		self.fc = nn.Linear(in_features=n_hidden*2, out_features=num_classes)

	def forward(self, input_data):
		...
		return self.fc(atten_output), attention

##

**1.3 代码实现：**

***1.3.1 导入包：***

	import torch
	import numpy as np
	import torch.nn as nn
	import torch.optim as optim
	import torch.nn.functional as F
	import matplotlib.pyplot as plt
	import torch.utils.data as Data

***1.3.2 在 GPU 上运行：***

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	dtype = torch.FloatTensor

***1.3.3 构建语料库：***

	# Corpus
	sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
	labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.
	vocab = set(" ".join(sentences).split())
	word2idx = {word: idx for idx, word in enumerate(vocab)}
	vocab_size = len(word2idx)

***1.3.4 参数设置：***

	# Bi-LSTM(Attention) Parameters
	batch_size = 3
	word_embedding_dim = 2
	seq_len = len(sentences[0])
	n_hidden = 5  # number of hidden units in one cell
	num_classes = 2  # 0 or 1

***1.3.5 构建 dataset 和 dataloader：*** 需要TensorDataset

	# dataset, dataloader
	def make_data(sentences):
	    inputs, targets = [], []
	    for sen in sentences:
	        inputs.append(np.asarray([word2idx[n] for n in sen.split()]))
	    for out in labels:
	        targets.append(out)  # To using Torch Softmax Loss function
	    return torch.LongTensor(inputs), torch.LongTensor(targets)
	
	inputs, targets = make_data(sentences)
	dataset = Data.TensorDataset(inputs, targets)
	loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


***1.3.6 构建 Bi-LSTM (Attention) 网络结构：***

# Bi-LSTM(Attention)
	class BiLSTM_Attention(nn.Module):
	    def __init__(self):
	        super(BiLSTM_Attention, self).__init__()
	        self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_embedding_dim)
	        self.lstm = nn.LSTM(batch_first=True, input_size=word_embedding_dim, hidden_size=n_hidden, bidirectional=True)
	        self.fc = nn.Linear(in_features=n_hidden*2, out_features=num_classes)
	    # LSTM_output: [batch_size, n_step, n_hidden*num_directions(= 2)]
	
	    def attention_layer(self, lstm_output, final_state):
	        batch_size = len(lstm_output)
	        hidden = final_state.view(batch_size, -1, 1)  # [batch_size, n_hidden*num_directions, n_hidden]
	        atten_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # [batch_size, n_step]
	        soft_atten_weights = F.softmax(atten_weights, 1)
	        context = torch.bmm(lstm_output.transpose(1, 2), soft_atten_weights.unsqueeze(2)).squeeze(2)
	        return context, soft_atten_weights
	
	    def forward(self, input_data):
	        input_data_embedding = self.word_embedding(input_data)  # [batch_size, seq_len, word_embedding_dim]
	        output, (hidden_state, cell_state) = self.lstm(input_data_embedding)  # output: [batch_size, seq_len, n_hidden]
	        # hidden_state, cell_state: [batch_size, num_layers*num_directions, n_hidden]
	        atten_output, attention = self.attention_layer(lstm_output=output, final_state=hidden_state)
	        return self.fc(atten_output), attention  # model: [batch_size, num_classes],  attention: [batch_size, n_step]

	model = BiLSTM_Attention()
	loss_fn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

***1.3.7 迭代训练：***

	# Training
	for epoch in range(5000):
	    for input_batch, target_batch in loader:
	        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
	        predict_batch, attention = model(input_batch)
	        loss = loss_fn(predict_batch, target_batch)
	        if (epoch + 1) % 1000 == 0:
	            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
	
	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()

***1.3.8 预测结果：***

	# Test & Predict
	test_text = 'i hate me'
	tests = [np.asarray([word2idx[n] for n in test_text.split()])]
	test_batch = torch.LongTensor(tests).to(device)
	predict, _ = model(test_batch)
	predict = predict.data.max(1, keepdim=True)[1]
	if predict[0][0] == 0:
	    print(test_text, "is Bad Mean...")
	else:
	    print(test_text, "is Good Mean!!")

***1.3.9 Figure：***

	fig = plt.figure(figsize=(6, 3))  # [batch_size, n_step]
	ax = fig.add_subplot(1, 1, 1)
	ax.matshow(attention.cpu().data, cmap='viridis')
	ax.set_xticklabels(['']+['first_word', 'second_word', 'third_word'], fontdict={'fontsize': 14}, rotation=90)
	ax.set_yticklabels(['']+['batch_1', 'batch_2', 'batch_3', 'batch_4', 'batch_5', 'batch_6'], fontdict={'fontsize': 14})
	plt.show()
