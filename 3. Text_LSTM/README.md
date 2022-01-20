# TextLSTM

类似于TextRNN

##

**1.1 作用：** 利用上文预测下文词。

##

**1.2 网络结构：** 三层神经网络，输入层、隐藏层、输出层。

***1.2.1 输入层：*** 输入矩阵 `input_data` 是one-hot编码，因此每一行的维度是词典去重后的长度 `[n_step, n_class]`。

	input_data: [batch_size, n_step, n_class]
和`torch.rnn`相同，应 `torch.LSTM` 的 `input_data: [seq_len, batch_size, word2vec]` 要求，进行变形：

	input_lstm = input_data.transpose(0, 1)  # [batch_size, n_step, n_class] --> [n_step, batch_size, n_class]
 
***1.2.2 隐藏层：*** 

	hidden_state = torch.zeros(1, batch_size, n_hidden)  # [num_layers * num_directions, batch_size, n_hidden]
    cell_state = torch.zeros(1, batch_size, n_hidden)  # [num_layers * num_directions, batch_size, n_hidden]
    outputs, (_, _) = self.lstm(input_lstm, (hidden_state, cell_state))

***1.2.3 输出层：*** 

	outputs = outputs[-1]  # [batch_size, n_hidden]
    TextLSTM_model = self.fc(outputs)

##

**1.3 代码实现：**

***1.3.1 导入包：***

	import torch
	import numpy as np
	import torch.nn as nn
	import torch.optim as optim
	import torch.utils.data as Data

***1.3.2 在 GPU 上运行：***

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	dtype = torch.FloatTensor

***1.3.3 构建语料库：***

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

***1.3.4 参数设置：***

	# TextLSTM Parameters
	n_step = len(seq_data[0]) - 1  # seq_data[0]: make
	n_hidden = 128

***1.3.5 构建 dataset 和 dataloader：***

	# dataset, dataloader
	def make_data(seq_data):
    	input_batch, target_batch = [], []
    	for seq in seq_data:
        	input = [word2idx[n] for n in seq[:-1]]  # 'm', 'a' , 'k' is input
        	target = word2idx[seq[-1]]  # 'e' is target
        	input_batch.append(np.eye(n_class)[input])
        	target_batch.append(target)
    	return torch.Tensor(input_batch), torch.LongTensor(target_batch)

	input_batch, target_batch = make_data(seq_data)
	dataset = Data.TensorDataset(input_batch, target_batch)
	loader = Data.DataLoader(dataset=dataset, batch_size=3, shuffle=True)

***1.3.6 构建一层LSTM网络结构：***

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

***1.3.7 迭代训练：***

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

***1.3.8 预测结果：***

	# Predict
	inputs = [sen[:3] for sen in seq_data]
	predict = model(input_batch).data.max(1, keepdim=True)[1]
	print(inputs, '-->', [idx2word[n.item()] for n in predict.squeeze()])

---

# Tips

1.代码**第18行**：

	char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']  # ['a', 'b', 'c', ...]

相当于 "**2. Text_RNN**" 中的：

	sentences = ['abcdefghijklmnopqrstuvwxyz']
	sentences_list = ' '.join(sentences).split()

##

2.“**参数设置**”中可以不设置 batch_size，然后在"**构建 dataset 和dataloader**"的时候设置，也就是：

	# TextLSTM Parameters
	n_step = len(seq_data[0]) - 1
	n_hidden = 128

	def make_data(seq_data):
		...

	input_batch, target_batch = make_data(seq_data)
	dataset = Data.TensorDataset(input_batch, target_batch)
	loader = Data.DataLoader(dataset=dataset, batch_size=3, shuffle=True)

相当于：

	# TextLSTM Parameters
	n_step = len(seq_data[0]) - 1  # seq_data[0]: make
	n_hidden = 128
	batch_size = 3

	def make_data(seq_data):
		...

	input_batch, target_batch = make_data(seq_data)
	dataset = Data.TensorDataset(input_batch, target_batch)
	loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

##

3.“**构建 dataset 和 dataloader**”时，input\_batch 和 output_batch 需要改为 Tensor 类型的，既可以在函数里面直接定义，也可以在实例化的时候再过一遍，也就是：

	# dataset, dataloader
	def make_data(seq_data):
		...
    	return torch.Tensor(input_batch), torch.LongTensor(target_batch)

	input_batch, target_batch = make_data(seq_data)
	dataset = Data.TensorDataset(input_batch, target_batch)
	loader = Data.DataLoader(dataset=dataset, batch_size=3, shuffle=True)

相当于：

	# dataset, dataloader
	def make_data(seq_data):
		...
    	return input_batch, target_batch

	input_batch, target_batch = make_data(seq_data)
	input_batch = torch.Tensor(input_batch)
	target_batch = torch.LongTensor(target_batch)
	dataset = Data.TensorDataset(input_batch, target_batch)
	loader = Data.DataLoader(dataset=dataset, batch_size=3, shuffle=True)

##

4.torch.LSTM、torch.RNN、torch.GRU都在 rnn.py 文件中，之后会将源码的分析放出。
